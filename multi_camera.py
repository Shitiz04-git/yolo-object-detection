import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import json

class CameraThread:
    """Individual camera thread"""
    def __init__(self, camera_id, url=None, name=None):
        self.camera_id = camera_id
        self.url = url if url else camera_id
        self.name = name if name else f"Camera_{camera_id}"
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.last_frame = None
        self.fps = 0
        self.stats = {'total_frames': 0, 'dropped_frames': 0}
        
    def start(self):
        """Start camera capture thread"""
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            print(f"Failed to open {self.name}")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def _capture_loop(self):
        """Capture frames in loop"""
        prev_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Failed to read from {self.name}")
                break
            
            # Calculate FPS
            current_time = time.time()
            self.fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Add to queue (drop if full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.stats['dropped_frames'] += 1
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put_nowait(frame)
                self.stats['total_frames'] += 1
            except queue.Full:
                pass
            
            self.last_frame = frame
        
        self.cap.release()
    
    def get_frame(self):
        """Get latest frame from queue"""
        try:
            frame = self.frame_queue.get_nowait()
            return frame
        except queue.Empty:
            return self.last_frame
    
    def stop(self):
        """Stop camera thread"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
    
    def get_stats(self):
        """Get camera statistics"""
        return {
            'name': self.name,
            'camera_id': self.camera_id,
            'fps': round(self.fps, 2),
            'total_frames': self.stats['total_frames'],
            'dropped_frames': self.stats['dropped_frames'],
            'is_running': self.running
        }

class MultiCameraDetector:
    """Multi-camera object detection system"""
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.cameras = {}
        self.detection_results = {}
        self.running = False
        
    def add_camera(self, camera_id, url=None, name=None):
        """Add a camera to the system"""
        camera = CameraThread(camera_id, url, name)
        self.cameras[camera_id] = camera
        print(f"Added camera: {camera.name}")
        return camera
    
    def add_cameras_batch(self, camera_configs):
        """Add multiple cameras from configuration"""
        for config in camera_configs:
            self.add_camera(
                config['id'],
                config.get('url'),
                config.get('name')
            )
    
    def start_all(self):
        """Start all cameras"""
        for camera_id, camera in self.cameras.items():
            if camera.start():
                print(f"Started {camera.name}")
            else:
                print(f"Failed to start {camera.name}")
        self.running = True
    
    def detect_frame(self, frame):
        """Run detection on single frame"""
        results = self.model(frame, verbose=False)
        detections = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    'class': self.model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist()[0]
                })
        
        annotated = results[0].plot()
        return annotated, detections
    
    def process_all_cameras(self):
        """Process all cameras and run detection"""
        results = {}
        
        for camera_id, camera in self.cameras.items():
            frame = camera.get_frame()
            if frame is not None:
                annotated, detections = self.detect_frame(frame)
                results[camera_id] = {
                    'frame': annotated,
                    'detections': detections,
                    'camera_stats': camera.get_stats(),
                    'timestamp': datetime.now().isoformat()
                }
                self.detection_results[camera_id] = results[camera_id]
        
        return results
    
    def create_grid_display(self, max_cols=2):
        """Create grid display of all camera feeds"""
        frames = []
        labels = []
        
        for camera_id, camera in self.cameras.items():
            frame = camera.get_frame()
            if frame is not None:
                # Add detection overlay if available
                if camera_id in self.detection_results:
                    detections = self.detection_results[camera_id]['detections']
                    for det in detections:
                        bbox = det['bbox']
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                                    (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        label = f"{det['class']}: {det['confidence']:.2f}"
                        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1])-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add camera name
                cv2.putText(frame, camera.name, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add FPS
                fps_text = f"FPS: {camera.fps:.1f}"
                cv2.putText(frame, fps_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                frames.append(frame)
        
        # Create grid
        if frames:
            rows = (len(frames) + max_cols - 1) // max_cols
            grid_height = max(frame.shape[0] for frame in frames)
            grid_width = max(frame.shape[1] for frame in frames)
            
            grid = np.zeros((grid_height * rows, grid_width * max_cols, 3), dtype=np.uint8)
            
            for idx, frame in enumerate(frames):
                row = idx // max_cols
                col = idx % max_cols
                h, w = frame.shape[:2]
                grid[row*grid_height:row*grid_height+h, 
                     col*grid_width:col*grid_width+w] = frame
            
            return grid
        
        return None
    
    def stop_all(self):
        """Stop all cameras"""
        for camera in self.cameras.values():
            camera.stop()
        self.running = False
        print("All cameras stopped")
    
    def get_system_stats(self):
        """Get overall system statistics"""
        total_frames = sum(c.stats['total_frames'] for c in self.cameras.values())
        total_dropped = sum(c.stats['dropped_frames'] for c in self.cameras.values())
        
        return {
            'total_cameras': len(self.cameras),
            'active_cameras': sum(1 for c in self.cameras.values() if c.running),
            'total_frames_processed': total_frames,
            'total_frames_dropped': total_dropped,
            'drop_rate': total_dropped / total_frames if total_frames > 0 else 0,
            'cameras': {cid: c.get_stats() for cid, c in self.cameras.items()}
        }

# Configuration examples
CAMERA_CONFIGS = [
    {'id': 0, 'name': 'Front Door', 'url': 0},
    {'id': 1, 'name': 'Back Door', 'url': 1},
    # {'id': 2, 'name': 'IP Camera 1', 'url': 'http://192.168.1.100:8080/video'},
    # {'id': 3, 'name': 'IP Camera 2', 'url': 'rtsp://192.168.1.101:554/stream'},
]

# Multi-camera GUI using tkinter
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class MultiCameraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Camera Object Detection System")
        self.root.geometry("1400x800")
        
        self.detector = MultiCameraDetector()
        self.setup_ui()
        
    def setup_ui(self):
        # Control panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="Start All Cameras", 
                 command=self.start_cameras, bg="green", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Stop All Cameras", 
                 command=self.stop_cameras, bg="red", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Add Camera", 
                 command=self.add_camera_dialog, bg="blue", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Save Stats", 
                 command=self.save_stats, bg="orange").pack(side=tk.LEFT, padx=5)
        
        # Camera grid display
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='black')
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Stats panel
        self.stats_text = tk.Text(self.root, height=8, width=50)
        self.stats_text.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update loop
        self.update_display()
    
    def start_cameras(self):
        # Add default cameras
        for config in CAMERA_CONFIGS:
            self.detector.add_camera(config['id'], config.get('url'), config.get('name'))
        
        self.detector.start_all()
        self.status_var.set(f"Started {len(self.detector.cameras)} cameras")
    
    def stop_cameras(self):
        self.detector.stop_all()
        self.status_var.set("All cameras stopped")
    
    def add_camera_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Camera")
        dialog.geometry("400x250")
        
        tk.Label(dialog, text="Camera ID:").pack(pady=5)
        camera_id_entry = tk.Entry(dialog)
        camera_id_entry.pack(pady=5)
        
        tk.Label(dialog, text="Camera Name:").pack(pady=5)
        name_entry = tk.Entry(dialog)
        name_entry.pack(pady=5)
        
        tk.Label(dialog, text="URL (optional):").pack(pady=5)
        url_entry = tk.Entry(dialog)
        url_entry.pack(pady=5)
        
        def add():
            camera_id = int(camera_id_entry.get())
            name = name_entry.get()
            url = url_entry.get() or camera_id
            self.detector.add_camera(camera_id, url, name)
            dialog.destroy()
            self.status_var.set(f"Added camera: {name}")
        
        tk.Button(dialog, text="Add", command=add).pack(pady=10)
    
    def update_display(self):
        if self.detector.running:
            # Process detections
            results = self.detector.process_all_cameras()
            
            # Create grid display
            grid = self.detector.create_grid_display(max_cols=2)
            
            if grid is not None:
                # Resize for display
                h, w = grid.shape[:2]
                scale = min(800/h, 1200/w)
                new_w, new_h = int(w * scale), int(h * scale)
                grid = cv2.resize(grid, (new_w, new_h))
                
                # Convert to PhotoImage
                grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(grid_rgb)
                photo = ImageTk.PhotoImage(img)
                
                self.canvas.config(width=new_w, height=new_h)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo
            
            # Update stats
            stats = self.detector.get_system_stats()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, json.dumps(stats, indent=2))
            self.stats_text.see(tk.END)
            
            self.status_var.set(f"Active: {stats['active_cameras']}/{stats['total_cameras']} cameras")
        
        # Schedule next update
        self.root.after(100, self.update_display)
    
    def save_stats(self):
        stats = self.detector.get_system_stats()
        filename = f"camera_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        self.status_var.set(f"Stats saved to {filename}")

if __name__ == "__main__":
    # Option 1: Run GUI
    root = tk.Tk()
    app = MultiCameraGUI(root)
    root.mainloop()
    
    # Option 2: Run headless
    # detector = MultiCameraDetector()
    # detector.add_cameras_batch(CAMERA_CONFIGS)
    # detector.start_all()
    # 
    # while True:
    #     results = detector.process_all_cameras()
    #     grid = detector.create_grid_display()
    #     if grid is not None:
    #         cv2.imshow('Multi-Camera Detection', grid)
    #     
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # 
    # detector.stop_all()
    # cv2.destroyAllWindows()