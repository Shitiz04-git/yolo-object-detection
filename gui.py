import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import threading
from PIL import Image, ImageTk
from detector import YOLODetector
from utils import save_detection_result
import os

class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Object Detection System")
        self.root.geometry("1200x800")

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")

        self.detector = YOLODetector()
        self.cap = None
        self.video_thread = None
        self.is_running = False
        self.current_image = None
        self.current_results = None

        self.setup_ui()

    def setup_ui(self):
        # Video/Image display canvas
        self.canvas = tk.Label(self.root, bg='black')
        self.canvas.pack(pady=10, expand=True, fill=tk.BOTH)

        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Webcam On", command=self.start_webcam, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Webcam Off", command=self.stop_capture, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Image", command=self.load_image, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Video", command=self.load_video, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save Result", command=self.save_result, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear", command=self.clear_canvas, width=15).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def start_webcam(self):
        if self.is_running:
            return

        # Try different camera indices
        for i in range(3):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                print(f"Camera {i} started")
                break
            else:
                self.cap.release()
                self.cap = None
        
        if self.cap is None:
            messagebox.showerror("Error", "No webcam found!")
            return

        self.is_running = True
        self.video_thread = threading.Thread(target=self.update_webcam)
        self.video_thread.daemon = True
        self.video_thread.start()
        self.status_var.set("Webcam started")

    def update_webcam(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            annotated, results = self.detector.detect_on_frame(frame)
            self.current_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            self.current_results = results
            self.root.after(0, self.update_canvas)
        self.cap.release()

    def stop_capture(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.status_var.set("Capture stopped")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            annotated, results = self.detector.detect_on_image(path)
            self.current_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            self.current_results = results
            self.update_canvas()
            self.status_var.set(f"Image loaded: {os.path.basename(path)}")

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.status_var.set("Video playback started")
            # Run video in separate thread
            video_thread = threading.Thread(target=self.play_video, args=(path,))
            video_thread.daemon = True
            video_thread.start()

    def play_video(self, path):
        """Play video in separate thread"""
        for annotated_frame, results in self.detector.detect_on_video(path):
            if not self.is_running:
                break
            self.current_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            self.current_results = results
            self.root.after(0, self.update_canvas)
            cv2.waitKey(30)  # ~30 FPS

    def save_result(self):
        if self.current_image is not None and self.current_results is not None:
            img_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            filepath, jsonpath = save_detection_result(img_bgr, self.current_results)
            messagebox.showinfo("Saved", f"Results saved:\n{filepath}\n{jsonpath}")
        else:
            messagebox.showwarning("Warning", "No detection result to save")

    def clear_canvas(self):
        self.canvas.config(image='')
        self.current_image = None
        self.current_results = None
        self.status_var.set("Canvas cleared")

    def update_canvas(self):
        if self.current_image is not None:
            img = Image.fromarray(self.current_image)
            # Resize to fit canvas
            img.thumbnail((900, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.canvas.config(image=photo)
            self.canvas.image = photo  # Keep reference

# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionGUI(root)
    root.mainloop()