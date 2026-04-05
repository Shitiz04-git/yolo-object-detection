import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import defaultdict

class ObjectTracker:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.track_history = defaultdict(list)  # Stores track IDs and their paths
        self.colors = {}  # Stores random colors for each track ID
        self.next_id = 0
        
    def get_color(self, track_id):
        """Generate consistent color for each track ID"""
        if track_id not in self.colors:
            self.colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.colors[track_id]
    
    def track_frame(self, frame, persist=True):
        """Track objects in frame and return annotated frame with trails"""
        # Run tracking
        results = self.model.track(frame, persist=persist, verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get tracking data
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy().tolist()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Update track history
            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                self.track_history[track_id].append(center)
                
                # Keep only last 30 points for trail
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
                
                # Draw bounding box
                color = self.get_color(track_id)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), 
                             (int(box[2]), int(box[3])), color, 2)
                
                # Draw label with track ID
                label = f"ID:{track_id} {self.model.names[int(cls)]} {conf:.2f}"
                cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw movement trail
                points = self.track_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(frame, (int(points[i-1][0]), int(points[i-1][1])),
                            (int(points[i][0]), int(points[i][1])), color, 2)
        
        # Add FPS counter
        fps_str = f"Tracking FPS: {self.get_fps()}"
        cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame, results
    
    def get_fps(self):
        """Calculate FPS"""
        if not hasattr(self, 'prev_time'):
            self.prev_time = time.time()
            return 0
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return round(fps, 2)
    
    def get_tracking_stats(self):
        """Get statistics about tracked objects"""
        stats = {
            'active_tracks': len(self.track_history),
            'track_ids': list(self.track_history.keys()),
            'total_movement': sum(len(v) for v in self.track_history.values())
        }
        return stats
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.track_history.clear()
        self.colors.clear()

# Example usage
if __name__ == "__main__":
    tracker = ObjectTracker()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated_frame, results = tracker.track_frame(frame)
        cv2.imshow('Object Tracking', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()