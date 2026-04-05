import cv2
from ultralytics import YOLO
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any
import hashlib
import base64

class FPSCounter:
    """FPS Counter class"""
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0
    
    def get_fps_str(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return f"FPS: {self.fps:.2f}"

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.fps_counter = FPSCounter()
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names  # COCO class names
                
    def detect_on_frame(self, frame):
        """Detect on single frame with enhanced results"""
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        annotated_frame = results[0].plot()
        fps_str = self.fps_counter.get_fps_str()
        cv2.putText(annotated_frame, fps_str, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return annotated_frame, results
    
    def detect_with_metadata(self, frame) -> Dict:
        """Detect and return structured metadata"""
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        # Extract detection metadata
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    'class': self.class_names[int(box.cls)],
                    'class_id': int(box.cls),
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist()[0],
                    'center': [(box.xyxy[0][0] + box.xyxy[0][2]) / 2,
                              (box.xyxy[0][1] + box.xyxy[0][3]) / 2]
                })
        
        return {
            'detections': detections,
            'total_objects': len(detections),
            'timestamp': time.time(),
            'fps': self.fps_counter.fps
        }
    
    def detect_on_image(self, image_path):
        """Detect on image file"""
        results = self.model(image_path, verbose=False, conf=self.conf_threshold)
        annotated_img = results[0].plot()
        fps_str = self.fps_counter.get_fps_str()
        cv2.putText(annotated_img, fps_str, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return annotated_img, results
    
    def detect_on_video(self, video_path):
        """Generator for video detection frames"""
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame, results = self.detect_on_frame(frame)
            yield annotated_frame, results
        cap.release()
    
    def get_detection_stats(self, results) -> Dict:
        """Generate statistics from detection results"""
        stats = {}
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_name = self.class_names[int(box.cls)]
                stats[class_name] = stats.get(class_name, 0) + 1
        return stats    