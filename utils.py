import time
import json
import os
import cv2
from PIL import Image
from datetime import datetime
from pathlib import Path

class FPSCounter:
    def __init__(self):
        self.prev_time = 0
        self.fps = 0

    def update(self):
        current_time = time.time()
        if current_time - self.prev_time > 0:
            self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return self.fps

    def get_fps_str(self):
        return f"FPS: {int(self.fps)}"

def save_detection_result(image, results, output_dir="outputs"):
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save annotated image
    cv2.imwrite(filepath, image)
    
    # Save JSON metadata
    json_filename = f"detection_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)
    metadata = {
        "timestamp": timestamp,
        "num_detections": len(results),
        "detections": []
    }
    for r in results:
        metadata["detections"].append({
            "class": r.names[int(r.cls)],
            "confidence": float(r.conf),
            "bbox": r.xyxy.tolist()
        })
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    return filepath, json_path

def get_output_path(filename):
    return os.path.join("outputs", os.path.basename(filename))

