# Advanced YOLOv8 Object Detection System

## Features
- Real-time webcam detection
- Image and video file detection
- Intuitive GUI with buttons for all functions
- Save annotated results (PNG + JSON metadata)
- Live FPS counter
- Modular Python code

## Setup & Run (Final Year BCA Project)
1. Install dependencies: `pip install -r requirements.txt` (or use venv)
2. Run: `python main.py`
3. On first run, YOLOv8 model (yolov8n.pt) auto-downloads (~6MB).

## Usage
- **Webcam**: Click "Webcam On" for live detection (Off to stop).
- **Image**: "Load Image" → detects and displays.
- **Video**: "Load Video" → plays with detections.
- **Save**: Saves current frame + bbox/classes/conf to `outputs/`.
- **Clear**: Resets canvas.

## Dependencies
- ultralytics (YOLOv8)
- opencv-python
- numpy, pillow
- Python 3.8+

## Notes
- FPS depends on hardware (nano model for speed).
- Webcam index 0 (change in gui.py if needed).
- Outputs in `outputs/` folder.

Perfect for major project demo!

