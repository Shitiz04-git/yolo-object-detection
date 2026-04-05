import asyncio
import websockets
import cv2
import base64
import json
import numpy as np
from ultralytics import YOLO
import threading
from datetime import datetime

class WebSocketDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.connected_clients = set()
        self.streaming = False
        
    async def register(self, websocket):
        """Register new client"""
        self.connected_clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.connected_clients)}")
        
    async def unregister(self, websocket):
        """Unregister client"""
        self.connected_clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(self.connected_clients)}")
    
    async def send_frame(self, websocket, frame, detections):
        """Send frame and detection data to client"""
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create message
        message = {
            'type': 'frame',
            'timestamp': datetime.now().isoformat(),
            'image': frame_base64,
            'detections': detections,
            'total_objects': len(detections)
        }
        
        await websocket.send(json.dumps(message))
    
    async def handle_client(self, websocket, path):
        """Handle individual client connection"""
        await self.register(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get('command') == 'start':
                    self.streaming = True
                elif data.get('command') == 'stop':
                    self.streaming = False
                elif data.get('command') == 'status':
                    await websocket.send(json.dumps({
                        'type': 'status',
                        'streaming': self.streaming,
                        'clients': len(self.connected_clients)
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    def detect_frame(self, frame):
        """Run detection on frame"""
        results = self.model(frame, verbose=False)
        
        # Extract detections
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
    
    async def broadcast_frames(self):
        """Broadcast frames to all connected clients"""
        cap = cv2.VideoCapture(0)
        
        while True:
            if self.streaming and self.connected_clients:
                ret, frame = cap.read()
                if ret:
                    annotated, detections = self.detect_frame(frame)
                    
                    # Send to all connected clients
                    await asyncio.gather(
                        *[self.send_frame(client, annotated, detections) 
                          for client in self.connected_clients]
                    )
            
            await asyncio.sleep(0.033)  # ~30 FPS
        
        cap.release()

async def main():
    detector = WebSocketDetector()
    
    # Start broadcasting task
    broadcast_task = asyncio.create_task(detector.broadcast_frames())
    
    # Start WebSocket server
    async with websockets.serve(detector.handle_client, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())