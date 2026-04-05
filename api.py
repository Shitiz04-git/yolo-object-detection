from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
import uuid
from datetime import datetime
from detector import YOLODetector
import threading
import hashlib
import hmac

app = Flask(__name__)
CORS(app)

# Initialize detector
detector = YOLODetector()

# Simple API key authentication (for demo)
API_KEYS = {
    'production_key_123': 'write',
    'readonly_key_456': 'read'
}

def verify_api_key():
    """Verify API key from request headers"""
    api_key = request.headers.get('X-API-Key')
    if not api_key or api_key not in API_KEYS:
        return False
    return True

def rate_limit_check(api_key):
    """Simple rate limiting"""
    # Store request counts per API key
    # For production, use Redis or database
    return True

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'YOLOv8',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect/frame', methods=['POST'])
def detect_frame():
    """Detect objects in uploaded frame"""
    if not verify_api_key():
        return jsonify({'error': 'Invalid API key'}), 401
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform detection with metadata
    annotated_frame, results = detector.detect_on_frame(frame)
    metadata = detector.detect_with_metadata(frame)
    
    # Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'detections': metadata['detections'],
        'total_objects': metadata['total_objects'],
        'annotated_image': image_base64,
        'timestamp': metadata['timestamp']
    })

@app.route('/api/detect/url', methods=['POST'])
def detect_from_url():
    """Detect objects from image URL"""
    if not verify_api_key():
        return jsonify({'error': 'Invalid API key'}), 401
    
    data = request.get_json()
    if 'url' not in data:
        return jsonify({'error': 'URL required'}), 400
    
    # Download image from URL
    import requests
    response = requests.get(data['url'])
    nparr = np.frombuffer(response.content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    metadata = detector.detect_with_metadata(frame)
    
    return jsonify({
        'success': True,
        'detections': metadata['detections'],
        'total_objects': metadata['total_objects']
    })

@app.route('/api/detect/stream', methods=['POST'])
def detect_stream():
    """Stream detection endpoint (WebSocket would be better)"""
    # For production, implement WebSocket streaming
    return jsonify({'message': 'WebSocket endpoint for streaming'})

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model': 'YOLOv8',
        'classes': list(detector.class_names.values()),
        'num_classes': len(detector.class_names),
        'confidence_threshold': detector.conf_threshold
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)