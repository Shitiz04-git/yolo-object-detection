import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from collections import deque
import threading

# Page configuration
st.set_page_config(
    page_title="Object Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

class DashboardDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.detection_history = deque(maxlen=100)
        self.class_counts = {}
        
    def detect_frame(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_name = self.model.names[int(box.cls)]
                confidence = float(box.conf)
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                })
                
                # Update class counts
                self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
        
        annotated = results[0].plot()
        self.detection_history.append({
            'timestamp': datetime.now(),
            'count': len(detections),
            'detections': detections
        })
        
        return annotated, detections

# Initialize detector
@st.cache_resource
def get_detector():
    return DashboardDetector()

detector = get_detector()

# Sidebar
st.sidebar.title("🎯 Control Panel")
st.sidebar.markdown("---")

# Camera selection
camera_source = st.sidebar.selectbox("Camera Source", ["Webcam (0)", "Webcam (1)", "IP Camera"])
if camera_source == "Webcam (0)":
    camera_id = 0
elif camera_source == "Webcam (1)":
    camera_id = 1
else:
    camera_id = st.sidebar.text_input("Enter IP Camera URL", "http://192.168.1.100:8080/video")

# Confidence threshold
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Detection settings
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)
show_labels = st.sidebar.checkbox("Show Labels", True)

# Main dashboard
st.title("🎯 Real-Time Object Detection Dashboard")
st.markdown("---")

# Metrics row
col1, col2, col3, col4 = st.columns(4)

metrics_placeholder = st.empty()
camera_placeholder = st.empty()
chart_placeholder = st.empty()
detection_table_placeholder = st.empty()

# Start/Stop buttons
col1, col2, col3 = st.columns(3)
with col1:
    start_button = st.button("▶️ Start Detection", use_container_width=True)
with col2:
    stop_button = st.button("⏹️ Stop Detection", use_container_width=True)
with col3:
    clear_button = st.button("🗑️ Clear History", use_container_width=True)

# State management
if 'running' not in st.session_state:
    st.session_state.running = False
    st.session_state.cap = None

if clear_button:
    detector.detection_history.clear()
    detector.class_counts.clear()
    st.success("History cleared!")

if start_button and not st.session_state.running:
    st.session_state.cap = cv2.VideoCapture(camera_id if isinstance(camera_id, int) else camera_id)
    st.session_state.running = True

if stop_button:
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None

# Main loop
if st.session_state.running and st.session_state.cap:
    frame_placeholder = st.empty()
    
    while st.session_state.running:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break
        
        # Run detection
        annotated, detections = detector.detect_frame(frame)
        
        # Convert BGR to RGB for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Display frame
        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
        
        # Update metrics
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate metrics
            total_detections = sum(len(h['detections']) for h in detector.detection_history)
            avg_confidence = np.mean([d['confidence'] 
                                     for h in detector.detection_history 
                                     for d in h['detections']]) if total_detections > 0 else 0
            
            unique_classes = len(set([d['class'] 
                                     for h in detector.detection_history 
                                     for d in h['detections']]))
            
            current_fps = len(detector.detection_history) / 5 if len(detector.detection_history) > 0 else 0
            
            col1.metric("Total Detections", total_detections)
            col2.metric("Avg Confidence", f"{avg_confidence:.1%}")
            col3.metric("Unique Classes", unique_classes)
            col4.metric("Current FPS", f"{current_fps:.1f}")
        
        # Update charts
        with chart_placeholder.container():
            tab1, tab2 = st.tabs(["📊 Detection Trend", "📈 Class Distribution"])
            
            with tab1:
                if len(detector.detection_history) > 0:
                    df = pd.DataFrame([
                        {'Time': h['timestamp'], 'Count': h['count']}
                        for h in detector.detection_history
                    ])
                    fig = px.line(df, x='Time', y='Count', title='Objects Detected Over Time')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if detector.class_counts:
                    df_classes = pd.DataFrame([
                        {'Class': k, 'Count': v} 
                        for k, v in detector.class_counts.items()
                    ]).sort_values('Count', ascending=False).head(10)
                    
                    fig = px.bar(df_classes, x='Class', y='Count', 
                                title='Top 10 Detected Objects',
                                color='Count', color_continuous_scale='Viridis')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Update detection table
        with detection_table_placeholder.container():
            if detections:
                df_detections = pd.DataFrame(detections)
                df_detections['confidence'] = df_detections['confidence'].apply(lambda x: f"{x:.1%}")
                st.dataframe(df_detections, use_container_width=True)
        
        time.sleep(0.033)  # ~30 FPS
    
    if st.session_state.cap:
        st.session_state.cap.release()

elif not st.session_state.running:
    st.info("Click 'Start Detection' to begin")

# Footer
st.markdown("---")
st.markdown("### 📊 Session Statistics")
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Detections in Session", 
              sum(len(h['detections']) for h in detector.detection_history))
with col2:
    st.metric("Most Detected Object", 
              max(detector.class_counts, key=detector.class_counts.get) if detector.class_counts else "None")