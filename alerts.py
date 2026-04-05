import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import os
import cv2
import numpy as np  # ✅ YEH LINE IMPORTANT HAI
from ultralytics import YOLO

# Try to import Twilio (optional, for SMS)
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("Twilio not installed. SMS alerts disabled. Install with: pip install twilio")

class AlertSystem:
    def __init__(self, config_file='alert_config.json'):
        self.load_config(config_file)
        self.alert_history = []
        
    def load_config(self, config_file):
        """Load alert configuration"""
        default_config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': 'your_email@gmail.com',
                'sender_password': 'your_app_password',
                'recipient_emails': ['admin@example.com']
            },
            'sms': {
                'enabled': False,
                'account_sid': 'your_twilio_sid',
                'auth_token': 'your_twilio_token',
                'from_number': '+1234567890',
                'to_numbers': ['+9876543210']
            },
            'webhook': {
                'enabled': False,
                'url': 'https://your-webhook.com/alert',
                'headers': {'Content-Type': 'application/json'}
            },
            'rules': [
                {
                    'object_type': 'person',
                    'min_confidence': 0.8,
                    'min_count': 1,
                    'cooldown_seconds': 60
                },
                {
                    'object_type': 'car',
                    'min_confidence': 0.7,
                    'min_count': 3,
                    'cooldown_seconds': 30
                }
            ]
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"Created config file: {config_file}")
    
    def send_email_alert(self, subject, body, image_path=None):
        """Send email alert"""
        if not self.config['email']['enabled']:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['sender_email']
            msg['To'] = ', '.join(self.config['email']['recipient_emails'])
            msg['Subject'] = f"[Object Detection Alert] {subject}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image if provided
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    img_attachment = MIMEText(img_data, 'base64', 'utf-8')
                    img_attachment.add_header('Content-Disposition', 
                                             f'attachment; filename={os.path.basename(image_path)}')
                    msg.attach(img_attachment)
            
            server = smtplib.SMTP(self.config['email']['smtp_server'], 
                                 self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['sender_email'], 
                        self.config['email']['sender_password'])
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Email alert sent: {subject}")
            return True
            
        except Exception as e:
            print(f"❌ Email alert failed: {e}")
            return False
    
    def send_sms_alert(self, message):
        """Send SMS alert using Twilio"""
        if not self.config['sms']['enabled']:
            return False
        
        if not TWILIO_AVAILABLE:
            print("❌ Twilio not available. Install with: pip install twilio")
            return False
        
        try:
            client = Client(self.config['sms']['account_sid'], 
                          self.config['sms']['auth_token'])
            
            for to_number in self.config['sms']['to_numbers']:
                client.messages.create(
                    body=message,
                    from_=self.config['sms']['from_number'],
                    to=to_number
                )
            
            print(f"✅ SMS alert sent: {message[:50]}...")
            return True
            
        except Exception as e:
            print(f"❌ SMS alert failed: {e}")
            return False
    
    def send_webhook_alert(self, data):
        """Send webhook alert to external service"""
        if not self.config['webhook']['enabled']:
            return False
        
        try:
            response = requests.post(
                self.config['webhook']['url'],
                headers=self.config['webhook']['headers'],
                json=data,
                timeout=5
            )
            if response.status_code == 200:
                print("✅ Webhook alert sent")
                return True
            else:
                print(f"❌ Webhook failed: {response.status_code}")
                return False
            
        except Exception as e:
            print(f"❌ Webhook alert failed: {e}")
            return False
    
    def should_send_alert(self, object_type, confidence, count):
        """Check if alert should be sent based on rules"""
        for rule in self.config['rules']:
            if (rule['object_type'] == object_type and 
                confidence >= rule['min_confidence'] and 
                count >= rule['min_count']):
                
                # Check cooldown
                last_alert = self.get_last_alert_time(object_type)
                if last_alert:
                    cooldown = rule.get('cooldown_seconds', 60)
                    if (datetime.now() - last_alert).seconds < cooldown:
                        return False
                
                return True
        return False
    
    def get_last_alert_time(self, object_type):
        """Get last alert time for object type"""
        for alert in reversed(self.alert_history):
            if alert['object_type'] == object_type:
                return alert['timestamp']
        return None
    
    def check_and_alert(self, detections, frame=None, save_image=True):
        """Main method to check detections and send alerts"""
        alerts_sent = []
        
        # Group detections by type
        detection_counts = {}
        for det in detections:
            obj_type = det['class']
            confidence = det['confidence']
            detection_counts[obj_type] = detection_counts.get(obj_type, 0) + 1
        
        # Check each detection type
        for obj_type, count in detection_counts.items():
            # Calculate average confidence for this object type
            confidences = [d['confidence'] for d in detections if d['class'] == obj_type]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            if self.should_send_alert(obj_type, avg_confidence, count):
                # Save image if needed
                image_path = None
                if save_image and frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"alert_{obj_type}_{timestamp}.jpg"
                    cv2.imwrite(image_path, frame)
                    print(f"📸 Alert image saved: {image_path}")
                
                # Prepare alert message
                subject = f"{obj_type.upper()} Detected!"
                body = f"""
Alert Type: Object Detection
Object: {obj_type}
Count: {count}
Average Confidence: {avg_confidence:.1%}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Details:
- Total detections in frame: {len(detections)}
- Multiple objects detected simultaneously

Action Required: Please check the attached image.
"""
                
                # Send alerts
                if self.config['email']['enabled']:
                    self.send_email_alert(subject, body, image_path)
                
                if self.config['sms']['enabled']:
                    sms_msg = f"Alert: {count} {obj_type}(s) detected with {avg_confidence:.0%} confidence"
                    self.send_sms_alert(sms_msg)
                
                # Webhook data
                webhook_data = {
                    'event': 'object_detection',
                    'object_type': obj_type,
                    'count': count,
                    'confidence': avg_confidence,
                    'timestamp': datetime.now().isoformat(),
                    'total_detections': len(detections)
                }
                self.send_webhook_alert(webhook_data)
                
                # Log alert
                alert_record = {
                    'object_type': obj_type,
                    'count': count,
                    'confidence': avg_confidence,
                    'timestamp': datetime.now(),
                    'image_path': image_path
                }
                self.alert_history.append(alert_record)
                alerts_sent.append(alert_record)
                print(f"🚨 ALERT TRIGGERED: {count}x {obj_type} (conf: {avg_confidence:.1%})")
        
        return alerts_sent
    
    def get_alert_history(self, hours=24):
        """Get recent alert history"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history 
                if alert['timestamp'] > cutoff]
    
    def print_summary(self):
        """Print alert system summary"""
        print("\n" + "="*50)
        print("📊 ALERT SYSTEM SUMMARY")
        print("="*50)
        print(f"Total alerts sent: {len(self.alert_history)}")
        print(f"Email enabled: {self.config['email']['enabled']}")
        print(f"SMS enabled: {self.config['sms']['enabled']}")
        print(f"Webhook enabled: {self.config['webhook']['enabled']}")
        
        if self.alert_history:
            print("\nRecent alerts:")
            for alert in self.alert_history[-5:]:
                print(f"  - {alert['timestamp'].strftime('%H:%M:%S')}: {alert['count']}x {alert['object_type']}")
        print("="*50)

# Alert-enabled detector class
class AlertEnabledDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.alert_system = AlertSystem()
        print("✅ Alert-enabled detector initialized")
        
    def detect_and_alert(self, frame):
        """Run detection and send alerts"""
        results = self.model(frame, verbose=False)
        detections = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detections.append({
                    'class': self.model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist()[0]
                })
        
        # Check and send alerts
        alerts = self.alert_system.check_and_alert(detections, frame)
        
        annotated = results[0].plot()
        return annotated, detections, alerts

def create_alert_config():
    """Create alert configuration file"""
    config = {
        'email': {
            'enabled': False,  # Change to True to enable
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'your_email@gmail.com',
            'sender_password': 'your_app_password',  # Gmail app password
            'recipient_emails': ['admin@example.com']
        },
        'sms': {
            'enabled': False,  # Enable only if you have Twilio
            'account_sid': 'ACxxxxxxxxxxxxxx',
            'auth_token': 'xxxxxxxxxxxxxxxx',
            'from_number': '+1234567890',
            'to_numbers': ['+9876543210']
        },
        'webhook': {
            'enabled': False,
            'url': 'https://your-webhook.com/alert',
            'headers': {'Content-Type': 'application/json'}
        },
        'rules': [
            {
                'object_type': 'person',
                'min_confidence': 0.7,
                'min_count': 1,
                'cooldown_seconds': 30
            },
            {
                'object_type': 'car',
                'min_confidence': 0.6,
                'min_count': 2,
                'cooldown_seconds': 20
            },
            {
                'object_type': 'traffic light',
                'min_confidence': 0.8,
                'min_count': 1,
                'cooldown_seconds': 10
            }
        ]
    }
    
    with open('alert_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("✅ Alert configuration created: alert_config.json")
    print("📝 Please update with your actual credentials before enabling!")
    return config

# Simple test without any external dependencies
def test_alert_system():
    """Test alert system without email/SMS"""
    print("\n🔔 TESTING ALERT SYSTEM (No actual emails will be sent)")
    print("-" * 40)
    
    alert_system = AlertSystem()
    
    # Test detections
    test_detections = [
        {'class': 'person', 'confidence': 0.85},
        {'class': 'person', 'confidence': 0.78},
        {'class': 'car', 'confidence': 0.92},
        {'class': 'dog', 'confidence': 0.65}
    ]
    
    print(f"Test detections: {len(test_detections)} objects")
    alerts = alert_system.check_and_alert(test_detections)
    
    if alerts:
        print(f"\n✅ Alert system working! {len(alerts)} alerts triggered.")
    else:
        print("\n⚠️ No alerts triggered (rules not matched or cooldown active)")
    
    alert_system.print_summary()
    
    return alert_system

# Real-time webcam with alerts
def run_alert_webcam():
    """Run webcam detection with alerts"""
    print("\n🎥 Starting Webcam with Alerts")
    print("Press 'q' to quit")
    print("-" * 40)
    
    detector = AlertEnabledDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection with alerts
        annotated, detections, alerts = detector.detect_and_alert(frame)
        
        # Display alert count on frame
        if alerts:
            cv2.putText(annotated, f"ALERTS: {len(alerts)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show FPS
        cv2.putText(annotated, "Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Alert-Enabled Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    detector.alert_system.print_summary()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚨 OBJECT DETECTION ALERT SYSTEM")
    print("="*50)
    
    print("\nSelect option:")
    print("1. Create config file only")
    print("2. Test alert system (no email/SMS)")
    print("3. Run webcam with alerts")
    print("4. Run all tests")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == "1":
        create_alert_config()
        
    elif choice == "2":
        test_alert_system()
        
    elif choice == "3":
        # Check if numpy is installed
        try:
            import numpy as np
            print("✅ NumPy available")
        except ImportError:
            print("❌ Installing numpy...")
            import subprocess
            subprocess.run(["pip", "install", "numpy"])
            import numpy as np
        
        run_alert_webcam()
        
    elif choice == "4":
        print("\n🧪 Running all tests...\n")
        create_alert_config()
        test_alert_system()
        
        print("\n✅ All tests completed!")
        
    else:
        print("Invalid choice")