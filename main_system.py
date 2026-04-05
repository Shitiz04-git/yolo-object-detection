import threading
import subprocess
import time

def run_tracking():
    subprocess.run(["python", "tracking.py"])

def run_websocket():
    subprocess.run(["python", "websocket_server.py"])

def run_dashboard():
    subprocess.run(["streamlit", "run", "dashboard.py"])

def run_multicamera():
    subprocess.run(["python", "multi_camera.py"])

if __name__ == "__main__":
    # Choose which system to run
    print("Select system:")
    print("1. Object Tracking")
    print("2. WebSocket Server")
    print("3. Streamlit Dashboard")
    print("4. Multi-Camera System")
    print("5. All Systems")
    
    choice = input("Enter choice: ")
    
    if choice == "1":
        run_tracking()
    elif choice == "2":
        run_websocket()
    elif choice == "3":
        run_dashboard()
    elif choice == "4":
        run_multicamera()
    elif choice == "5":
        # Run all in separate threads
        threading.Thread(target=run_tracking).start()
        threading.Thread(target=run_websocket).start()
        threading.Thread(target=run_dashboard).start()
        threading.Thread(target=run_multicamera).start()
        
        print("All systems running!")
        print("- Tracking: GUI window")
        print("- WebSocket: ws://localhost:8765")
        print("- Dashboard: http://localhost:8501")
        print("- Multi-Camera: GUI window")
        
        # Keep main thread alive
        while True:
            time.sleep(1)