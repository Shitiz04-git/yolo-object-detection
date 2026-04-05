import subprocess
import threading
import time
import webbrowser
import os

def run_command(cmd, name):
    """Run command in subprocess"""
    print(f"\n🚀 Starting {name}...")
    return subprocess.Popen(cmd, shell=True)

def main():
    print("=" * 50)
    print("🎯 OBJECT DETECTION SYSTEM - LAUNCHER")
    print("=" * 50)
    
    print("\nSelect mode:")
    print("1. Run Only Tracking (Recommended for beginners)")
    print("2. Run Dashboard (Web interface)")
    print("3. Run Multi-Camera")
    print("4. Run WebSocket Server")
    print("5. RUN EVERYTHING (Advanced)")
    
    choice = input("\nEnter choice (1-5): ")
    
    if choice == "1":
        print("\n▶️ Starting Object Tracking...")
        subprocess.run("python tracking.py", shell=True)
    
    elif choice == "2":
        print("\n▶️ Starting Dashboard...")
        print("📊 Opening browser at http://localhost:8501")
        webbrowser.open("http://localhost:8501")
        subprocess.run("streamlit run dashboard.py", shell=True)
    
    elif choice == "3":
        print("\n▶️ Starting Multi-Camera System...")
        subprocess.run("python multi_camera.py", shell=True)
    
    elif choice == "4":
        print("\n▶️ Starting WebSocket Server...")
        print("🌐 Server running at ws://localhost:8765")
        print("📁 Open websocket_client.html in browser")
        subprocess.run("python websocket_server.py", shell=True)
    
    elif choice == "5":
        print("\n▶️ Starting ALL systems...")
        print("⚠️ This will open multiple windows!")
        
        # Start all systems
        processes = []
        
        # Tracking
        p1 = run_command("python tracking.py", "Tracking")
        processes.append(p1)
        
        # Dashboard
        p2 = run_command("streamlit run dashboard.py", "Dashboard")
        processes.append(p2)
        
        # WebSocket
        p3 = run_command("python websocket_server.py", "WebSocket")
        processes.append(p3)
        
        # Open browser for dashboard
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
        
        print("\n✅ All systems started!")
        print("\n📍 Access points:")
        print("   - Tracking: GUI Window")
        print("   - Dashboard: http://localhost:8501")
        print("   - WebSocket: ws://localhost:8765")
        print("\n❌ Press Ctrl+C to stop all\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all systems...")
            for p in processes:
                p.terminate()
            print("✅ All systems stopped")

if __name__ == "__main__":
    main()