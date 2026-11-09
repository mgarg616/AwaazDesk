"""
AwaazDesk - Server Orchestrator
Launches FastAPI backend (port 8000) and Streamlit frontend (port 8501) concurrently.
"""
import subprocess
import sys
import time
import os
import signal
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run():
    print("=" * 70)
    print("🎙️ Starting AwaazDesk – AI Voice Grievance Intelligence System")
    print("=" * 70)

    # 1. Start FastAPI Backend
    print("[1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--log-level", "info"
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(BASE_DIR))

    # Give backend a moment to initialize database
    time.sleep(2)

    # 2. Start Streamlit Frontend
    print("[2/2] Launching Streamlit Frontend on http://127.0.0.1:8501 ...")
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(BASE_DIR))

    print("\n✅ AwaazDesk is running!")
    print("👉 Frontend UI: http://127.0.0.1:8501")
    print("👉 Backend API: http://127.0.0.1:8000")
    print("👉 API Docs:    http://127.0.0.1:8000/docs")
    print("\nPress Ctrl+C to shut down both servers.\n")

    def signal_handler(sig, frame):
        print("\nStopping AwaazDesk servers...")
        frontend_proc.terminate()
        backend_proc.terminate()
        frontend_proc.wait()
        backend_proc.wait()
        print("Servers stopped cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(1)
            # Check if any process terminated unexpectedly
            if backend_proc.poll() is not None:
                print("Backend terminated unexpectedly!")
                break
            if frontend_proc.poll() is not None:
                print("Frontend terminated unexpectedly!")
                break
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    run()
