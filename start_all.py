"""
start_all.py — Helper script to launch both TRUEFRAME FastAPI AI Backend & Next.js Frontend.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
FRONTEND_DIR = ROOT_DIR / "kj 3" / "kji-saa-s-landing-page" / "brutalist-ai-saa-s-landing-page"

def main():
    print("=" * 65)
    print("  🚀 TRUEFRAME AI SYSTEM LAUNCHER")
    print("=" * 65)
    
    python_exe = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = sys.executable

    print("\n1️⃣ Starting FastAPI AI Backend (port 8000)...")
    backend_proc = subprocess.Popen(
        [str(python_exe), "api_server.py"],
        cwd=str(ROOT_DIR)
    )

    time.sleep(3)

    print("2️⃣ Starting Next.js Web Frontend (port 3000)...")
    frontend_proc = subprocess.Popen(
        ["npm.cmd" if sys.platform == "win32" else "npm", "run", "dev"],
        cwd=str(FRONTEND_DIR)
    )

    print("\n✅ Systems active!")
    print("   • Web Frontend: http://localhost:3000")
    print("   • API Backend:  http://localhost:8000")
    print("   • API Docs:     http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all servers.")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_proc.terminate()
        frontend_proc.terminate()

if __name__ == "__main__":
    main()
