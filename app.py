"""
app.py — Entry point for CAREER GUIDANCE AI
Launches: streamlit run web_app/dashboard.py
"""

import sys
import os
import subprocess

# ── Base directory (folder chứa file app.py này) ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Path tới dashboard ────────────────────────────────────────────────────────
DASHBOARD_PATH = os.path.join(BASE_DIR, "web_app", "dashboard.py")


def check_dashboard_exists():
    if not os.path.exists(DASHBOARD_PATH):
        print(f"[ERROR] Khong tim thay dashboard: {DASHBOARD_PATH}")
        sys.exit(1)


def run_streamlit():
    """Chạy Streamlit dashboard dưới dạng subprocess."""
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        DASHBOARD_PATH,
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
        "--server.headless", "false",       # tự mở trình duyệt
    ]

    print("[INFO] Khoi dong Streamlit ...")
    print(f"[INFO] Dashboard : {DASHBOARD_PATH}")
    print(f"[INFO] URL        : http://localhost:8501")
    print("[INFO] Nhan Ctrl+C de dung.\n")

    try:
        process = subprocess.run(cmd, cwd=BASE_DIR)
        sys.exit(process.returncode)
    except KeyboardInterrupt:
        print("\n[INFO] Da dung ung dung.")
        sys.exit(0)
    except FileNotFoundError:
        print("[ERROR] Streamlit chua duoc cai dat.")
        print("        Chay: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    check_dashboard_exists()
    run_streamlit()
