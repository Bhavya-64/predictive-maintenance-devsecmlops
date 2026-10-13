"""
One-click launcher for the whole application.

    python run.py

Starts the web app (http://localhost:8000) and the MLflow tracking UI
(http://localhost:5000), then opens the app in your browser. If no model has been
trained yet, the app shows a "Train model" button - everything else is done in the browser.
Press Ctrl+C in this window to stop everything.
"""
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

from src.utils import ROOT

APP_URL = "http://localhost:8000"


def open_browser_when_ready():
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{APP_URL}/api/status", timeout=1)  # nosec B310
            webbrowser.open(APP_URL)
            return
        except Exception:
            time.sleep(0.5)


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    mlflow_log = open(ROOT / "logs" / "mlflow_ui.log", "w", encoding="utf-8")
    mlflow_ui = subprocess.Popen(  # nosec B603
        [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db",
         "--host", "127.0.0.1", "--port", "5000"],
        cwd=ROOT, stdout=mlflow_log, stderr=subprocess.STDOUT,
    )
    print(f"\n  Predictive Maintenance Console -> {APP_URL}")
    print("  MLflow tracking UI             -> http://localhost:5000")
    print("  Press Ctrl+C to stop.\n")
    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    try:
        import uvicorn
        uvicorn.run("api.main:app", host="127.0.0.1", port=8000, log_level="warning")
    finally:
        mlflow_ui.terminate()


if __name__ == "__main__":
    main()
