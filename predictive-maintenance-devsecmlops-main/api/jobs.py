"""
Background jobs for long tasks started from the web app (training, retraining).

Runs the task in a thread so the app stays responsive, and captures the
project's log lines so the UI can show live progress.
"""
import logging
import threading
from datetime import UTC, datetime

LOG_SOURCES = ("src", "__main__", "alerts")


class _Capture(logging.Handler):
    def __init__(self, sink: list):
        super().__init__(logging.INFO)
        self.sink = sink

    def emit(self, record):
        if record.name.startswith(LOG_SOURCES):
            self.sink.append({"time": datetime.now(UTC).strftime("%H:%M:%S"), "level": record.levelname,
                              "message": record.getMessage()})
            del self.sink[:-200]  # keep the last 200 lines


class JobRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self.state = {"kind": None, "status": "idle", "logs": [], "result": None, "error": None,
                      "started_at": None, "finished_at": None}

    @property
    def running(self) -> bool:
        return self.state["status"] == "running"

    def start(self, kind: str, fn, on_success=None) -> dict:
        with self._lock:
            if self.running:
                raise RuntimeError(f"A '{self.state['kind']}' job is already running")
            self.state = {"kind": kind, "status": "running", "logs": [], "result": None, "error": None,
                          "started_at": datetime.now(UTC).isoformat(timespec="seconds"), "finished_at": None}
        threading.Thread(target=self._run, args=(fn, on_success), daemon=True).start()
        return self.snapshot()

    def _run(self, fn, on_success):
        handler = _Capture(self.state["logs"])
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            result = fn()
            if on_success:
                on_success()
            self.state.update(status="done", result=result)
        except Exception as exc:  # report any failure to the UI
            self.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            root.removeHandler(handler)
            self.state["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    def snapshot(self) -> dict:
        return {**self.state, "logs": list(self.state["logs"])}
