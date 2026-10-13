"""
Machine Failure Prediction - web application + REST API.

Core API
  GET  /health                - liveness check (Docker / Kubernetes)
  GET  /model/info            - model name, version, metrics
  POST /predict               - one sensor reading  -> prediction (real-time)
  POST /predict/batch         - up to 1000 readings -> predictions
  GET  /drift                 - drift check on readings this API has received
  POST /model/reload          - load a newly retrained model
  GET  /metrics/              - Prometheus metrics

Endpoints used by the web app (/api/...)
  GET  /api/status, /api/overview, /api/registry, /api/alerts, /api/predictions/recent, /api/sample
  POST /api/drift             - drift check on live / simulated production data
  POST /api/jobs/train        - run the full training pipeline in the background
  POST /api/jobs/retrain      - drift-triggered retraining in the background
  GET  /api/jobs/current      - progress + live logs of the running job

Security
  * Strict input validation (types + physical ranges) -> HTTP 422 on bad input
  * Optional API key: set env var API_KEY and clients must send header X-API-Key
  * Batch size limit, alert de-duplication, non-root Docker user
"""
import csv
import json
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from api.jobs import JobRunner
from src.alerts import send_alert
from src.predict import load_artifacts, predict_df, reload_artifacts
from src.utils import FEATURE_COLS, ROOT, get_logger, load_config, mlflow_tracking_uri, path

logger = get_logger("api")
cfg = load_config()
STATIC_DIR = ROOT / "api" / "static"
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://localhost:5000")
jobs = JobRunner()


def _model_loaded() -> bool:
    try:
        load_artifacts()
        return True
    except FileNotFoundError:
        return False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if _model_loaded():
        _, meta = load_artifacts()
        MODEL_VERSION.set(float(meta.get("model_version") or 0))
        logger.info("Loaded model %s v%s", meta["model_name"], meta.get("model_version"))
    else:
        logger.warning("No trained model yet - open the app and click 'Train model'")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Machine Failure Prediction API",
    description="Predictive maintenance: failure probability, health score, risk level and maintenance action. "
                "The web app is at [/](/).",
    version="2.0.0",
)

# ---------------- monitoring (Prometheus) ----------------
REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions made, by risk level", ["risk_level"])
FAILURE_PROB = Histogram("prediction_failure_probability", "Predicted failure probabilities",
                         buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
MODEL_VERSION = Gauge("model_version", "Currently loaded model version")
app.mount("/metrics", make_asgi_app())
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/reports", StaticFiles(directory=ROOT / "reports", check_dir=False), name="reports")

UNTRACKED_PREFIXES = ("/metrics", "/static", "/reports", "/api/jobs/current")


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    endpoint = request.url.path
    if not endpoint.startswith(UNTRACKED_PREFIXES):
        LATENCY.labels(endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(endpoint, request.method, response.status_code).inc()
    return response


# ---------------- security ----------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str | None = Security(api_key_header)):
    expected = os.getenv("API_KEY")
    if expected and not (key and secrets.compare_digest(key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------- schemas ----------------
class SensorReading(BaseModel):
    machine_id: str | None = Field(None, max_length=50, examples=["M14860"])
    machine_type: Literal["L", "M", "H"] = Field(..., description="L=low, M=medium, H=high quality variant")
    air_temperature_k: float = Field(..., ge=280, le=330, examples=[298.1])
    process_temperature_k: float = Field(..., ge=290, le=340, examples=[308.6])
    rotational_speed_rpm: float = Field(..., ge=500, le=4000, examples=[1551])
    torque_nm: float = Field(..., ge=0, le=120, examples=[42.8])
    tool_wear_min: float = Field(..., ge=0, le=400, examples=[108])


class BatchRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1, max_length=1000)


class DriftRequest(BaseModel):
    source: Literal["live", "new_normal", "new_drifted"] = "new_drifted"


class RetrainRequest(BaseModel):
    force: bool = False


# ---------------- prediction logging + alerting ----------------
_log_lock = threading.Lock()
_last_alert: dict[str, float] = {}
ALERT_COOLDOWN_S = 120  # don't alert about the same machine more than once per 2 minutes


def log_predictions(df: pd.DataFrame, results: list[dict]):
    log_file = path(cfg["paths"]["prediction_log"])
    rows = df[FEATURE_COLS].copy()
    rows.insert(0, "machine_id", df["machine_id"].fillna("").values)
    rows["failure_probability"] = [r["failure_probability"] for r in results]
    rows["risk_level"] = [r["risk_level"] for r in results]
    rows["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    with _log_lock:
        rows.to_csv(log_file, mode="a", index=False, header=not log_file.exists(), quoting=csv.QUOTE_MINIMAL)


def run_predictions(readings: list[SensorReading], batch: bool = False) -> list[dict]:
    if jobs.running and not _model_loaded():
        raise HTTPException(503, "Model is being trained. Try again when training finishes.")
    df = pd.DataFrame([r.model_dump() for r in readings])
    try:
        results = predict_df(df)
    except FileNotFoundError:
        raise HTTPException(503, "No trained model yet. Open the app and click 'Train model'.")

    critical = []
    for reading, res in zip(readings, results):
        res["machine_id"] = reading.machine_id
        PREDICTIONS.labels(res["risk_level"]).inc()
        FAILURE_PROB.observe(res["failure_probability"])
        if res["risk_level"] == "CRITICAL":
            critical.append(res)

    if critical and batch and len(results) > 20:  # big batch -> one summary alert instead of hundreds
        send_alert("CRITICAL_FAILURE_RISK", "critical", {
            "source": "batch", "critical_machines": len(critical), "batch_size": len(results),
            "examples": [c["machine_id"] for c in critical[:5]],
        })
    elif critical:
        now = time.time()
        for c in critical:
            key = c["machine_id"] or "unknown"
            if now - _last_alert.get(key, 0) > ALERT_COOLDOWN_S:
                _last_alert[key] = now
                send_alert("CRITICAL_FAILURE_RISK", "critical", {
                    "machine_id": c["machine_id"], "failure_probability": c["failure_probability"],
                    "top_factor": c["top_contributing_factors"][0]["feature"],
                })
    log_predictions(df, results)
    return results


# ---------------- core API ----------------
@app.get("/", include_in_schema=False)
def web_app():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    if not _model_loaded():
        raise HTTPException(503, detail={"status": "degraded", "model_loaded": False})
    return {"status": "ok", "model_loaded": True}


@app.get("/model/info")
def model_info():
    if not _model_loaded():
        raise HTTPException(503, "No trained model yet")
    _, meta = load_artifacts()
    return {k: meta[k] for k in ("model_name", "model_version", "trained_at", "trigger",
                                 "training_rows", "test_metrics", "feature_importance")}


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(reading: SensorReading):
    return run_predictions([reading])[0]


@app.post("/predict/batch", dependencies=[Depends(verify_api_key)])
def predict_batch(batch: BatchRequest):
    results = run_predictions(batch.readings, batch=True)
    risk_counts = pd.Series([r["risk_level"] for r in results]).value_counts().to_dict()
    return {"count": len(results), "risk_summary": risk_counts, "predictions": results}


@app.get("/drift", dependencies=[Depends(verify_api_key)])
def drift():
    from src.drift import run_drift_check
    if not path(cfg["paths"]["prediction_log"]).exists():
        raise HTTPException(404, "No predictions logged yet")
    return run_drift_check("logs")


def _refresh_model():
    _, meta = reload_artifacts()
    MODEL_VERSION.set(float(meta.get("model_version") or 0))


@app.post("/model/reload", dependencies=[Depends(verify_api_key)])
def reload_model():
    _refresh_model()
    _, meta = load_artifacts()
    return {"reloaded": True, "model_version": meta.get("model_version")}


# ---------------- web app endpoints ----------------
def _read_json(relative: str):
    p = ROOT / relative
    return json.loads(p.read_text()) if p.exists() else None


def _prediction_stats() -> dict:
    log_file = path(cfg["paths"]["prediction_log"])
    if not log_file.exists():
        return {"total": 0, "by_risk": {}, "avg_probability": None}
    df = pd.read_csv(log_file)
    return {
        "total": len(df),
        "by_risk": df["risk_level"].value_counts().to_dict(),
        "avg_probability": round(float(df["failure_probability"].mean()), 4) if len(df) else None,
    }


@app.get("/api/status")
def api_status():
    meta = load_artifacts()[1] if _model_loaded() else None
    keys = ("model_name", "model_version", "trained_at", "test_metrics")
    return {
        "model_loaded": meta is not None,
        "model": {k: meta.get(k) for k in keys} if meta else None,
        "job": {k: jobs.state[k] for k in ("kind", "status")},
        "api_key_required": bool(os.getenv("API_KEY")),
        "mlflow_ui_url": MLFLOW_UI_URL,
    }


@app.get("/api/overview")
def api_overview():
    meta = load_artifacts()[1] if _model_loaded() else None
    reports = cfg["paths"]["reports_dir"]
    alerts_file = path("logs/alerts.jsonl")
    return {
        "model": meta,
        "eda": _read_json(f"{reports}/eda_summary.json"),
        "validation": _read_json(f"{reports}/validation_report.json"),
        "last_drift": _read_json(f"{reports}/drift_report.json"),
        "last_retraining": _read_json(f"{reports}/retraining_result.json"),
        "predictions": _prediction_stats(),
        "alerts_total": sum(1 for _ in alerts_file.open(encoding="utf-8")) if alerts_file.exists() else 0,
    }


@app.get("/api/sample")
def api_sample(n: int = Query(200, ge=1, le=3000), drift: bool = False):
    """Synthetic sensor readings for demoing batch scoring (includes the actual outcome)."""
    from src.data_generation import generate_sensor_data
    df = generate_sensor_data(n, random_state=int(time.time()) % 100000, drift=drift)
    return df.rename(columns={"machine_failure": "actual_failure"}).to_dict(orient="records")


@app.get("/api/predictions/recent")
def api_recent_predictions(limit: int = Query(50, ge=1, le=500)):
    log_file = path(cfg["paths"]["prediction_log"])
    if not log_file.exists():
        return {"stats": _prediction_stats(), "rows": []}
    df = pd.read_csv(log_file).tail(limit).iloc[::-1].fillna("")
    return {"stats": _prediction_stats(), "rows": df.to_dict(orient="records")}


@app.get("/api/alerts")
def api_alerts(limit: int = Query(50, ge=1, le=500)):
    alerts_file = path("logs/alerts.jsonl")
    if not alerts_file.exists():
        return []
    lines = alerts_file.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in reversed(lines[-limit:])]


@app.get("/api/registry")
def api_registry():
    """Model versions from the MLflow Model Registry, newest first."""
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri())
        client = mlflow.MlflowClient()
        name = cfg["mlflow"]["registered_model_name"]
        aliases: dict[str, list[str]] = {}
        for alias, ver in (client.get_registered_model(name).aliases or {}).items():
            aliases.setdefault(str(ver), []).append(alias)
        versions = []
        for mv in client.search_model_versions(f"name='{name}'"):
            run = client.get_run(mv.run_id)
            versions.append({
                "version": int(mv.version),
                "aliases": aliases.get(str(mv.version), []),
                "created": datetime.fromtimestamp(mv.creation_timestamp / 1000, UTC).isoformat(timespec="seconds"),
                "model_type": run.data.tags.get("model_type"),
                "trigger": run.data.tags.get("trigger"),
                "metrics": {k.replace("test_", ""): round(v, 4) for k, v in run.data.metrics.items()
                            if k.startswith("test_")},
                "run_id": mv.run_id,
            })
        return sorted(versions, key=lambda v: -v["version"])
    except Exception as exc:  # registry is optional for the UI
        logger.warning("Registry lookup failed: %s", exc)
        return []


@app.post("/api/drift", dependencies=[Depends(verify_api_key)])
def api_drift(req: DriftRequest):
    from src.data_generation import generate_sensor_data
    from src.drift import run_drift_check

    if jobs.running:
        raise HTTPException(409, "Wait for the running job to finish")
    if not _model_loaded():
        raise HTTPException(503, "Train a model first")
    if req.source == "live":
        log_file = path(cfg["paths"]["prediction_log"])
        if not log_file.exists() or len(pd.read_csv(log_file)) < 50:
            raise HTTPException(400, "Need at least 50 live predictions. Run the plant floor simulation first.")
        report = run_drift_check("logs")
    else:
        df = generate_sensor_data(3000, random_state=int(time.time()) % 100000, drift=req.source == "new_drifted")
        df.to_csv(path(cfg["paths"]["new_data"]), index=False)
        report = run_drift_check("file")
    report["source"] = req.source
    return report


@app.post("/api/jobs/train", dependencies=[Depends(verify_api_key)])
def api_train():
    from src.pipeline import main as run_pipeline
    try:
        return jobs.start("train", run_pipeline, on_success=_refresh_model)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/jobs/retrain", dependencies=[Depends(verify_api_key)])
def api_retrain(req: RetrainRequest):
    from src.data_generation import generate_sensor_data
    from src.retrain import retrain

    if not _model_loaded():
        raise HTTPException(503, "Train a model first")
    new_data = path(cfg["paths"]["new_data"])

    def task():
        if not new_data.exists():
            generate_sensor_data(3000, random_state=7, drift=True).to_csv(new_data, index=False)
        return retrain(force=req.force)

    try:
        return jobs.start("retrain", task, on_success=_refresh_model)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/jobs/current")
def api_job():
    return jobs.snapshot()
