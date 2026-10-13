"""API tests using FastAPI's TestClient (no server needed)."""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict(sample_reading):
    r = client.post("/predict", json=sample_reading)
    assert r.status_code == 200
    body = r.json()
    assert body["machine_id"] == "TEST-001"
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_batch_predict(sample_reading, risky_reading):
    r = client.post("/predict/batch", json={"readings": [sample_reading, risky_reading]})
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_rejects_invalid_machine_type(sample_reading):
    sample_reading["machine_type"] = "Z"
    assert client.post("/predict", json=sample_reading).status_code == 422


def test_rejects_impossible_sensor_value(sample_reading):
    sample_reading["air_temperature_k"] = -50
    assert client.post("/predict", json=sample_reading).status_code == 422


def test_rejects_missing_field(sample_reading):
    del sample_reading["torque_nm"]
    assert client.post("/predict", json=sample_reading).status_code == 422


def test_api_key_enforced(monkeypatch, sample_reading):
    monkeypatch.setenv("API_KEY", "secret123")
    assert client.post("/predict", json=sample_reading).status_code == 401
    ok = client.post("/predict", json=sample_reading, headers={"X-API-Key": "secret123"})
    assert ok.status_code == 200


def test_metrics_endpoint():
    client.post("/predict", json={
        "machine_type": "L", "air_temperature_k": 300, "process_temperature_k": 310,
        "rotational_speed_rpm": 1500, "torque_nm": 40, "tool_wear_min": 50,
    })
    r = client.get("/metrics/")
    assert r.status_code == 200
    assert "predictions_total" in r.text


# ---------------- web app ----------------
def test_web_app_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Predictive Maintenance Console" in r.text


def test_status_and_overview():
    status = client.get("/api/status").json()
    assert status["model_loaded"] is True
    overview = client.get("/api/overview").json()
    assert overview["model"]["test_metrics"]["f1"] > 0


def test_sample_data_endpoint():
    rows = client.get("/api/sample?n=5").json()
    assert len(rows) == 5
    assert "actual_failure" in rows[0]


def test_drift_endpoint_detects_shift():
    report = client.post("/api/drift", json={"source": "new_drifted"}).json()
    assert report["drift_detected"] is True


def test_registry_lists_champion():
    versions = client.get("/api/registry").json()
    assert any("champion" in v["aliases"] for v in versions)
