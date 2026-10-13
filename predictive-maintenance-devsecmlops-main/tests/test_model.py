"""Tests for the trained model, prediction logic and drift detection."""
import pandas as pd

from src.data_generation import generate_sensor_data
from src.drift import detect_drift, psi
from src.predict import load_artifacts, predict_one, risk_level
from src.utils import FEATURE_COLS, load_config, path


def test_model_meets_minimum_quality():
    _, meta = load_artifacts()
    assert meta["test_metrics"]["f1"] >= 0.5        # quality gate - CI fails if model gets worse
    assert meta["test_metrics"]["roc_auc"] >= 0.85


def test_prediction_output_fields(sample_reading):
    result = predict_one(sample_reading)
    for key in ("machine_failure", "failure_probability", "health_score", "risk_level",
                "recommended_action", "maintenance_urgency", "top_contributing_factors"):
        assert key in result
    assert 0 <= result["failure_probability"] <= 1
    assert 0 <= result["health_score"] <= 100
    assert len(result["top_contributing_factors"]) == 3


def test_risky_machine_scores_worse_than_healthy(sample_reading, risky_reading):
    healthy = predict_one(sample_reading)
    risky = predict_one(risky_reading)
    assert risky["failure_probability"] > healthy["failure_probability"]
    assert risky["health_score"] < healthy["health_score"]


def test_risk_levels():
    assert risk_level(0.05) == "LOW"
    assert risk_level(0.3) == "MEDIUM"
    assert risk_level(0.6) == "HIGH"
    assert risk_level(0.95) == "CRITICAL"


def test_psi_zero_for_identical_data():
    s = pd.Series(range(1000)).values
    assert psi(s, s) < 0.01


def test_no_drift_on_similar_data():
    cfg = load_config()
    ref = pd.read_csv(path(cfg["paths"]["reference_data"]))
    current = generate_sensor_data(2000, random_state=99)[FEATURE_COLS]
    report = detect_drift(ref, current)
    assert not report["sensor_drift"]["dataset_drift"]


def test_drift_detected_on_shifted_data():
    cfg = load_config()
    ref = pd.read_csv(path(cfg["paths"]["reference_data"]))
    current = generate_sensor_data(2000, random_state=99, drift=True)[FEATURE_COLS]
    report = detect_drift(ref, current)
    assert report["sensor_drift"]["dataset_drift"]
    assert "air_temperature_k" in report["sensor_drift"]["drifted_features"]
