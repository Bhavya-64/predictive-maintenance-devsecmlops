"""
Turns a raw model probability into decisions a maintenance team can act on:

  failure_probability   -> model output (0-1)
  machine_failure       -> 0 / 1 (threshold 0.5)
  health_score          -> 100 = perfectly healthy, 0 = about to fail
  risk_level            -> LOW / MEDIUM / HIGH / CRITICAL
  recommended_action    -> what to do
  maintenance_urgency   -> how soon
  top_contributing_factors -> which sensor readings pushed the risk up

How "top contributing factors" works (simple + explainable):
  for every sensor / engineered feature we compute how unusual the reading is
  (|z-score| vs training data) and multiply it by how important that feature is
  to the model. The 3 highest scores are the main risk drivers.
"""
import json
from functools import lru_cache

import joblib
import pandas as pd

from src.features import add_engineered_features
from src.utils import FEATURE_COLS, load_config, path

ACTIONS = {
    "LOW": ("Continue normal operation; follow routine maintenance schedule", "Routine (next scheduled maintenance)"),
    "MEDIUM": ("Schedule preventive inspection; check tool wear and cooling", "Within 7 days"),
    "HIGH": ("Plan maintenance at next shift change; prepare spare parts", "Within 24 hours"),
    "CRITICAL": ("Stop the machine and perform immediate maintenance", "Immediate"),
}

FRIENDLY_NAMES = {
    "air_temperature_k": "Air temperature",
    "process_temperature_k": "Process temperature",
    "rotational_speed_rpm": "Rotational speed",
    "torque_nm": "Torque",
    "tool_wear_min": "Tool wear",
    "temp_diff_k": "Process-air temperature gap",
    "power_w": "Mechanical power",
    "wear_strain": "Tool wear x torque (strain)",
    "wear_per_torque": "Tool wear per unit torque",
}


@lru_cache(maxsize=1)
def load_artifacts():
    cfg = load_config()
    model = joblib.load(path(cfg["paths"]["model"]))
    metadata = json.loads(path(cfg["paths"]["model_metadata"]).read_text())
    return model, metadata


def reload_artifacts():
    """Call after retraining so the API picks up the new model."""
    load_artifacts.cache_clear()
    return load_artifacts()


def risk_level(prob: float) -> str:
    t = load_config()["risk_thresholds"]
    if prob >= t["critical"]:
        return "CRITICAL"
    if prob >= t["high"]:
        return "HIGH"
    if prob >= t["medium"]:
        return "MEDIUM"
    return "LOW"


def top_factors(row: pd.Series, metadata: dict, k: int = 3) -> list[dict]:
    scores = []
    for feat, importance in metadata["feature_importance"].items():
        stats = metadata["feature_stats"][feat]
        z = (row[feat] - stats["mean"]) / (stats["std"] or 1)
        scores.append({
            "feature": FRIENDLY_NAMES.get(feat, feat),
            "value": round(float(row[feat]), 2),
            "deviation": "above normal" if z > 0 else "below normal",
            "z_score": round(float(z), 2),
            "impact_score": round(abs(z) * importance, 4),
        })
    return sorted(scores, key=lambda s: -s["impact_score"])[:k]


def predict_df(df: pd.DataFrame) -> list[dict]:
    model, metadata = load_artifacts()
    X = df[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]
    engineered = add_engineered_features(X)

    results = []
    for i, prob in enumerate(probs):
        prob = float(prob)
        level = risk_level(prob)
        action, urgency = ACTIONS[level]
        results.append({
            "machine_failure": int(prob >= 0.5),
            "failure_probability": round(prob, 4),
            "health_score": round(100 * (1 - prob), 1),
            "risk_level": level,
            "recommended_action": action,
            "maintenance_urgency": urgency,
            "top_contributing_factors": top_factors(engineered.iloc[i], metadata),
            "model_version": metadata.get("model_version"),
        })
    return results


def predict_one(reading: dict) -> dict:
    return predict_df(pd.DataFrame([reading]))[0]
