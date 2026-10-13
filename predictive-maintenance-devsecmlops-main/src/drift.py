"""
Drift detection: is new production data different from what the model was trained on?

Sensor (data) drift      - compares each input sensor's distribution:
                             * PSI (Population Stability Index): <0.1 stable, 0.1-0.2 moderate, >0.2 significant
                             * KS test (Kolmogorov-Smirnov): p-value < 0.05 = distributions differ
                           A feature is "drifted" if PSI > threshold. Dataset drift if
                           more than `drift_share_threshold` of features drifted.
Prediction drift         - compares the distribution of predicted failure probabilities
                           (are we suddenly predicting far more / fewer failures?).

Usage:
  python -m src.drift                          # compares data/raw/new_sensor_data.csv
  python -m src.drift --source logs            # compares live API predictions log
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.utils import SENSOR_COLS, get_logger, load_config, path

logger = get_logger(__name__)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric samples."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference) | set(current))
    ref_pct = reference.value_counts(normalize=True).reindex(cats, fill_value=0).clip(1e-6)
    cur_pct = current.value_counts(normalize=True).reindex(cats, fill_value=0).clip(1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, current_probs=None) -> dict:
    cfg = load_config()["drift"]
    features = {}

    for col in SENSOR_COLS:
        score = psi(reference[col].values, current[col].values)
        p_value = float(ks_2samp(reference[col], current[col]).pvalue)
        features[col] = {
            "psi": round(score, 4),
            "ks_p_value": round(p_value, 6),
            "reference_mean": round(float(reference[col].mean()), 2),
            "current_mean": round(float(current[col].mean()), 2),
            "drifted": score > cfg["psi_threshold"],
        }
    cat_score = categorical_psi(reference["machine_type"], current["machine_type"])
    features["machine_type"] = {"psi": round(cat_score, 4), "drifted": cat_score > cfg["psi_threshold"]}

    n_drifted = sum(f["drifted"] for f in features.values())
    share = n_drifted / len(features)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "sensor_drift": {
            "dataset_drift": share > cfg["drift_share_threshold"],
            "drifted_features": [k for k, v in features.items() if v["drifted"]],
            "drift_share": round(share, 3),
            "features": features,
        },
    }

    if current_probs is not None and "failure_probability" in reference:
        ref_p = reference["failure_probability"].values
        cur_p = np.asarray(current_probs)
        p_psi = psi(ref_p, cur_p)
        report["prediction_drift"] = {
            "psi": round(p_psi, 4),
            "drifted": p_psi > cfg["psi_threshold"],
            "reference_predicted_failure_rate": round(float((ref_p >= 0.5).mean()), 4),
            "current_predicted_failure_rate": round(float((cur_p >= 0.5).mean()), 4),
        }

    report["drift_detected"] = report["sensor_drift"]["dataset_drift"] or \
        report.get("prediction_drift", {}).get("drifted", False)
    return report


def run_drift_check(source: str = "file") -> dict:
    """Load reference + current data, run checks, save report, fire alert if drifted."""
    from src.alerts import send_alert
    from src.predict import load_artifacts

    cfg = load_config()
    reference = pd.read_csv(path(cfg["paths"]["reference_data"]))

    if source == "logs":
        current = pd.read_csv(path(cfg["paths"]["prediction_log"]))
        probs = current["failure_probability"].values
    else:
        current = pd.read_csv(path(cfg["paths"]["new_data"]))
        model, _ = load_artifacts()
        probs = model.predict_proba(current[reference.columns.drop("failure_probability")])[:, 1]

    report = detect_drift(reference, current, probs)
    out = path(f"{cfg['paths']['reports_dir']}/drift_report.json")
    out.write_text(json.dumps(report, indent=2))

    if report["drift_detected"]:
        send_alert("DRIFT_DETECTED", "warning", {
            "drifted_features": report["sensor_drift"]["drifted_features"],
            "prediction_drift": report.get("prediction_drift"),
        })
    logger.info("Drift check (%s): drift_detected=%s | drifted features=%s | report -> %s",
                source, report["drift_detected"], report["sensor_drift"]["drifted_features"], out)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["file", "logs"], default="file")
    run_drift_check(parser.parse_args().source)
