"""
Batch prediction: score a whole CSV of sensor readings offline (no API needed).
Typical use: a nightly job that scores every machine in the plant.

Usage:
  python -m scripts.batch_predict --input data/raw/new_sensor_data.csv --output reports/batch_predictions.csv
"""
import argparse

import pandas as pd

from src.data_validation import validate_data
from src.predict import predict_df
from src.utils import get_logger, path

logger = get_logger("batch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/new_sensor_data.csv")
    parser.add_argument("--output", default="reports/batch_predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(path(args.input))
    validate_data(df, require_target=False)
    results = predict_df(df)

    out = df.copy()
    for key in ("machine_failure", "failure_probability", "health_score", "risk_level",
                "recommended_action", "maintenance_urgency"):
        out[f"pred_{key}"] = [r[key] for r in results]
    out["pred_top_factor"] = [r["top_contributing_factors"][0]["feature"] for r in results]
    out = out.sort_values("pred_failure_probability", ascending=False)  # riskiest machines first
    out.to_csv(path(args.output), index=False)

    logger.info("Scored %d machines -> %s", len(out), args.output)
    logger.info("Risk summary: %s", out["pred_risk_level"].value_counts().to_dict())


if __name__ == "__main__":
    main()
