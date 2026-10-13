"""
Data validation: checks the data BEFORE it reaches the model.

Checks: required columns, missing values, valid categories, physical sensor
ranges, valid target values, duplicates. Writes reports/validation_report.json.
Pipeline stops if a critical check fails (bad data in = bad model out).
"""
import json

import pandas as pd

from src.utils import FEATURE_COLS, TARGET_COL, get_logger, load_config, path

logger = get_logger(__name__)

# Physically plausible ranges for each sensor (anything outside = sensor fault / bad data)
SENSOR_RANGES = {
    "air_temperature_k": (280, 330),
    "process_temperature_k": (290, 340),
    "rotational_speed_rpm": (500, 4000),
    "torque_nm": (0, 120),
    "tool_wear_min": (0, 400),
}
VALID_MACHINE_TYPES = {"L", "M", "H"}


class DataValidationError(Exception):
    pass


def validate_data(df: pd.DataFrame, require_target: bool = True) -> dict:
    checks = {}
    required = FEATURE_COLS + ([TARGET_COL] if require_target else [])

    missing_cols = [c for c in required if c not in df.columns]
    checks["required_columns_present"] = {"passed": not missing_cols, "missing": missing_cols}
    if missing_cols:  # nothing else can be checked
        return _finish(checks, raise_on_fail=True)

    nulls = {c: int(n) for c, n in df[required].isna().sum().items() if n > 0}
    checks["no_missing_values"] = {"passed": not nulls, "null_counts": nulls}

    bad_types = sorted(set(df["machine_type"].dropna().unique()) - VALID_MACHINE_TYPES)
    checks["valid_machine_types"] = {"passed": not bad_types, "invalid_values": bad_types}

    out_of_range = {}
    for col, (lo, hi) in SENSOR_RANGES.items():
        n_bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_bad:
            out_of_range[col] = n_bad
    checks["sensor_values_in_range"] = {"passed": not out_of_range, "out_of_range_counts": out_of_range}

    if require_target:
        bad_target = sorted(set(df[TARGET_COL].dropna().unique()) - {0, 1})
        rate = float(df[TARGET_COL].mean())
        checks["valid_target"] = {"passed": not bad_target and 0 < rate < 1, "failure_rate": round(rate, 4)}

    dupes = int(df.duplicated().sum())
    checks["duplicates"] = {"passed": True, "count": dupes, "note": "warning only, removed during ingestion"}

    return _finish(checks, raise_on_fail=True)


def _finish(checks: dict, raise_on_fail: bool) -> dict:
    report = {"all_passed": all(c["passed"] for c in checks.values()), "checks": checks}
    cfg = load_config()
    out = path(f"{cfg['paths']['reports_dir']}/validation_report.json")
    out.write_text(json.dumps(report, indent=2))

    if report["all_passed"]:
        logger.info("Data validation passed (%d checks)", len(checks))
    else:
        failed = [k for k, v in checks.items() if not v["passed"]]
        logger.error("Data validation FAILED: %s", failed)
        if raise_on_fail:
            raise DataValidationError(f"Validation failed: {failed}. See {out}")
    return report
