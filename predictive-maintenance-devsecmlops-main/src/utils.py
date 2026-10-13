"""Shared helpers: config loading and logging."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Column names used across the whole project
SENSOR_COLS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
CATEGORICAL_COLS = ["machine_type"]
FEATURE_COLS = CATEGORICAL_COLS + SENSOR_COLS
TARGET_COL = "machine_failure"


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mlflow_tracking_uri() -> str:
    """Absolute SQLite URI so MLflow works no matter which folder you run from."""
    return load_config()["mlflow"]["tracking_uri"].replace("sqlite:///", f"sqlite:///{ROOT.as_posix()}/")


def path(relative: str) -> Path:
    """Turn a config path into an absolute path and make sure its folder exists."""
    p = ROOT / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to the console and to logs/app.log (rotated at 1 MB)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_file = path("logs/app.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger
