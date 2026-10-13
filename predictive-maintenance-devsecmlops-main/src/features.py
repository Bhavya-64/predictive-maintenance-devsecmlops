"""
Sensor preprocessing + feature engineering.

Engineered features (each maps to a real failure mechanism):
  temp_diff_k     = process temp - air temp      -> heat dissipation problems
  power_w         = torque x angular speed       -> power failures (too low / too high)
  wear_strain     = tool wear x torque           -> overstrain failures
  wear_per_torque = tool wear / torque           -> tool wear relative to load

Everything is wrapped in ONE sklearn Pipeline (features -> scaling/encoding -> model),
so the API can send raw sensor readings and get the exact same transformations
that were used during training (no training/serving skew).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.utils import CATEGORICAL_COLS, SENSOR_COLS

ENGINEERED_COLS = ["temp_diff_k", "power_w", "wear_strain", "wear_per_torque"]
NUMERIC_COLS = SENSOR_COLS + ENGINEERED_COLS


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_diff_k"] = df["process_temperature_k"] - df["air_temperature_k"]
    df["power_w"] = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60
    df["wear_strain"] = df["tool_wear_min"] * df["torque_nm"]
    df["wear_per_torque"] = df["tool_wear_min"] / df["torque_nm"].clip(lower=1)
    return df


def build_preprocessor() -> Pipeline:
    column_transform = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])
    return Pipeline([
        ("feature_engineering", FunctionTransformer(add_engineered_features)),
        ("transform", column_transform),
    ])
