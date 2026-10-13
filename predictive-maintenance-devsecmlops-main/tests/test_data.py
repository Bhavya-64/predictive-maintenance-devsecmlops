"""Tests for data generation, validation and feature engineering."""
import pytest

from src.data_generation import generate_sensor_data
from src.data_validation import DataValidationError, validate_data
from src.features import ENGINEERED_COLS, add_engineered_features
from src.utils import FEATURE_COLS, TARGET_COL


@pytest.fixture
def data():
    return generate_sensor_data(n_samples=500, random_state=1)


def test_generated_data_shape_and_columns(data):
    assert len(data) == 500
    for col in FEATURE_COLS + [TARGET_COL]:
        assert col in data.columns


def test_generated_data_has_both_classes(data):
    assert set(data[TARGET_COL].unique()) == {0, 1}
    assert 0.01 < data[TARGET_COL].mean() < 0.3  # failures are rare but present


def test_generation_is_reproducible():
    a = generate_sensor_data(100, random_state=5)
    b = generate_sensor_data(100, random_state=5)
    assert a.equals(b)


def test_validation_passes_on_clean_data(data):
    assert validate_data(data)["all_passed"]


def test_validation_catches_missing_values(data):
    data.loc[0, "torque_nm"] = None
    with pytest.raises(DataValidationError):
        validate_data(data)


def test_validation_catches_out_of_range_sensor(data):
    data.loc[0, "air_temperature_k"] = 900  # impossible reading
    with pytest.raises(DataValidationError):
        validate_data(data)


def test_validation_catches_bad_category(data):
    data.loc[0, "machine_type"] = "X"
    with pytest.raises(DataValidationError):
        validate_data(data)


def test_feature_engineering(data):
    out = add_engineered_features(data)
    for col in ENGINEERED_COLS:
        assert col in out.columns
        assert out[col].notna().all()
    row = out.iloc[0]
    assert row["temp_diff_k"] == pytest.approx(row["process_temperature_k"] - row["air_temperature_k"])
