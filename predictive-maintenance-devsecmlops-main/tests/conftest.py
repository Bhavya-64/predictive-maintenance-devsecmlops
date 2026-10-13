"""Shared test setup: make sure a trained model exists before API/prediction tests run."""
import pytest

from src.utils import load_config, path


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    cfg = load_config()
    if not path(cfg["paths"]["model"]).exists():
        from src.pipeline import main
        main()


@pytest.fixture
def sample_reading():
    return {
        "machine_id": "TEST-001",
        "machine_type": "M",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 10,
    }


@pytest.fixture
def risky_reading():
    # very worn tool under high torque -> overstrain territory
    return {
        "machine_id": "TEST-002",
        "machine_type": "L",
        "air_temperature_k": 302.0,
        "process_temperature_k": 310.5,
        "rotational_speed_rpm": 1250,
        "torque_nm": 65.0,
        "tool_wear_min": 230,
    }
