"""
Synthetic industrial sensor data generator.

Modelled on the public AI4I 2020 Predictive Maintenance dataset (UCI). A machine
fails if ANY of these physical failure modes happens:

  TWF  Tool wear failure      - tool is worn out (200-240 min)
  HDF  Heat dissipation fail  - air/process temp gap too small AND low speed
  PWF  Power failure          - power (torque x speed) too low or too high
  OSF  Overstrain failure     - tool wear x torque above a limit for the product type
  RNF  Random failure         - 0.1% chance, unexplained

Failures are decided on the TRUE sensor values, then small measurement noise is
added to what the "sensors" report. That makes the ML problem realistic
(not perfectly separable).

`drift=True` simulates a changed production environment (hotter factory,
older tools, heavier loads) - used to demo drift detection and retraining.
"""
import argparse

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, path

logger = get_logger(__name__)


def generate_sensor_data(n_samples: int = 10000, random_state: int = 42, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    machine_type = rng.choice(["L", "M", "H"], size=n_samples, p=[0.5, 0.3, 0.2])

    air_temp = rng.normal(300.0, 2.0, n_samples)
    torque = np.clip(rng.normal(40.0, 10.0, n_samples), 3, None)
    tool_wear = rng.uniform(0, 250, n_samples)

    if drift:  # environment changed
        air_temp += 3.0
        torque = np.clip(torque * 1.15, 3, None)
        tool_wear = rng.uniform(50, 260, n_samples)

    process_temp = air_temp + 10.0 + rng.normal(0, 1.0, n_samples)
    # speed is inversely linked to torque (higher load -> slower spindle)
    rpm = 1540 - (torque - 40) * 16 + rng.normal(0, 80, n_samples)
    rpm = np.clip(rpm, 1150, 2900)

    power = torque * rpm * 2 * np.pi / 60
    strain_limit = np.select([machine_type == "L", machine_type == "M"], [11000, 12000], 13000)

    twf = (tool_wear >= 200) & (tool_wear <= 240) & (rng.random(n_samples) < 0.1)
    hdf = ((process_temp - air_temp) < 8.6) & (rpm < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf = (tool_wear * torque) > strain_limit
    rnf = rng.random(n_samples) < 0.001
    failure = (twf | hdf | pwf | osf | rnf).astype(int)

    # what the sensors actually report (with measurement noise)
    df = pd.DataFrame({
        "machine_id": [f"{t}{rng.integers(10000, 99999)}" for t in machine_type],
        "machine_type": machine_type,
        "air_temperature_k": (air_temp + rng.normal(0, 0.3, n_samples)).round(1),
        "process_temperature_k": (process_temp + rng.normal(0, 0.3, n_samples)).round(1),
        "rotational_speed_rpm": (rpm + rng.normal(0, 15, n_samples)).round(0).astype(int),
        "torque_nm": (torque + rng.normal(0, 1.0, n_samples)).clip(1).round(1),
        "tool_wear_min": (tool_wear + rng.normal(0, 3, n_samples)).clip(0).round(0).astype(int),
        "machine_failure": failure,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic machine sensor data")
    parser.add_argument("--drift", action="store_true", help="generate drifted 'new production' data")
    parser.add_argument("--n", type=int, default=None, help="number of rows")
    args = parser.parse_args()

    cfg = load_config()
    n = args.n or cfg["data_generation"]["n_samples"]

    if args.drift:
        df = generate_sensor_data(n, random_state=7, drift=True)
        out = path(cfg["paths"]["new_data"])
    else:
        df = generate_sensor_data(n, cfg["data_generation"]["random_state"])
        out = path(cfg["paths"]["raw_data"])

    df.to_csv(out, index=False)
    logger.info("Saved %d rows to %s (failure rate %.2f%%)", len(df), out, 100 * df["machine_failure"].mean())


if __name__ == "__main__":
    main()
