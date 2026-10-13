"""
Real-time sensor simulator: acts like IoT sensors on the factory floor, sending one
reading per machine every `--interval` seconds to the API's /predict endpoint.

Each simulated machine slowly degrades (tool wear goes up, temperatures creep up),
so you can watch its health score fall and risk level rise in real time.

Usage (API must be running):
  python -m scripts.simulate_stream                      # 3 machines, normal conditions
  python -m scripts.simulate_stream --drift --steps 60   # drifted factory conditions
"""
import argparse
import os
import random
import time

import requests

RISK_ICON = {"LOW": "[ OK ]", "MEDIUM": "[WARN]", "HIGH": "[HIGH]", "CRITICAL": "[CRIT]"}


def make_machine(machine_id: str, drift: bool) -> dict:
    return {
        "machine_id": machine_id,
        "machine_type": random.choice(["L", "M", "H"]),
        "air_temperature_k": random.gauss(300, 1) + (3 if drift else 0),
        "torque_nm": random.gauss(40, 5) * (1.15 if drift else 1),
        "tool_wear_min": random.uniform(0, 120) + (60 if drift else 0),
    }


def next_reading(m: dict) -> dict:
    m["tool_wear_min"] = min(m["tool_wear_min"] + random.uniform(2, 6), 300)  # degradation
    m["air_temperature_k"] += random.gauss(0.02, 0.1)
    torque = max(5, m["torque_nm"] + random.gauss(0, 3))
    return {
        "machine_id": m["machine_id"],
        "machine_type": m["machine_type"],
        "air_temperature_k": round(m["air_temperature_k"], 1),
        "process_temperature_k": round(m["air_temperature_k"] + 10 + random.gauss(0, 0.8), 1),
        "rotational_speed_rpm": round(max(1150, 1540 - (torque - 40) * 16 + random.gauss(0, 60))),
        "torque_nm": round(torque, 1),
        "tool_wear_min": round(m["tool_wear_min"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--machines", type=int, default=3)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between rounds")
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()

    headers = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}
    machines = [make_machine(f"MCH-{i + 1:03d}", args.drift) for i in range(args.machines)]

    print(f"Streaming to {args.url}/predict  (Ctrl+C to stop)\n")
    for step in range(1, args.steps + 1):
        for m in machines:
            reading = next_reading(m)
            try:
                r = requests.post(f"{args.url}/predict", json=reading, headers=headers, timeout=5)
                r.raise_for_status()
                p = r.json()
                print(f"step {step:3d} | {reading['machine_id']} | wear {reading['tool_wear_min']:3d} min | "
                      f"health {p['health_score']:5.1f} | {RISK_ICON[p['risk_level']]} {p['risk_level']:<8} | "
                      f"{p['maintenance_urgency']}")
            except requests.RequestException as exc:
                print(f"Request failed: {exc}")
                return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
