"""
Automated retraining (drift-triggered).

Flow:
  1. Run drift check on new production data
  2. No drift  -> stop, keep current model
  3. Drift     -> validate new data, combine old + new data, retrain (same pipeline)
  4. Compare new model vs current model on the NEW test data
  5. Promote only if it is not worse (champion / challenger) -> else keep old model
  6. Alert either way

Usage:
  python -m src.data_generation --drift   # simulate new drifted production data
  python -m src.retrain                   # drift check + retrain if needed
  python -m src.retrain --force           # retrain even without drift

Scheduled automatically by .github/workflows/retrain.yml (weekly) as well.
"""
import argparse
import json
import shutil

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.alerts import send_alert
from src.data_ingestion import load_raw_data
from src.data_validation import validate_data
from src.drift import run_drift_check
from src.predict import load_artifacts, reload_artifacts
from src.train import run_training, set_champion
from src.utils import FEATURE_COLS, TARGET_COL, get_logger, load_config, path

logger = get_logger(__name__)


def retrain(force: bool = False) -> dict:
    cfg = load_config()
    drift_report = run_drift_check("file")

    if not drift_report["drift_detected"] and not force:
        logger.info("No drift detected - retraining not needed")
        return {"retrained": False, "reason": "no drift"}

    new_df = load_raw_data(path(cfg["paths"]["new_data"]))
    validate_data(new_df)
    new_df = new_df[FEATURE_COLS + [TARGET_COL]].drop_duplicates()
    new_train, new_test = train_test_split(new_df, test_size=0.3, stratify=new_df[TARGET_COL], random_state=42)

    # champion = current model, scored on fresh data it has never seen
    old_model, old_meta = load_artifacts()
    old_f1 = f1_score(new_test[TARGET_COL], old_model.predict(new_test[FEATURE_COLS]))

    # back up current model so we can roll back
    model_path, meta_path = path(cfg["paths"]["model"]), path(cfg["paths"]["model_metadata"])
    ref_path = path(cfg["paths"]["reference_data"])
    for p in (model_path, meta_path, ref_path):
        shutil.copy(p, p.with_suffix(p.suffix + ".bak"))

    # challenger = retrained on old training data + new production data
    old_train = pd.read_csv(path(cfg["paths"]["train_data"]))
    old_test = pd.read_csv(path(cfg["paths"]["test_data"]))
    combined_train = pd.concat([old_train, new_train], ignore_index=True)
    combined_test = pd.concat([old_test, new_test], ignore_index=True)
    new_meta = run_training(combined_train, combined_test, run_label="drift_retraining")

    new_model, _ = reload_artifacts()
    new_f1 = f1_score(new_test[TARGET_COL], new_model.predict(new_test[FEATURE_COLS]))
    improvement = new_f1 - old_f1
    logger.info("Champion F1 on new data=%.4f | Challenger F1=%.4f", old_f1, new_f1)

    if improvement >= cfg["retraining"]["min_f1_improvement"]:
        set_champion(new_meta["model_version"])
        send_alert("MODEL_RETRAINED", "info", {
            "old_version": old_meta.get("model_version"), "new_version": new_meta["model_version"],
            "old_f1_on_new_data": round(old_f1, 4), "new_f1_on_new_data": round(new_f1, 4),
        })
        result = {"retrained": True, "promoted": True, "old_f1": round(old_f1, 4), "new_f1": round(new_f1, 4)}
    else:  # roll back
        for p in (model_path, meta_path, ref_path):
            shutil.copy(p.with_suffix(p.suffix + ".bak"), p)
        reload_artifacts()
        send_alert("RETRAINING_REJECTED", "warning", {"old_f1": round(old_f1, 4), "new_f1": round(new_f1, 4)})
        result = {"retrained": True, "promoted": False, "old_f1": round(old_f1, 4), "new_f1": round(new_f1, 4)}

    path(f"{cfg['paths']['reports_dir']}/retraining_result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retrain even if no drift is detected")
    print(json.dumps(retrain(parser.parse_args().force), indent=2))
