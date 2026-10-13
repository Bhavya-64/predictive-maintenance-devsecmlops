"""
Runs the full training pipeline end to end:

  generate data -> ingest -> validate -> EDA -> split -> train/compare/tune/evaluate -> register

Usage:  python -m src.pipeline
"""
from src.data_generation import generate_sensor_data
from src.data_ingestion import load_raw_data, split_and_save
from src.data_validation import validate_data
from src.eda import run_eda
from src.train import run_training, set_champion
from src.utils import get_logger, load_config, path

logger = get_logger(__name__)


def main():
    cfg = load_config()
    raw_path = path(cfg["paths"]["raw_data"])

    logger.info("STEP 1/5  Data generation")
    if not raw_path.exists():
        generate_sensor_data(cfg["data_generation"]["n_samples"], cfg["data_generation"]["random_state"]) \
            .to_csv(raw_path, index=False)
    else:
        logger.info("Raw data already exists at %s - reusing it", raw_path)

    logger.info("STEP 2/5  Data ingestion")
    df = load_raw_data(raw_path)

    logger.info("STEP 3/5  Data validation")
    validate_data(df)

    logger.info("STEP 4/5  EDA + train/test split")
    run_eda(df)
    train, test = split_and_save(df)

    logger.info("STEP 5/5  Training, comparison, tuning, evaluation, registry")
    meta = run_training(train, test, run_label="initial_training")
    set_champion(meta["model_version"])
    logger.info("Pipeline finished. Model: %s v%s | test F1=%.4f recall=%.4f",
                meta["model_name"], meta["model_version"], meta["test_metrics"]["f1"], meta["test_metrics"]["recall"])


if __name__ == "__main__":
    main()
