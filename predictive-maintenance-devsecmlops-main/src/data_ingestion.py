"""
Data ingestion: load raw sensor CSV, standardise it, and create train/test splits.

If you later get a real dataset (e.g. the AI4I 2020 CSV from Kaggle), rename its
columns to match FEATURE_COLS + TARGET_COL and drop it in data/raw/ - nothing else
needs to change.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import FEATURE_COLS, TARGET_COL, get_logger, load_config, path

logger = get_logger(__name__)


def load_raw_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    logger.info("Ingested %d rows x %d columns from %s", *df.shape, csv_path)
    return df


def split_and_save(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    df = df.drop_duplicates()
    df = df[FEATURE_COLS + [TARGET_COL]]  # keep only modelling columns (drops machine_id)

    train, test = train_test_split(
        df,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=df[TARGET_COL],  # keep the same failure rate in both splits
    )
    train.to_csv(path(cfg["paths"]["train_data"]), index=False)
    test.to_csv(path(cfg["paths"]["test_data"]), index=False)
    logger.info("Train: %d rows | Test: %d rows", len(train), len(test))
    return train, test
