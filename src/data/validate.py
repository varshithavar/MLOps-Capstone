
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "breast_cancer.csv"


def validate_data() -> bool:
    """Validate the raw dataset before model training."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if "target" not in df.columns:
        raise ValueError("Missing target column.")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    print("Data validation passed.")
    print(f"Rows: {len(df)}, columns: {len(df.columns)}")

    return True


if __name__ == "__main__":
    validate_data()
