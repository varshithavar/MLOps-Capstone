
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "breast_cancer.csv"
REPORT_PATH = ROOT_DIR / "reports" / "data_validation.txt"


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

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = (
        "Data validation: PASSED\n"
        f"Rows: {len(df)}\n"
        f"Columns: {len(df.columns)}\n"
        f"Missing values: {int(df.isnull().sum().sum())}\n"
        f"Target classes: {sorted(df['target'].unique().tolist())}\n"
    )

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Data validation passed.")
    print(f"Rows: {len(df)}, columns: {len(df.columns)}")
    print(f"Validation report saved: {REPORT_PATH}")

    return True


if __name__ == "__main__":
    validate_data()
