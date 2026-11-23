
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "breast_cancer.csv"


def load_data() -> pd.DataFrame:
    """Load the built-in dataset and save a local CSV copy."""
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Dataset saved to: {RAW_DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nTarget distribution:")
    print(df["target"].value_counts().sort_index())

    return df


if __name__ == "__main__":
    load_data()
