
from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "breast_cancer.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PARAMS_PATH = ROOT_DIR / "params.yaml"


def preprocess_data():
    """Split, scale, and save training and testing datasets."""

    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    df = pd.read_csv(RAW_DATA_PATH)

    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df["target"] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df["target"] = y_test.reset_index(drop=True)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")
    print("Scaler saved successfully.")
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_data()
