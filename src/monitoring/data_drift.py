
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


ROOT_DIR = Path(__file__).resolve().parents[2]

REFERENCE_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
CURRENT_PATH = ROOT_DIR / "data" / "processed" / "test.csv"

REPORT_DIR = ROOT_DIR / "reports"


def generate_data_drift_report():
    """Compare reference and current feature distributions."""

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_PATH}")

    if not CURRENT_PATH.exists():
        raise FileNotFoundError(f"Current dataset not found: {CURRENT_PATH}")

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(CURRENT_PATH)

    # Exclude target from feature drift analysis.
    feature_columns = [
        column for column in reference.columns
        if column != "target"
    ]

    if not feature_columns:
        raise ValueError("No feature columns available for drift analysis.")

    missing_columns = set(feature_columns) - set(current.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in current data: {missing_columns}")

    # Use identical feature columns in both datasets.
    reference_features = reference[feature_columns]
    current_features = current[feature_columns]

    # Check for missing values.
    if reference_features.isnull().any().any():
        raise ValueError("Reference dataset contains missing values.")

    if current_features.isnull().any().any():
        raise ValueError("Current dataset contains missing values.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Create Evidently data drift report.
    report = Report([
        DataDriftPreset(columns=feature_columns)
    ])

    snapshot = report.run(
        current_data=current_features,
        reference_data=reference_features
    )

    # Save HTML and JSON reports.
    html_path = REPORT_DIR / "data_drift.html"
    json_path = REPORT_DIR / "data_drift.json"

    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    print("Data drift analysis completed successfully.")
    print(f"Reference rows: {len(reference_features)}")
    print(f"Current rows: {len(current_features)}")
    print(f"Features monitored: {len(feature_columns)}")
    print(f"HTML report: {html_path}")
    print(f"JSON report: {json_path}")

    return snapshot


if __name__ == "__main__":
    generate_data_drift_report()
