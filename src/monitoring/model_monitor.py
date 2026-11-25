
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
SCALER_PATH = ROOT_DIR / "data" / "processed" / "scaler.joblib"
TEST_PATH = ROOT_DIR / "data" / "processed" / "test.csv"
REPORT_DIR = ROOT_DIR / "reports"


def run_model_monitoring():
    # Check required files
    for path in [MODEL_PATH, SCALER_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load model, scaler and test dataset
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    test_data = pd.read_csv(TEST_PATH)

    # Separate features and actual labels
    X_scaled = test_data.drop(columns=["target"])
    y_true = test_data["target"]

    # Test CSV is already scaled. Restore original feature values
    # because best_model.joblib contains its own scaler.
    X_raw = pd.DataFrame(
        scaler.inverse_transform(X_scaled),
        columns=X_scaled.columns,
        index=X_scaled.index,
    )

    # Generate predictions
    y_pred = model.predict(X_raw)

    # Calculate classification metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }

    # Create reports directory
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = REPORT_DIR / "model_monitoring.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Save confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    )
    matrix_path = REPORT_DIR / "confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    # Print results
    print("\nMODEL MONITORING RESULTS")
    print("-" * 35)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(matrix)

    print(f"\nMetrics saved: {metrics_path}")
    print(f"Confusion matrix saved: {matrix_path}")

    return metrics


if __name__ == "__main__":
    run_model_monitoring()
