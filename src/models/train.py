
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MLFLOW_DB = ROOT_DIR / "mlflow.db"
ARTIFACT_DIR = ROOT_DIR / "mlartifacts"

EXPERIMENT_NAME = "Breast_Cancer_Classification"
RANDOM_STATE = 42


# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

def build_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "support_vector_machine": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


# --------------------------------------------------
# TRAIN AND EVALUATE MODELS
# --------------------------------------------------

def train_models():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            "Run: python src\\data\\preprocess.py"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Testing data not found: {test_path}\n"
            "Run: python src\\data\\preprocess.py"
        )

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "target" not in train_df.columns:
        raise ValueError("Training data must contain a 'target' column.")

    if "target" not in test_df.columns:
        raise ValueError("Testing data must contain a 'target' column.")

    if train_df.empty or test_df.empty:
        raise ValueError("Training and testing datasets must not be empty.")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing features do not match."
        )

    if X_train.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError("Feature data contains missing values.")

    # Create output directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # MLFLOW SQLITE TRACKING CONFIGURATION
    # --------------------------------------------------

    # Absolute SQLite URI works with the project's database path.
    mlflow.set_tracking_uri(
        MLFLOW_DB.as_uri().replace("file:///", "sqlite:///")
    )

    # Store artifacts in the project directory.
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    results = []
    best_f1 = -1.0
    best_model = None
    best_name = None

    # --------------------------------------------------
    # TRAIN EACH MODEL
    # --------------------------------------------------

    for model_name, estimator in build_models().items():
        print(f"\nTraining {model_name}...")

        # The pipeline scales the features and then fits the model.
        # The current preprocessing script also scales the data;
        # this additional scaling is redundant but keeps the model
        # pipeline self-contained for future predictions.
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

        with mlflow.start_run(run_name=model_name) as run:
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
            }

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("testing_rows", len(X_test))

            # Log metrics
            mlflow.log_metrics({
                key: float(value)
                for key, value in metrics.items()
            })

            # Log trained model artifact
            mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            results.append({
                "model": model_name,
                **metrics,
                "run_id": run.info.run_id,
            })

            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 score:  {metrics['f1_score']:.4f}")
            print(f"MLflow run ID: {run.info.run_id}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model = pipeline
                best_name = model_name

    # --------------------------------------------------
    # SAVE COMPARISON REPORT
    # --------------------------------------------------

    result_df = pd.DataFrame(results).sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
    )

    report_path = REPORT_DIR / "model_comparison.csv"
    result_df.to_csv(report_path, index=False)

    # --------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------

    if best_model is None:
        raise RuntimeError("Training failed: no model was produced.")

    best_model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)

    # Save the selected model's name and score for reference.
    summary_path = REPORT_DIR / "best_model_summary.txt"

    summary_path.write_text(
        f"Best model: {best_name}\n"
        f"Best F1 score: {best_f1:.4f}\n"
        f"Selection metric: F1 score\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)

    print(
        result_df[
            ["model", "accuracy", "precision", "recall", "f1_score"]
        ].to_string(index=False)
    )

    print(f"\nBest model: {best_name}")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Saved model: {best_model_path}")
    print(f"Comparison report: {report_path}")
    print(f"MLflow database: {MLFLOW_DB}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train_models()
