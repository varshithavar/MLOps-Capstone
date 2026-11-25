
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# ==================================================
# PROJECT PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "test.csv"
SCALER_PATH = ROOT_DIR / "data" / "processed" / "scaler.joblib"

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

PARAMS_PATH = ROOT_DIR / "params.yaml"
MLFLOW_DB = ROOT_DIR / "mlflow.db"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD PARAMETERS
# ==================================================

def load_params():
    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ==================================================
# BUILD MODELS
# ==================================================

def build_models(params):

    model_params = params["model"]

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=model_params["logistic_regression_max_iter"],
            random_state=model_params["random_state"],
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=model_params["random_forest_n_estimators"],
            random_state=model_params["random_state"],
        ),

        "support_vector_machine": SVC(
            kernel=model_params["svm_kernel"],
            probability=True,
            random_state=model_params["random_state"],
        ),
    }

    return models


# ==================================================
# MAIN TRAINING FUNCTION
# ==================================================

def train_models():

    params = load_params()

    experiment_name = params["tracking"]["experiment_name"]

    # Check required files
    required_files = [
        TRAIN_PATH,
        TEST_PATH,
        SCALER_PATH,
        PARAMS_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if "target" not in train_df.columns:
        raise ValueError("Target column missing in training data.")

    if "target" not in test_df.columns:
        raise ValueError("Target column missing in testing data.")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing features do not match.")

    if X_train.isnull().values.any() or X_test.isnull().values.any():
        raise ValueError("Missing values found in processed datasets.")

    # Load scaler fitted during preprocessing
    scaler = joblib.load(SCALER_PATH)

    # Reconstruct original unscaled training features
    X_train_raw = pd.DataFrame(
        scaler.inverse_transform(X_train),
        columns=X_train.columns,
    )

    print(f"Experiment: {experiment_name}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Features: {len(X_train.columns)}")

    # ==================================================
    # MLFLOW CONFIGURATION
    # ==================================================

    tracking_uri = MLFLOW_DB.as_uri().replace(
        "file:///",
        "sqlite:///",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    # ==================================================
    # MODEL TRAINING
    # ==================================================

    results = []

    best_f1 = -1.0
    best_model = None
    best_name = None
    best_run_id = None

    for model_name, estimator in build_models(params).items():

        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name) as run:

            # Data is already scaled by preprocess.py.
            # Do not scale it again.
            estimator.fit(X_train, y_train)

            predictions = estimator.predict(X_test)

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

            # Build complete inference pipeline.
            # This pipeline accepts raw input and applies
            # the saved preprocessing scaler exactly once.
            inference_pipeline = Pipeline([
                ("scaler", scaler),
                ("model", estimator),
            ])

            # Log parameters
            mlflow.log_param("model_name", model_name)

            for key, value in estimator.get_params().items():
                if value is not None:
                    mlflow.log_param(
                        f"model_{key}",
                        str(value),
                    )

            # Log evaluation metrics
            mlflow.log_metrics(metrics)

            # Log complete inference pipeline
            mlflow.sklearn.log_model(
                sk_model=inference_pipeline,
                name="model",
                input_example=X_train_raw.head(3),
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ],
            )

            results.append({
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "run_id": run.info.run_id,
            })

            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 score: {metrics['f1_score']:.4f}")

            # Select best model
            if metrics["f1_score"] > best_f1:

                best_f1 = metrics["f1_score"]
                best_model = inference_pipeline
                best_name = model_name
                best_run_id = run.info.run_id

    # ==================================================
    # MODEL COMPARISON REPORT
    # ==================================================

    result_df = pd.DataFrame(results).sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
    )

    report_path = REPORT_DIR / "model_comparison.csv"

    result_df.to_csv(report_path, index=False)

    print("\nModel comparison report saved.")

    # ==================================================
    # SAVE BEST MODEL
    # ==================================================

    if best_model is None or best_run_id is None:
        raise RuntimeError("Best model was not selected.")

    best_model_path = MODEL_DIR / "best_model.joblib"

    joblib.dump(best_model, best_model_path)

    print(f"\nBest model: {best_name}")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Saved model: {best_model_path}")

    # ==================================================
    # MLFLOW MODEL REGISTRY
    # ==================================================

    print("\nRegistering best model in MLflow...")

    registered_model = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name="BreastCancerClassifier",
    )

    print(f"Registered model: {registered_model.name}")
    print(f"Model version: {registered_model.version}")

    # ==================================================
    # BEST MODEL SUMMARY
    # ==================================================

    summary_path = REPORT_DIR / "best_model_summary.txt"

    summary_path.write_text(
        f"Best model: {best_name}\n"
        f"Best F1 score: {best_f1:.4f}\n"
        f"Selection metric: F1 score\n"
        f"MLflow run ID: {best_run_id}\n"
        f"Registered model: {registered_model.name}\n"
        f"Model version: {registered_model.version}\n",
        encoding="utf-8",
    )

    print("\nTraining completed successfully!")
    print(f"Summary saved: {summary_path}")


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    train_models()
