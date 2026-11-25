
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

ROOT_DIR = Path(__file__).resolve().parent
MLFLOW_DB = ROOT_DIR / "mlflow.db"

TRACKING_URI = (
    MLFLOW_DB.as_uri().replace("file:///", "sqlite:///")
)

MODEL_NAME = "BreastCancerClassifier"
MODEL_VERSION = "2"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

app = FastAPI(
    title="MLOps Capstone",
    description="Full Production Machine Learning Pipeline",
    version="1.0.0",
)

model = None

FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float]


@app.on_event("startup")
def load_model():
    global model

    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.sklearn.load_model(model_uri)


@app.get("/")
def home():
    return {
        "project": "MLOps Capstone",
        "message": "Application is running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "mlops-capstone",
        "model_loaded": model is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "features_required": len(FEATURE_NAMES),
    }


@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    supplied_features = data.features

    missing = [
        name for name in FEATURE_NAMES
        if name not in supplied_features
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing},
        )

    if len(supplied_features) != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=422,
            detail="Exactly 30 features must be supplied",
        )

    try:
        input_df = pd.DataFrame(
            [[supplied_features[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )

        prediction = int(model.predict(input_df)[0])

        probabilities = model.predict_proba(input_df)[0]

        return {
            "prediction": prediction,
            "diagnosis": (
                "Malignant" if prediction == 0 else "Benign"
            ),
            "probabilities": {
                "malignant": float(probabilities[0]),
                "benign": float(probabilities[1]),
            },
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )
