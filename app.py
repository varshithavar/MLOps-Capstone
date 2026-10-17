
from fastapi import FastAPI

app = FastAPI(
    title="MLOps Capstone",
    description="Full Production Machine Learning Pipeline",
    version="1.0.0",
)


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
    }
