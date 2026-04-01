"""API REST churn Telco : /health et /predict."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import CustomerFeatures, PredictResponse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "artifacts" / "churn_bundle.joblib"
MODEL_PATH = Path(os.environ.get("CHURN_MODEL_PATH", str(DEFAULT_MODEL)))

app = FastAPI(title="Telco Churn API", version="1.0.0")

_bundle = None


def load_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Modèle introuvable : {MODEL_PATH}. Lancez python -m ml.train_pipeline"
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def risk_segment(proba: float) -> str:
    if proba >= 0.7:
        return "critique"
    if proba >= 0.4:
        return "élevé"
    if proba >= 0.2:
        return "modéré"
    return "faible"


@app.get("/health")
def health():
    ok = MODEL_PATH.is_file()
    return {"status": "ok" if ok else "degraded", "model_path": str(MODEL_PATH), "model_loaded": ok}


@app.post("/predict", response_model=PredictResponse)
def predict(body: CustomerFeatures):
    try:
        bundle = load_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("threshold", 0.5))
    row = body.model_dump()
    X = pd.DataFrame([row])
    proba = float(pipeline.predict_proba(X)[0, 1])
    churn = proba >= threshold
    return PredictResponse(
        churn_probability=round(proba, 4),
        churn_predicted=bool(churn),
        risk_segment=risk_segment(proba),
    )
