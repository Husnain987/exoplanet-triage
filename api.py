"""
FastAPI service for the exoplanet-triage XGBoost classifier.

Loads the trained model bundle once at startup and exposes:
  GET  /health   -> liveness check
  POST /predict  -> classify one Kepler Object of Interest (KOI)
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Load the model bundle once, at import time ---
# Loading here (not per-request) means the model sits in memory and every
# request is fast. The bundle is the source of truth for its own contract:
# which features it needs and which class names its numbers map back to.
BUNDLE_PATH = Path(__file__).parent / "models" / "exoplanet_model_bundle.joblib"
bundle = joblib.load(BUNDLE_PATH)

model = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_columns = bundle["feature_columns"]

app = FastAPI(title="Exoplanet Triage API", version="0.1.0")


# --- Request/response shapes ---
# The request carries a dict of feature-name -> value. We validate that dict
# against feature_columns at request time rather than hardcoding 99 fields.
class PredictionRequest(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]


@app.get("/health")
def health():
    """Liveness check. Returns ok plus how many features the model expects."""
    return {"status": "ok", "n_features": len(feature_columns)}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Classify one KOI from its feature dict."""
    # Reject requests missing any feature the model needs.
    missing = [c for c in feature_columns if c not in request.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing {len(missing)} required features: {missing[:5]}...",
        )

    # Build a one-row DataFrame with columns in the exact order the model
    # was trained on. Column order matters to XGBoost; this enforces it.
    row = pd.DataFrame([request.features])[feature_columns]

    # Predict the class index, then translate it back to a human label.
    pred_index = model.predict(row)[0]
    pred_label = label_encoder.inverse_transform([pred_index])[0]

    # Full probability distribution across all three classes.
    proba = model.predict_proba(row)[0]
    probabilities = {
        label_encoder.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(proba)
    }

    return PredictionResponse(
        predicted_class=str(pred_label),
        probabilities=probabilities,
    )