"""
FastAPI service for the exoplanet-triage XGBoost classifier.

Loads the trained model bundle once at startup and exposes:
  GET  /health   -> liveness check
  POST /predict  -> classify one Kepler Object of Interest (KOI)
"""

import json
import logging
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

# Load reference statistics for input-drift monitoring. Computed from the
# training split only (see scripts/build_reference_stats.py). For each feature
# we flag incoming values that fall outside the training 1st-99th percentile
# range -- i.e. in the extreme tails of what the model actually learned from.
REFERENCE_STATS_PATH = Path(__file__).parent / "reference_stats.json"
with open(REFERENCE_STATS_PATH) as f:
    reference_stats = json.load(f)

logger = logging.getLogger("exoplanet_triage")

app = FastAPI(title="Exoplanet Triage API", version="0.1.0")


# --- Request/response shapes ---
# The request carries a dict of feature-name -> value. We validate that dict
# against feature_columns at request time rather than hardcoding 99 fields.
class PredictionRequest(BaseModel):
    features: dict[str, float]


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    drift: dict


def check_drift(features: dict[str, float]) -> dict:
    """
    Compare incoming feature values against the training reference range.

    For each feature we flag values below the training 1st percentile or above
    the 99th -- i.e. in the extreme tails of what the model learned from. This
    does NOT block the prediction; it surfaces a signal that the input looks
    unlike training data, which is how you'd catch a model silently going stale
    in production. Returns a summary plus the specific out-of-range features.
    """
    out_of_range = {}
    for name, value in features.items():
        stats = reference_stats.get(name)
        if stats is None:
            continue
        if value < stats["p01"] or value > stats["p99"]:
            out_of_range[name] = {
                "value": value,
                "expected_range": [stats["p01"], stats["p99"]],
            }

    n_checked = sum(1 for name in features if name in reference_stats)
    n_drifted = len(out_of_range)
    fraction = n_drifted / n_checked if n_checked else 0.0

    if n_drifted:
        logger.warning(
            "Input drift detected: %d/%d features out of training range (%.0f%%)",
            n_drifted, n_checked, fraction * 100,
        )

    return {
        "n_features_checked": n_checked,
        "n_features_out_of_range": n_drifted,
        "fraction_out_of_range": round(fraction, 4),
        "out_of_range_features": out_of_range,
    }


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

    # Check whether this input looks like the training data. Logged as a
    # warning if not, and returned so callers can see it -- but never blocks
    # the prediction.
    drift = check_drift(request.features)
 
    return PredictionResponse(
        predicted_class=str(pred_label),
        probabilities=probabilities,
        drift=drift,
    )