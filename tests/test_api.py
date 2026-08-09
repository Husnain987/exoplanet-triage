"""
Tests for the exoplanet-triage API.

Uses FastAPI's TestClient, which runs the app in-process (no separate
server needed) and lets us send requests and assert on the responses.
The demo_sample row bundled with the model gives us a real, known input
to test /predict against.
"""

import joblib
from pathlib import Path

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

# Load one real row from the model bundle to use as a valid /predict input.
BUNDLE_PATH = Path(__file__).parent.parent / "models" / "exoplanet_model_bundle.joblib"
bundle = joblib.load(BUNDLE_PATH)
DEMO_ROW = (
    bundle["demo_sample"]
    .head(1)
    .drop(columns=["kepoi_name", "true_label"])
    .to_dict("records")[0]
)
VALID_CLASSES = set(bundle["label_encoder"].classes_)


def test_health_returns_ok():
    """/health should report ok and the expected feature count."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["n_features"] == 99


def test_predict_valid_row_returns_a_known_class():
    """A real feature row should classify into one of the model's classes."""
    response = client.post("/predict", json={"features": DEMO_ROW})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in VALID_CLASSES


def test_predict_probabilities_sum_to_one():
    """The returned probabilities should cover all classes and sum to ~1."""
    response = client.post("/predict", json={"features": DEMO_ROW})
    body = response.json()
    probs = body["probabilities"]
    assert set(probs.keys()) == VALID_CLASSES
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_predict_missing_features_is_rejected():
    """A request missing required features should be rejected, not guessed at."""
    response = client.post("/predict", json={"features": {"koi_gmag": 12.4}})
    assert response.status_code == 422