# Exoplanet Triage

![CI](https://github.com/Husnain987/exoplanet-triage/actions/workflows/ci.yml/badge.svg)

A production ML service that classifies Kepler telescope signals as **CONFIRMED**, **CANDIDATE**, or **FALSE POSITIVE**, to help prioritize which of NASA's ~9,500 Kepler Objects of Interest (KOI) deserve human follow-up.

An XGBoost classifier is served behind a FastAPI endpoint, containerized with Docker, covered by an automated test suite that runs in CI, and instrumented with input-drift monitoring. The model itself is trained end to end from data pulled directly off the NASA Exoplanet Archive, so the whole thing is reproducible from scratch.

🔗 [Live model demo](https://exoplanet-triage-w2uelay4dsjpijhst2ntg9.streamlit.app) — pick a real Kepler candidate and see the prediction, confidence, and SHAP explanation.

---

## What's here

Two layers, each independently useful:

- **The model** — an XGBoost classifier reaching 0.83 macro F1 on 3 imbalanced classes, built with strict leakage control and a SHAP faithfulness check. Full modeling story below.
- **The service** — a FastAPI app that loads the trained model and answers prediction requests over HTTP, packaged in Docker so it runs identically anywhere, with drift monitoring on every request and a passing CI pipeline.

---

## Quickstart

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --reload
```

Then open http://127.0.0.1:8000/docs for an interactive API console, or hit http://127.0.0.1:8000/health.

### Run with Docker

```bash
docker build -t exoplanet-triage .
docker run -p 8000:8000 exoplanet-triage
```

The container binds the service on port 8000. Same endpoints as above — no Python or dependency setup needed on the host.

---

## API

### `GET /health`

Liveness check. Returns status and the number of features the model expects.

```json
{ "status": "ok", "n_features": 99 }
```

### `POST /predict`

Classifies one KOI from a dictionary of its feature values. The request must supply all features the model was trained on; missing features are rejected with a 422 rather than silently guessed. The response carries the predicted class, the full probability distribution across all three classes, and a drift summary (see below).

**Request:**

```json
{ "features": { "koi_gmag": 12.407, "koi_period": 117.68, "...": "..." } }
```

**Response:**

```json
{
  "predicted_class": "FALSE POSITIVE",
  "probabilities": {
    "CANDIDATE": 0.00005,
    "CONFIRMED": 0.00002,
    "FALSE POSITIVE": 0.99993
  },
  "drift": {
    "n_features_checked": 99,
    "n_features_out_of_range": 0,
    "fraction_out_of_range": 0.0,
    "out_of_range_features": {}
  }
}
```

---

## Input-drift monitoring

A model silently degrades when the data it receives in production drifts away from what it was trained on — and nothing errors, the predictions just quietly get worse. This service checks for that on every request.

`reference_stats.json` holds per-feature statistics (mean, std, 1st and 99th percentiles) computed from the **training split only** — the same rows the model actually learned from, mirroring the train-only discipline used for the scaler in modeling. On each prediction, any incoming feature value that falls below the training 1st percentile or above the 99th is flagged as out-of-range, logged as a warning, and reported in the response. It never blocks the prediction; it surfaces a signal.

Regenerate the reference statistics (re-pulls from NASA and reruns the training pipeline):

```bash
python scripts/build_reference_stats.py
```

---

## Tests and CI

```bash
pytest -v
```

The suite covers the health check, valid prediction, probability normalization, rejection of malformed input, and drift detection in both directions (a real row stays in-range; an absurd value is caught). GitHub Actions runs the full suite on a clean Ubuntu machine with Python 3.12 on every push — the badge at the top reflects its current state.

---

## The model

Data is pulled directly from the NASA Exoplanet Archive via its TAP API, so the entire pipeline is reproducible from scratch.

### Pipeline

| Notebook | What it does |
|---|---|
| `01_acquisition` | Pulls the KOI cumulative table (9,564 × 153) from the NASA Exoplanet Archive TAP API |
| `02_sql_exploration` | Loads data into SQLite, explores class balance and feature distributions with SQL + visualizations |
| `03_cleaning` | Drops empty/leakage/metadata columns, median-imputes missing values (153 → 106 features) |
| `04_modeling` | Trains and compares three classifiers with proper evaluation |
| `05_explainability` | Computes global SHAP feature importance across the test set and validates explanation faithfulness with a feature-deletion test |

### Key decisions

**Leakage discipline.** Columns encoding NASA's own vetting verdict (`koi_score`, `koi_pdisposition`, the `koi_fpflag_*` flags) were dropped before modeling — training on them would inflate accuracy while teaching the model nothing about the underlying physics.

**Metric choice.** Classes are imbalanced (~51% false positives), so accuracy is misleading — a model guessing "false positive" every time scores ~51%. Models are judged on precision, recall, and F1.

### Results

| Model | Macro F1 |
|---|---|
| Logistic Regression (baseline) | 0.76 |
| Random Forest | 0.83 |
| XGBoost | 0.83 |

Two strong, mechanically different tree models converging on 0.83 suggests the limit is the data's information content, not model choice. The CANDIDATE class is hardest (F1 ~0.66) — candidates are by definition unresolved signals that overlap both other classes, so the ambiguity is real, not a modeling flaw.

The strongest predictive feature is planet radius (`koi_prad`): confirmed planets average ~2.9 Earth radii vs ~165 for false positives, since many false positives are stellar-sized eclipsing binaries.

**Explainability.** Global SHAP analysis identifies `koi_max_mult_ev` (transit signal strength) and `koi_prad` as top drivers. To verify these explanations reflect real model behavior rather than plausible-looking noise, a deletion test replaced each test row's top-5 SHAP-ranked features with their training-set median and measured the resulting confidence drop. Removing SHAP-ranked features caused a mean confidence drop of **0.217**, versus **0.014** for randomly chosen features — SHAP-targeted deletion beat random deletion on **92.4% of test rows**, confirming the explanations are faithful to the model's actual decision-making.

### Limitations

- Candidate ambiguity caps overall performance — see above.
- **Near-leakage features.** The top centroid-offset features (`koi_dicco_msky`, `koi_dikco_msky`) are closely related to a dropped false-positive flag. Kept as raw measurements, but a stricter version might drop them to test performance on fully independent signal.
- Uses the KOI catalog (pre-extracted features), not raw light curves.

---

## Project layout

```
exoplanet-triage/
├── api.py                       # FastAPI service (endpoints + drift check)
├── Dockerfile                   # Container definition
├── requirements.txt             # Service runtime dependencies (pinned)
├── reference_stats.json         # Training-split feature stats for drift monitoring
├── models/                      # Trained model bundle (model + encoder + features)
├── scripts/
│   └── build_reference_stats.py # Regenerates reference_stats.json from NASA data
├── tests/
│   └── test_api.py              # API + drift tests (run in CI)
├── notebooks/                   # 01–05: acquisition through explainability
├── app/                         # Streamlit demo
└── .github/workflows/ci.yml     # Test pipeline
```