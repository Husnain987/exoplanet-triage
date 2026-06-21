# Exoplanet Triage

🔗 **[Try the live demo](https://exoplanet-triage-w2uelay4dsjpijhst2ntg9.streamlit.app)** — pick a real Kepler candidate and see the model's prediction, confidence, and SHAP explanation.

Machine learning classification of Kepler exoplanet candidates from NASA's
Kepler Objects of Interest (KOI) catalog — ~9,500 telescope signals labeled
**CONFIRMED**, **CANDIDATE**, or **FALSE POSITIVE**. The goal is a triage model
that helps prioritize which signals deserve human follow-up.

Data is pulled directly from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu)
via its TAP API, so the entire pipeline is reproducible from scratch.

**Status:** Acquisition, SQL exploration, cleaning, modeling, and explainability
complete — full pipeline finished.

## Pipeline

| Notebook | What it does |
|---|---|
| `01_acquisition` | Pulls the KOI cumulative table (9,564 × 153) from the NASA Exoplanet Archive TAP API |
| `02_sql_exploration` | Loads data into SQLite, explores class balance and feature distributions with SQL + visualizations |
| `03_cleaning` | Drops empty/leakage/metadata columns, median-imputes missing values (153 → 106 features) |
| `04_modeling` | Trains and compares three classifiers with proper evaluation |
| `05_explainability` | Computes global SHAP feature importance across the test set and validates explanation faithfulness with a feature-deletion test |

## Key Decisions

**Leakage discipline.** Columns encoding NASA's own vetting verdict
(`koi_score`, `koi_pdisposition`, the `koi_fpflag_*` flags) were dropped before
modeling — training on them would inflate accuracy while teaching the model
nothing about the underlying physics.

**Metric choice.** Classes are imbalanced (~51% false positives), so accuracy is
misleading — a model guessing "false positive" every time scores ~51%. Models
are judged on precision, recall, and F1.

## Results

| Model | Macro F1 |
|---|---|
| Logistic Regression (baseline) | 0.76 |
| Random Forest | 0.83 |
| XGBoost | 0.83 |

Two strong, mechanically different tree models converging on 0.83 suggests the
limit is the data's information content, not model choice. The **CANDIDATE**
class is hardest (F1 ~0.66) — candidates are by definition unresolved signals
that overlap both other classes, so the ambiguity is real, not a modeling flaw.

The strongest predictive feature is planet radius (`koi_prad`): confirmed planets
average ~2.9 Earth radii vs ~165 for false positives, since many false positives
are stellar-sized eclipsing binaries.

**Explainability.** Global SHAP analysis broadly agrees with the modeling
results above — `koi_max_mult_ev` (transit signal strength) and `koi_prad`
rank among the top drivers, alongside the centroid-offset features flagged
below as near-leakage. To verify these explanations reflect real model
behavior rather than plausible-looking noise, a deletion test replaced each
test row's top-5 SHAP-ranked features with their training-set median and
measured the resulting drop in the model's confidence. Removing SHAP-ranked
features caused a mean confidence drop of 0.217, versus 0.014 for randomly
chosen features — SHAP-targeted deletion beat random deletion on 92.4% of
test rows, evidence the explanations are faithful to the model's actual
decision-making.

## Limitations

- **Candidate ambiguity** caps overall performance — see above.
- **Near-leakage features.** The top features (`koi_dicco_msky`, `koi_dikco_msky`)
  are centroid-offset measurements closely related to a dropped false-positive
  flag. Kept as raw measurements, but a stricter version might drop them to test
  performance on fully independent signal.
- Uses the KOI **catalog** (pre-extracted features), not raw light curves.

## Setup
