"""
Build reference feature statistics for input-drift monitoring.

Reproduces the exact training pipeline (acquisition -> cleaning -> split)
from notebooks 01/03/04, then computes per-feature statistics from the
TRAINING split only. These become the reference distribution that the API's
drift check compares incoming requests against.

Train-only is deliberate: the model never saw the test rows, so drift is
measured against what the model actually learned from -- the same reason
the scaler in nb 04 was fit on train only.

Run:  python scripts/build_reference_stats.py
Output:  reference_stats.json  (in the project root)
"""

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
BUNDLE_PATH = Path(__file__).parent.parent / "models" / "exoplanet_model_bundle.joblib"
OUTPUT_PATH = Path(__file__).parent.parent / "reference_stats.json"


def fetch_koi_table() -> pd.DataFrame:
    """Pull the full KOI cumulative table from NASA TAP (matches nb 01)."""
    print("Fetching KOI table from NASA TAP...")
    params = {"query": "select * from cumulative", "format": "csv"}
    response = requests.get(TAP_URL, params=params, timeout=120)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    print(f"  fetched {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the nb 03 cleaning steps in order."""
    # Drop columns missing more than 50% of values.
    missing_pct = df.isnull().sum() / len(df) * 100
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
    df = df.drop(columns=cols_to_drop)

    # Drop leakage + identifier columns (nb 03).
    leakage_and_ids = [
        "koi_pdisposition", "koi_score",
        "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
        "kepid", "kepoi_name", "ra", "dec",
    ]
    df = df.drop(columns=[c for c in leakage_and_ids if c in df.columns])

    # Drop leftover string coordinate columns.
    coord_strings = ["ra_str", "dec_str", "ra_err", "dec_err"]
    df = df.drop(columns=[c for c in coord_strings if c in df.columns])

    # Drop free-text vetting notes.
    if "koi_comment" in df.columns:
        df = df.drop(columns=["koi_comment"])

    # Median-impute remaining numeric gaps.
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Drop remaining text metadata columns.
    metadata_cols = [
        "koi_quarters", "koi_limbdark_mod", "koi_trans_mod", "koi_sparprov",
        "koi_tce_delivname", "koi_datalink_dvs", "koi_datalink_dvr",
    ]
    df = df.drop(columns=[c for c in metadata_cols if c in df.columns])

    return df


def main():
    # The model bundle tells us the exact feature columns to compute stats for.
    bundle = joblib.load(BUNDLE_PATH)
    feature_columns = bundle["feature_columns"]

    df = fetch_koi_table()
    df = clean(df)

    # Separate features and target, drop any remaining non-numeric columns (nb 04).
    y = df["koi_disposition"]
    X = df.drop(columns=["koi_disposition"])
    X = X.select_dtypes(include="number")

    # Reproduce the exact train/test split from nb 04.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  train rows: {X_train.shape[0]:,}, test rows: {X_test.shape[0]:,}")

    # Sanity check: the regenerated features should match the model's columns.
    regenerated = set(X_train.columns)
    expected = set(feature_columns)
    missing = expected - regenerated
    extra = regenerated - expected
    if missing:
        print(f"  WARNING: model expects {len(missing)} columns not in regenerated data: {sorted(missing)[:5]}")
    if extra:
        print(f"  WARNING: regenerated data has {len(extra)} columns the model doesn't use: {sorted(extra)[:5]}")
    if not missing and not extra:
        print("  OK: regenerated feature columns exactly match the model bundle")

    # Compute reference stats from the TRAINING split only, in the model's column order.
    stats = {}
    for col in feature_columns:
        if col not in X_train.columns:
            continue
        series = X_train[col]
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "p01": float(series.quantile(0.01)),
            "p99": float(series.quantile(0.99)),
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote reference stats for {len(stats)} features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()