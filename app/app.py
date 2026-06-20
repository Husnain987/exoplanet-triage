import streamlit as st
import pandas as pd
import altair as alt
import shap
import joblib
from pathlib import Path

st.set_page_config(page_title="Exoplanet Triage", page_icon="🪐", layout="wide")

@st.cache_resource
def load_bundle():
    path = Path(__file__).parent.parent / "models" / "exoplanet_model_bundle.joblib"
    return joblib.load(path)

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

bundle = load_bundle()
model = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_columns = bundle["feature_columns"]
demo_sample = bundle["demo_sample"]

explainer = get_explainer(model)

st.title("🪐 Exoplanet Triage")
st.write(
    "Pick a real Kepler Object of Interest from NASA's KOI catalog and see "
    "what the model — trained on 9,564 candidates — predicts."
)

selected_name = st.selectbox("Choose a KOI:", demo_sample["kepoi_name"])

mask = demo_sample["kepoi_name"] == selected_name
features = demo_sample.loc[mask, feature_columns]
row = demo_sample[mask].iloc[0]

pred_encoded = model.predict(features)[0]
pred_label = label_encoder.inverse_transform([pred_encoded])[0]
probs = model.predict_proba(features)[0]
confidence = probs[pred_encoded]

left, right = st.columns(2)

with left:
    st.subheader("Candidate profile")
    st.metric("Orbital period", f"{row['koi_period']:.2f} days")
    st.metric("Planet radius", f"{row['koi_prad']:.2f} Earth radii")
    st.metric("Equilibrium temp", f"{row['koi_teq']:.0f} K")
    st.metric("Star temp", f"{row['koi_steff']:.0f} K")

with right:
    st.subheader("Model verdict")
    st.metric("Prediction", pred_label, delta=f"{confidence:.0%} confidence")

    true_label = row["true_label"]
    if pred_label == true_label:
        st.success(f"NASA's actual disposition: {true_label} — model got it right.")
    else:
        st.error(f"NASA's actual disposition: {true_label} — model missed this one.")

    prob_df = pd.DataFrame({
        "disposition": label_encoder.classes_,
        "probability": probs,
    })
    prob_chart = alt.Chart(prob_df).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("disposition:N", sort="-x"),
        color=alt.condition(
            alt.datum.disposition == pred_label,
            alt.value("#F2C744"),
            alt.value("#3A3F4B"),
        ),
    )
    st.altair_chart(prob_chart, width="stretch")

st.divider()
st.subheader(f"Why {pred_label}? Top contributing features")

shap_values = explainer.shap_values(features)
row_shap = shap_values[0, :, pred_encoded]

shap_df = pd.DataFrame({
    "feature": feature_columns,
    "shap_value": row_shap,
})
shap_df["abs_value"] = shap_df["shap_value"].abs()
top_features = shap_df.sort_values("abs_value", ascending=False).head(10)

shap_chart = alt.Chart(top_features).mark_bar().encode(
    x=alt.X("shap_value:Q", title="Impact on prediction"),
    y=alt.Y("feature:N", sort=alt.EncodingSortField(field="abs_value", order="descending")),
    color=alt.condition(
        alt.datum.shap_value > 0,
        alt.value("#5FBF77"),
        alt.value("#E0697B"),
    ),
)
st.altair_chart(shap_chart, width="stretch")

with st.expander("How this works"):
    st.write(
        "This model is an XGBoost classifier trained on 9,564 Kepler Objects "
        "of Interest from NASA's cumulative KOI table, reaching 0.83 macro F1 "
        "across three classes. Leakage columns — NASA's own vetting flags and "
        "confidence scores — were explicitly identified and removed during "
        "cleaning. Full pipeline and notebooks are in this repo."
    )