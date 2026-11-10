"""Streamlit app for interactive Auto MPG predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predictive_regression.predict import load_or_train_model, predict_mpg  # noqa: E402

st.set_page_config(
    page_title="Auto MPG Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Auto MPG Predictor")
st.caption("Leakage-safe scikit-learn Ridge regression served with Streamlit")
st.markdown(
    "Estimate a vehicle's fuel economy from its technical specifications. "
    "The underlying model uses imputation, scaling, one-hot encoding, and Ridge regularization."
)

@st.cache_resource(show_spinner=False)
def get_model_and_metrics():
    """Cache the model for the lifetime of the Streamlit process."""
    return load_or_train_model()


with st.spinner("Loading the trained model…"):
    model, metrics = get_model_and_metrics()

with st.sidebar:
    st.header("Model information")
    test_metrics = metrics.get("test_metrics", {})
    if test_metrics:
        st.metric("Held-out R²", f"{test_metrics.get('r2', 0):.3f}")
        st.metric("Held-out RMSE", f"{test_metrics.get('rmse', 0):.2f} MPG")
        st.caption(f"Ridge alpha: {metrics.get('model', {}).get('best_alpha', 'n/a')}")
    st.divider()
    st.markdown("**Dataset:** Auto MPG  \\n**Target:** miles per gallon  \\n**Split:** 80/20, seed 42")

st.subheader("Vehicle specifications")
with st.form("prediction_form"):
    left, middle, right = st.columns(3)
    with left:
        cylinders = st.number_input("Cylinders", min_value=3, max_value=12, value=4, step=1)
        displacement = st.number_input("Displacement", min_value=40.0, max_value=500.0, value=140.0, step=1.0)
        horsepower = st.number_input("Horsepower", min_value=40.0, max_value=250.0, value=95.0, step=1.0)
    with middle:
        weight = st.number_input("Weight (lbs)", min_value=1500.0, max_value=5500.0, value=2500.0, step=50.0)
        acceleration = st.number_input("Acceleration (0–60 sec)", min_value=8.0, max_value=25.0, value=15.0, step=0.1)
    with right:
        model_year = st.number_input("Model year", min_value=70, max_value=82, value=76, step=1)
        origin = st.selectbox("Origin", options=["1", "2", "3"], format_func=lambda x: {"1": "USA", "2": "Europe", "3": "Japan"}[x])

    submitted = st.form_submit_button("Predict fuel economy", type="primary", use_container_width=True)

if submitted:
    features = {
        "cylinders": cylinders,
        "displacement": displacement,
        "horsepower": horsepower,
        "weight": weight,
        "acceleration": acceleration,
        "model_year": model_year,
        "origin": origin,
    }
    prediction = predict_mpg(model, features)
    st.success(f"Estimated fuel economy: **{prediction:.1f} MPG**")
    st.info("This estimate is for model exploration and should not be treated as an official vehicle specification.")

with st.expander("How this works"):
    st.markdown(
        "1. Numeric inputs are mean-imputed and standardized.\n"
        "2. Origin is imputed and one-hot encoded.\n"
        "3. A Ridge regression model applies regularization.\n"
        "4. The full preprocessing and model pipeline produces the prediction."
    )
