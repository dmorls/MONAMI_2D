"""Page 4: Predict full grid and compare with truth/samples."""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from sklearn.metrics import classification_report

from app.state import init_session_state, sampling_ready, training_ready
from monami.ml import evaluate_at_points, load_model_bundle, load_neighbor_pool, predict_grid
from monami.viz import (
    confusion_matrix_plot,
    difference_map,
    exhaustive_sample_prediction_maps,
    histogram_categorical,
    observed_vs_predicted_scatter,
    overlaid_histograms,
)

init_session_state()

st.title("4. Results and comparison")

if not sampling_ready():
    st.warning("Complete sampling on the **Sampling** page first.")
    st.stop()

project_root = Path(__file__).resolve().parents[2]
model_dir = project_root / "3_models"
saved_models = sorted(model_dir.glob("*.h5")) if model_dir.exists() else []

source = st.radio("Model source", ["Current session model", "Load saved model"], horizontal=True)

model = st.session_state.model
meta = st.session_state.model_meta

if source == "Load saved model" and saved_models:
    selected = st.selectbox("Saved models", saved_models, format_func=lambda p: p.name)
    if st.button("Load selected model"):
        model, meta, neighbor_pool_df = load_model_bundle(selected)
        st.session_state.model = model
        st.session_state.model_meta = meta
        st.session_state.model_path = str(selected)
        st.session_state.neighbor_pool_df = neighbor_pool_df
        st.success(f"Loaded {selected.name}")
elif source == "Load saved model" and not saved_models:
    st.info("No saved models in `3_models/`. Train a model first.")
elif not training_ready():
    st.warning("Train a model on the **Training** page or load a saved model.")
    st.stop()

model = st.session_state.model
meta = st.session_state.model_meta
truth = st.session_state.categorized_2d
grid_shape = truth.shape

neighbor_pool_df = st.session_state.get("neighbor_pool_df")
if neighbor_pool_df is None and st.session_state.model_path:
    neighbor_pool_df = load_neighbor_pool(Path(st.session_state.model_path), meta)
    st.session_state.neighbor_pool_df = neighbor_pool_df

if neighbor_pool_df is None:
    st.error("Sample pool for neighbor lookup is missing. Retrain or reload a saved model bundle.")
    st.stop()

if st.button("Run full-grid prediction", type="primary"):
    with st.spinner("Predicting..."):
        start = time.time()
        prediction = predict_grid(model, grid_shape, meta, neighbor_pool_df)
        elapsed = time.time() - start
        st.session_state.prediction_2d = prediction
    st.success(f"Prediction completed in {elapsed:.2f}s")

if st.session_state.prediction_2d is None:
    st.stop()

prediction = st.session_state.prediction_2d
test_df = st.session_state.test_df
categories = st.session_state.categories
vrange = (0, int(categories) - 1)

tab1, tab2, tab3 = st.tabs(["Maps", "Distributions", "Metrics"])

with tab1:
    samples_df = st.session_state.samples_df
    st.plotly_chart(
        exhaustive_sample_prediction_maps(
            truth,
            samples_df,
            prediction,
            title="Exhaustive vs samples vs DNN prediction",
        ),
        use_container_width=False,
    )
    st.plotly_chart(difference_map(truth, prediction), use_container_width=False)

with tab2:
    st.plotly_chart(
        overlaid_histograms(
            st.session_state.samples_df["V"],
            prediction.ravel(),
            title="Sample vs prediction field",
            vrange=vrange,
        ),
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            histogram_categorical(truth.ravel(), title="Truth field", vrange=vrange, nbins=int(categories)),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            histogram_categorical(prediction.ravel(), title="Predicted field", vrange=vrange, nbins=int(categories)),
            use_container_width=True,
        )

with tab3:
    if test_df is not None and len(test_df) > 0:
        y_true = test_df["V"].astype(int).to_numpy()
        y_pred = evaluate_at_points(model, meta, test_df, neighbor_pool_df).astype(int)
        st.plotly_chart(confusion_matrix_plot(y_true, y_pred), use_container_width=True)
        st.plotly_chart(observed_vs_predicted_scatter(y_true, y_pred), use_container_width=True)
        st.text(classification_report(y_true, y_pred))
    else:
        st.info("No test split available. Train a model to generate test metrics.")

st.download_button(
    "Download prediction CSV",
    data=__import__("pandas").DataFrame(prediction).to_csv(index=False),
    file_name="prediction_field.csv",
    mime="text/csv",
)
