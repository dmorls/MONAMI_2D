"""Page 5: Predict full grid, sequential simulation, and compare with truth/samples."""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from sklearn.metrics import classification_report

from app.state import init_session_state, sampling_ready, training_ready
from monami.algorithms.registry import get_algorithm, uses_relative_position
from monami.ml import load_model_bundle, load_neighbor_pool
from monami.report import ReportContext, generate_report, sanitize_version
from monami.viz import (
    category_proportion_comparison,
    confusion_matrix_plot,
    exhaustive_sample_prediction_maps,
    observed_vs_predicted_scatter,
)

init_session_state()

st.title("5. Results and comparison")

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
        st.session_state.selected_algorithm_id = meta.algorithm_id
        st.session_state.algorithm_config = dict(meta.algorithm_config or {})
        st.session_state.simulations = []
        st.success(f"Loaded {selected.name}")
elif source == "Load saved model" and not saved_models:
    st.info("No saved models in `3_models/`. Train a model first.")
elif not training_ready():
    st.warning("Train a model on the **Training** page or load a saved model.")
    st.stop()

model = st.session_state.model
meta = st.session_state.model_meta
if model is None or meta is None:
    st.warning("Train a model on the **Training** page or load a saved model.")
    st.stop()

algorithm = get_algorithm(meta.algorithm_id)
truth = st.session_state.categorized_2d
grid_shape = truth.shape
samples_df = st.session_state.samples_df
categories = st.session_state.categories
test_df = st.session_state.test_df

neighbor_pool_df = st.session_state.get("neighbor_pool_df")
if neighbor_pool_df is None and st.session_state.model_path:
    neighbor_pool_df = load_neighbor_pool(Path(st.session_state.model_path), meta)
    st.session_state.neighbor_pool_df = neighbor_pool_df

hard_df = st.session_state.get("train_df")
if hard_df is None:
    hard_df = neighbor_pool_df

if hard_df is None or len(hard_df) == 0:
    st.error("Training sample pool is missing. Retrain or reload a saved model bundle.")
    st.stop()

st.caption(f"Algorithm: **{algorithm.name}** (`{meta.algorithm_id}`)")


def _render_field_block(grid, title: str, prediction_label: str) -> None:
    """Show field maps, then category proportions (exhaustive / training / prediction)."""
    st.plotly_chart(
        exhaustive_sample_prediction_maps(
            truth,
            samples_df,
            grid,
            title=title,
            n_categories=int(categories),
        ),
        use_container_width=False,
    )
    st.plotly_chart(
        category_proportion_comparison(
            truth,
            hard_df,
            grid,
            n_categories=int(categories),
            title=f"Category proportions — {prediction_label}",
            prediction_label=prediction_label,
        ),
        use_container_width=True,
    )


# --- Most-likely prediction ---
st.subheader("Most-likely prediction")
if uses_relative_position(algorithm.id):
    st.caption(
        "Argmax of the DNN softmax at every cell (deterministic). "
        "Neighbor features use the **training samples** pool."
    )
else:
    st.caption(
        "Argmax of the DNN softmax at every cell (deterministic). "
        "Features are normalized **X, Y** at each cell."
    )
if st.button("Run full-grid prediction", type="primary"):
    with st.spinner("Predicting..."):
        start = time.time()
        prediction = algorithm.predict_grid(model, grid_shape, meta, hard_df)
        elapsed = time.time() - start
        st.session_state.prediction_2d = prediction
    st.success(f"Prediction completed in {elapsed:.2f}s")

if st.session_state.prediction_2d is not None:
    _render_field_block(
        st.session_state.prediction_2d,
        title="Exhaustive vs samples vs prediction",
        prediction_label="Prediction (most likely)",
    )

# --- Sequential simulation ---
st.subheader("Sequential simulation")
if uses_relative_position(algorithm.id):
    st.caption(
        "Classic sequential path over unsampled cells. Hard data = **training samples**. "
        "At each cell, relative-position neighbor features use the growing conditioning pool; "
        "a category is sampled from the DNN softmax. "
        "Each realization is shown as soon as it finishes (maps, then category proportions)."
    )
else:
    st.caption(
        "Classic sequential path over unsampled cells. Hard data = **training samples**. "
        "At each cell, features are that cell's normalized **X, Y**; "
        "a category is sampled from the DNN softmax. "
        "Each realization is shown as soon as it finishes (maps, then category proportions)."
    )
c1, c2 = st.columns(2)
with c1:
    n_realizations = st.number_input(
        "Number of realizations",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        help="How many sequential simulations to append this run.",
    )
with c2:
    sim_seed = st.number_input(
        "Base random seed",
        min_value=0,
        max_value=10_000_000,
        value=int(st.session_state.random_seed),
        help="Realization i uses seed = base + i − 1.",
    )

progress = st.progress(0.0, text="Waiting to run sequential simulation...")
status = st.empty()
live_gallery = st.container()

run_sims = st.button("Run sequential simulation")

if run_sims:
    sims = list(st.session_state.get("simulations") or [])
    start_all = time.time()
    n_real = int(n_realizations)

    with live_gallery:
        for sim in sims:
            st.markdown(f"### {sim['label']}")
            st.caption(f"Seed = {sim['seed']}")
            _render_field_block(
                sim["grid"],
                title=f"Exhaustive vs samples vs {sim['label']}",
                prediction_label=sim["label"],
            )

    for r in range(n_real):
        seed = int(sim_seed) + r
        label_num = len(sims) + 1
        label = f"Simulation {label_num}"
        status.info(f"Running {label} ({r + 1}/{n_real}, seed={seed})...")

        def _on_progress(done: int, total: int, _r=r, _n=n_real, _label=label) -> None:
            overall = (_r + done / max(total, 1)) / max(_n, 1)
            progress.progress(
                min(overall, 1.0),
                text=f"{_label} — cell {done:,}/{total:,}",
            )

        t0 = time.time()
        grid = algorithm.simulate_grid(
            model,
            meta,
            hard_df,
            grid_shape,
            seed=seed,
            progress_callback=_on_progress,
        )
        elapsed = time.time() - t0
        entry = {"label": label, "grid": grid, "seed": seed}
        sims.append(entry)
        st.session_state.simulations = list(sims)

        with live_gallery:
            st.markdown(f"### {label}")
            st.caption(f"Seed = {seed} · finished in {elapsed:.1f}s")
            _render_field_block(
                grid,
                title=f"Exhaustive vs samples vs {label}",
                prediction_label=label,
            )
        status.success(f"{label} finished in {elapsed:.1f}s (seed={seed})")

    progress.progress(1.0, text="Sequential simulation complete")
    status.success(
        f"Appended {n_real} realization(s) in {time.time() - start_all:.1f}s. "
        f"Total stored: {len(sims)}."
    )
else:
    simulations = st.session_state.get("simulations") or []
    if simulations:
        st.markdown("#### Stored realizations")
        for sim in simulations:
            st.markdown(f"### {sim['label']}")
            st.caption(f"Seed = {sim['seed']}")
            _render_field_block(
                sim["grid"],
                title=f"Exhaustive vs samples vs {sim['label']}",
                prediction_label=sim["label"],
            )

simulations = st.session_state.get("simulations") or []
has_prediction = st.session_state.prediction_2d is not None
if not has_prediction and not simulations:
    st.info("Run a full-grid prediction and/or sequential simulation to view results.")
    st.stop()

# --- Metrics (most-likely prediction) ---
st.subheader("Test-set metrics")
st.caption("Confusion matrix and scores use the most-likely (argmax) prediction only.")

if has_prediction and test_df is not None and len(test_df) > 0:
    y_true = test_df["V"].astype(int).to_numpy()
    y_pred = algorithm.evaluate_at_points(model, meta, test_df, hard_df).astype(int)
    st.plotly_chart(confusion_matrix_plot(y_true, y_pred), use_container_width=True)
    st.plotly_chart(observed_vs_predicted_scatter(y_true, y_pred), use_container_width=True)
    st.text(classification_report(y_true, y_pred))
elif has_prediction:
    st.info("No test split available. Train a model to generate test metrics.")
else:
    st.info("Run full-grid prediction to view test-set metrics.")

view_options = []
if has_prediction:
    view_options.append("Prediction (most likely)")
for sim in simulations:
    view_options.append(sim["label"])

selected_view = st.selectbox("Field to download", view_options, index=0)
if selected_view == "Prediction (most likely)":
    display_grid = st.session_state.prediction_2d
    download_name = "prediction_field.csv"
else:
    sim = next(s for s in simulations if s["label"] == selected_view)
    display_grid = sim["grid"]
    download_name = f"{selected_view.lower().replace(' ', '_')}.csv"

st.download_button(
    "Download selected field CSV",
    data=__import__("pandas").DataFrame(display_grid).to_csv(index=False),
    file_name=download_name,
    mime="text/csv",
)

# --- Export PDF report ---
st.subheader("Export report")
st.caption(
    "Generate a shareable PDF covering the full workflow (data, sampling, algorithm, training, "
    "results, and metrics). Saved under `4_reports/` as `Monami_<version>_<timestamp>.pdf`."
)

report_version = st.text_input(
    "Version series",
    value="v0.1",
    help="Manual tag included in the filename, e.g. v0.1 → Monami_v0.1_YYYYMMDD_HHMMSS.pdf",
)

gen_report = st.button("Generate PDF report", type="primary", key="generate_pdf_report")

if gen_report:
    try:
        sanitize_version(report_version)
    except ValueError as exc:
        st.error(str(exc))
    else:
        y_true_test = None
        y_pred_test = None
        clf_text = ""
        if has_prediction and test_df is not None and len(test_df) > 0:
            y_true_test = test_df["V"].astype(int).to_numpy()
            y_pred_test = algorithm.evaluate_at_points(model, meta, test_df, hard_df).astype(int)
            clf_text = classification_report(y_true_test, y_pred_test)

        # Best-effort stop-criteria summary from model meta / defaults
        stop_summary = ""
        try:
            # Reconstruct a lightweight view if session retained last MLConfig fields via meta only
            stop_summary = (
                f"algorithm={meta.algorithm_id}; "
                f"test_ratio={meta.test_ratio}; "
                f"layers={list(meta.nodes_per_layer)}; "
                f"dropout={meta.dropout}; optimizer={meta.optimizer}"
            )
        except Exception:
            stop_summary = ""

        ctx = ReportContext(
            version=report_version,
            project_root=project_root,
            source_name=str(st.session_state.get("source_name") or ""),
            property_name=str(st.session_state.get("property_name") or ""),
            selected_level=int(st.session_state.get("selected_level") or 0),
            grid_meta=st.session_state.get("grid_meta"),
            categories=int(categories),
            sample_pct=float(st.session_state.get("sample_pct") or 0.0),
            random_seed=int(st.session_state.get("random_seed") or 0),
            samples_df=samples_df,
            train_df=hard_df,
            test_df=test_df,
            truth=truth,
            algorithm_id=algorithm.id,
            algorithm_name=algorithm.name,
            algorithm_description=algorithm.description,
            algorithm_long_description=getattr(algorithm, "long_description", "") or "",
            algorithm_config=dict(meta.algorithm_config or {}),
            meta=meta,
            prediction_2d=st.session_state.prediction_2d,
            simulations=list(simulations),
            history=st.session_state.get("history"),
            live_training_history=st.session_state.get("live_training_history"),
            y_true_test=y_true_test,
            y_pred_test=y_pred_test,
            classification_report_text=clf_text,
            model_path=str(st.session_state.get("model_path") or ""),
            stop_criteria_summary=stop_summary,
        )

        with st.spinner("Building PDF report..."):
            try:
                pdf_path = generate_report(ctx, project_root / "4_reports")
                st.session_state.last_report_path = str(pdf_path)
                st.success(f"Report saved to `{pdf_path}`")
            except Exception as exc:
                st.error(f"Failed to generate report: {exc}")

last_report = st.session_state.get("last_report_path")
if last_report and Path(last_report).is_file():
    pdf_bytes = Path(last_report).read_bytes()
    st.download_button(
        "Download last PDF report",
        data=pdf_bytes,
        file_name=Path(last_report).name,
        mime="application/pdf",
        key="download_last_pdf_report",
    )

