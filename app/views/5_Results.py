"""Page 5: Predict full grid, sequential simulation, and compare with truth/samples."""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.state import init_session_state, sampling_ready, training_ready
from monami.algorithms.registry import get_algorithm
from monami.geostats import corrected_sis_validation_metrics
from monami.ml import load_model_bundle, load_neighbor_pool
from monami.report import ReportContext, generate_report, sanitize_version
from monami.viz import (
    category_proportion_comparison,
    exhaustive_sample_prediction_maps,
)

init_session_state()

st.title("5. Results and comparison")

if not sampling_ready():
    st.warning("Complete sampling on the **Sampling** page first.")
    st.stop()

project_root = Path(__file__).resolve().parents[2]
model_dir = project_root / "3_models"
saved_models = (
    sorted(
        list(model_dir.glob("*.h5"))
        + [
            path
            for path in model_dir.glob("*.json")
            if not path.name.endswith("_meta.json")
        ]
    )
    if model_dir.exists()
    else []
)


def _bundle_compatibility_error(bundle_meta, pool_df):
    """Reject bundles that do not belong to the active sampled workflow."""
    active_truth = st.session_state.categorized_2d
    active_samples = st.session_state.samples_df
    expected_shape = tuple(active_truth.shape)
    saved_shape = tuple(int(v) for v in (bundle_meta.grid_shape or []))
    if saved_shape and saved_shape != expected_shape:
        return (
            f"Saved grid shape {saved_shape} does not match the active grid "
            f"{expected_shape}."
        )
    active_categories = int(
        st.session_state.get("n_categories_effective")
        or st.session_state.categories
    )
    if int(bundle_meta.n_classes) != active_categories:
        return (
            f"Saved model has {bundle_meta.n_classes} categories, but the active "
            f"workflow has {active_categories}."
        )
    rows, cols = expected_shape
    x = pool_df["X"].to_numpy(dtype=int)
    y = pool_df["Y"].to_numpy(dtype=int)
    if ((x < 1) | (x > cols) | (y < 1) | (y > rows)).any():
        return "Saved conditioning coordinates fall outside the active grid."
    active_rows = {
        (int(row.X), int(row.Y), int(row.V))
        for row in active_samples[["X", "Y", "V"]].itertuples(index=False)
    }
    saved_rows = {
        (int(row.X), int(row.Y), int(row.V))
        for row in pool_df[["X", "Y", "V"]].itertuples(index=False)
    }
    if not saved_rows.issubset(active_rows):
        return (
            "Saved conditioning samples do not match the active sampling run. "
            "Restore the matching sample configuration or fit a new model."
        )
    return None


source = st.radio("Model source", ["Current session model", "Load saved model"], horizontal=True)

model = st.session_state.model
meta = st.session_state.model_meta

if source == "Load saved model" and saved_models:
    selected = st.selectbox("Saved models", saved_models, format_func=lambda p: p.name)
    if st.button("Load selected model"):
        loaded_model, loaded_meta, loaded_pool = load_model_bundle(selected)
        compatibility_error = _bundle_compatibility_error(loaded_meta, loaded_pool)
        if compatibility_error:
            st.error(f"Cannot load `{selected.name}`: {compatibility_error}")
        else:
            model, meta, neighbor_pool_df = loaded_model, loaded_meta, loaded_pool
            st.session_state.model = model
            st.session_state.model_meta = meta
            st.session_state.model_path = str(selected)
            st.session_state.neighbor_pool_df = neighbor_pool_df
            st.session_state.train_df = neighbor_pool_df.copy()
            st.session_state.test_df = None
            st.session_state.selected_algorithm_id = meta.algorithm_id
            st.session_state.algorithm_config = dict(meta.algorithm_config or {})
            st.session_state.prediction_2d = None
            st.session_state.prediction_statistics = None
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


def _statistical_metrics(grid, seed: int = 42):
    if getattr(meta, "model_type", "keras") != "corrected_sis":
        return None
    return corrected_sis_validation_metrics(
        grid,
        hard_df,
        model,
        seed=int(seed),
    )


def _render_field_block(
    grid,
    title: str,
    prediction_label: str,
    *,
    statistics=None,
    seed: int = 42,
) -> None:
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
    metrics = statistics if statistics is not None else _statistical_metrics(grid, seed=seed)
    if metrics is not None:
        st.caption("Sample-derived statistical fidelity (exhaustive truth is not used)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hard-data fidelity", f"{metrics['hard_data_fidelity']:.1%}")
        m2.metric("Proportion L1", f"{metrics['proportion_l1']:.4f}")
        m3.metric("Proportion RMSE", f"{metrics['proportion_rmse']:.4f}")
        vrmse = metrics["variogram_rmse_mean"]
        m4.metric("Indicator variogram RMSE", f"{vrmse:.4f}" if vrmse >= 0 else "n/a")
        tx = metrics["transition_error_x"]
        ty = metrics["transition_error_y"]
        tx_text = f"{tx:.4f}" if tx >= 0 else "n/a"
        ty_text = f"{ty:.4f}" if ty >= 0 else "n/a"
        lag_x = int(metrics.get("transition_lag_x", 1))
        lag_y = int(metrics.get("transition_lag_y", 1))
        st.caption(
            "Directional transition error (lower is better): "
            f"X lag {lag_x} = {tx_text} · Y lag {lag_y} = {ty_text}"
        )


# --- Most-likely prediction ---
st.subheader("Most-likely prediction")
st.caption(algorithm.prediction_description())
if st.button("Run full-grid prediction", type="primary"):
    with st.spinner("Predicting..."):
        start = time.time()
        prediction = algorithm.predict_grid(model, grid_shape, meta, hard_df)
        elapsed = time.time() - start
        st.session_state.prediction_2d = prediction
        st.session_state.prediction_statistics = _statistical_metrics(
            prediction,
            seed=int(st.session_state.random_seed),
        )
    st.success(f"Prediction completed in {elapsed:.2f}s")

if st.session_state.prediction_2d is not None:
    _render_field_block(
        st.session_state.prediction_2d,
        title="Exhaustive vs samples vs prediction",
        prediction_label="Prediction (most likely)",
        statistics=st.session_state.get("prediction_statistics"),
        seed=int(st.session_state.random_seed),
    )

# --- Sequential simulation ---
st.subheader("Sequential simulation")
st.caption(
    algorithm.simulation_description()
    + " Each realization is shown as soon as it finishes."
)
is_dnn_sim = getattr(meta, "model_type", "keras") != "corrected_sis"
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

sim_correction_strength = 0.5
if is_dnn_sim:
    sim_correction_strength = st.slider(
        "Sample-proportion correction",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=(
            "Blend local DNN probabilities with the remaining training-sample "
            "histogram quota at each sequential draw. 0 = pure DNN softmax; "
            "1 = hard quota. Recommended: 0.50."
        ),
        key="dnn_sim_correction_strength",
    )

progress = st.progress(0.0, text="Waiting to run sequential simulation...")
status = st.empty()
live_gallery = st.container()

run_sims = st.button("Run sequential simulation")


def _sim_caption(sim: dict) -> str:
    parts = [f"Seed = {sim['seed']}"]
    if sim.get("elapsed") is not None:
        parts.append(f"finished in {float(sim['elapsed']):.1f}s")
    if sim.get("correction_strength") is not None and is_dnn_sim:
        parts.append(f"proportion correction = {float(sim['correction_strength']):.2f}")
    return " · ".join(parts)


if run_sims:
    sims = list(st.session_state.get("simulations") or [])
    start_all = time.time()
    n_real = int(n_realizations)
    strength = float(sim_correction_strength)

    with live_gallery:
        for sim in sims:
            st.markdown(f"### {sim['label']}")
            st.caption(_sim_caption(sim))
            _render_field_block(
                sim["grid"],
                title=f"Exhaustive vs samples vs {sim['label']}",
                prediction_label=sim["label"],
                statistics=sim.get("statistics"),
                seed=int(sim["seed"]),
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
            correction_strength=strength,
        )
        elapsed = time.time() - t0
        entry = {
            "label": label,
            "grid": grid,
            "seed": seed,
            "elapsed": elapsed,
            "correction_strength": strength if is_dnn_sim else None,
            "statistics": _statistical_metrics(grid, seed=seed),
        }
        sims.append(entry)
        st.session_state.simulations = list(sims)

        with live_gallery:
            st.markdown(f"### {label}")
            st.caption(_sim_caption(entry))
            _render_field_block(
                grid,
                title=f"Exhaustive vs samples vs {label}",
                prediction_label=label,
                statistics=entry.get("statistics"),
                seed=seed,
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
            st.caption(_sim_caption(sim))
            _render_field_block(
                sim["grid"],
                title=f"Exhaustive vs samples vs {sim['label']}",
                prediction_label=sim["label"],
                statistics=sim.get("statistics"),
                seed=int(sim["seed"]),
            )

simulations = st.session_state.get("simulations") or []
has_prediction = st.session_state.prediction_2d is not None
if not has_prediction and not simulations:
    st.info("Run a full-grid prediction and/or sequential simulation to view results.")
    st.stop()

is_statistical = getattr(meta, "model_type", "keras") == "corrected_sis"
if is_statistical:
    st.subheader("Statistical validation")
    st.caption(
        "Corrected SIS uses all samples as hard data. Sample-derived statistical "
        "metrics are shown with each prediction/realization; exhaustive maps remain "
        "an independent visual validation."
    )

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
    "and results). Saved under `4_reports/` as `Monami_<version>_<timestamp>.pdf`."
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
        # Best-effort stop-criteria summary from model meta / defaults
        stop_summary = ""
        try:
            # Reconstruct a lightweight view if session retained last MLConfig fields via meta only
            if getattr(meta, "model_type", "keras") == "keras":
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
            prediction_description=algorithm.prediction_description(),
            simulation_description=algorithm.simulation_description(),
            statistical_model=(
                model.to_dict() if hasattr(model, "to_dict") else {}
            ),
            prediction_2d=st.session_state.prediction_2d,
            prediction_statistics=st.session_state.get("prediction_statistics"),
            simulations=list(simulations),
            history=st.session_state.get("history"),
            live_training_history=st.session_state.get("live_training_history"),
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

