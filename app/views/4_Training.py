"""Page 4: ML configuration and training."""

import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.invalidation import (
    commit_training_fingerprint,
    make_training_fingerprint,
    set_current_training_fingerprint,
)
from app.state import algorithm_ready, init_session_state, sampling_ready, training_ready
from monami.algorithms.registry import get_algorithm
from monami.config import MLConfig
from monami.ml import save_model_bundle, split_samples
from monami.training_stop import clear_manual_stop, request_manual_stop
from monami.viz import (
    LIVE_HISTORY_HEIGHT,
    LIVE_HISTORY_WIDTH,
    assemble_gif_from_png_frames,
    live_curves_placeholder_png,
    live_preview_canvas_size,
    live_preview_placeholder_png,
    preview_maps_to_png,
    sample_scatter,
    training_history_live_png,
)

init_session_state()

# Epochs per fragment execution — fewer reruns, Stop still checked every epoch inside fit().
_LIVE_EPOCH_BATCH = 3
_CONSOLE_LOG_LINES = 18
_PREVIEW_CAPTION_IDLE = (
    "Waiting for first preliminary prediction (generated every Preview interval epochs)."
)
_STATUS_IDLE = "Ready. Press Train model to start."
_STATUS_RUNNING = "Training in progress. Scroll freely — this panel keeps a fixed height."


def _ensure_preview_canvas_size() -> tuple[int, int]:
    """Lock preview canvas dimensions from the exhaustive grid (stable layout)."""
    cached = st.session_state.get("training_preview_canvas_size")
    if isinstance(cached, (tuple, list)) and len(cached) >= 2:
        return int(cached[0]), int(cached[1])
    truth = st.session_state.get("categorized_2d")
    if truth is not None:
        size = live_preview_canvas_size(tuple(truth.shape))
    else:
        size = (906, 280)
    st.session_state.training_preview_canvas_size = size
    return int(size[0]), int(size[1])


def _paint_preview_image(preview_slot, preview_caption, frame=None) -> None:
    """Show a grid-proportional preliminary map (placeholder until the first preview)."""
    canvas_w, canvas_h = _ensure_preview_canvas_size()
    png = st.session_state.get("training_preview_png") or live_preview_placeholder_png(
        canvas_w, canvas_h
    )
    preview_slot.image(
        png,
        caption="Preliminary exhaustive / samples / prediction",
        width=canvas_w,
    )
    if frame is not None:
        truth_grid = st.session_state.categorized_2d
        prediction = frame["prediction"]
        grid_acc = float(frame["grid_acc"])
        epoch = int(frame["epoch"])
        total = int(frame["total_epochs"])
        if truth_grid is not None:
            preview_caption.caption(
                f"Grid accuracy vs categorized truth at epoch {epoch}: **{grid_acc:.1%}** "
                f"({int((prediction == truth_grid).sum()):,}/{truth_grid.size:,} cells match)."
            )
            return
    preview_caption.caption(_PREVIEW_CAPTION_IDLE)


def _paint_metrics(metrics_slot, logs: dict) -> None:
    with metrics_slot.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Train accuracy", f"{logs.get('accuracy', 0.0):.3f}")
        m2.metric("Val accuracy", f"{logs.get('val_accuracy', 0.0):.3f}")
        m3.metric("Train loss", f"{logs.get('loss', 0.0):.3f}")
        m4.metric("Val loss", f"{logs.get('val_loss', 0.0):.3f}")


def _paint_live_curves(
    live_chart,
    history: dict,
    *,
    current_epoch=None,
    total_epochs=None,
) -> None:
    """Paint accuracy/loss as a fixed-size PNG (avoids Plotly iframe remount flicker)."""
    png = st.session_state.get("training_curves_png")
    if png is None:
        png = (
            training_history_live_png(
                history or {},
                current_epoch=current_epoch,
                total_epochs=total_epochs,
            )
            if (history or {}).get("accuracy")
            else live_curves_placeholder_png()
        )
    live_chart.image(png, width=LIVE_HISTORY_WIDTH)


def _format_training_log(lines) -> str:
    """Always return a fixed number of lines so the console height never grows."""
    recent = list(lines or [])[-_CONSOLE_LOG_LINES:]
    if len(recent) < _CONSOLE_LOG_LINES:
        recent = recent + [""] * (_CONSOLE_LOG_LINES - len(recent))
    return "\n".join(recent)


def _paint_training_log(log_panel, lines=None) -> None:
    log_panel.code(
        _format_training_log(lines if lines is not None else st.session_state.get("training_log")),
        language="text",
    )


def _restore_live_training_panels(
    preview_slot,
    preview_caption,
    metrics_slot,
    live_chart,
) -> None:
    """Repaint persisted training UI immediately so fragment remounts keep a stable height."""
    frames = st.session_state.get("training_preview_frames") or []
    history = st.session_state.get("live_training_history") or {}
    last = frames[-1] if frames else None
    _paint_preview_image(preview_slot, preview_caption, last)
    if last and last.get("logs"):
        _paint_metrics(metrics_slot, last["logs"])
    elif history.get("accuracy"):
        _paint_metrics(
            metrics_slot,
            {
                "accuracy": history["accuracy"][-1],
                "val_accuracy": (history.get("val_accuracy") or [0.0])[-1],
                "loss": (history.get("loss") or [0.0])[-1],
                "val_loss": (history.get("val_loss") or [0.0])[-1],
            },
        )
    else:
        _paint_metrics(
            metrics_slot,
            {"accuracy": 0.0, "val_accuracy": 0.0, "loss": 0.0, "val_loss": 0.0},
        )
    hist_len = len(history.get("accuracy") or [])
    if hist_len and not st.session_state.get("training_curves_png"):
        st.session_state.training_curves_png = training_history_live_png(
            history,
            current_epoch=hist_len,
            total_epochs=None,
        )
    _paint_live_curves(
        live_chart,
        history,
        current_epoch=hist_len or None,
        total_epochs=None,
    )


st.title("4. Model fitting / ML training")

if not sampling_ready():
    st.warning("Complete sampling on the **Sampling** page first.")
    st.stop()

if not algorithm_ready():
    st.warning("Select and apply an algorithm on the **Algorithm** page first.")
    st.stop()

algorithm = get_algorithm(st.session_state.selected_algorithm_id)
algo_config = st.session_state.algorithm_config

if not algorithm.supports_dnn_training_page():
    st.subheader("Statistical model fitting")
    st.caption(
        f"Algorithm: **{algorithm.name}** — {algorithm.feature_summary(algo_config)}"
    )
    st.info(
        "This algorithm fits category proportions and indicator variograms from "
        "**all sampled points**. DNN hyperparameters and train/test splitting are not used. "
        "The exhaustive field remains validation-only. "
        "An optional training image (TI) from Sampling is ignored here."
    )

    samples = st.session_state.samples_df
    grid_shape = (
        st.session_state.categorized_2d.shape
        if st.session_state.categorized_2d is not None
        else None
    )
    statistical_config = MLConfig(
        test_ratio=0.0,
        epochs=1,
        batch_size=1,
        nodes_per_layer=[],
        dropout=0.0,
        optimizer="indicator_kriging",
        suffix="SIS",
    )
    statistical_fingerprint = make_training_fingerprint(statistical_config)
    set_current_training_fingerprint(statistical_fingerprint)
    config_error = algorithm.validate_config(algo_config, samples)
    if config_error:
        st.error(config_error)
        st.stop()

    st.session_state.train_df = samples.copy()
    st.session_state.test_df = samples.iloc[0:0].copy()
    n_cat = int(
        st.session_state.get("n_categories_effective")
        or st.session_state.categories
    )
    st.plotly_chart(
        sample_scatter(
            samples,
            title=f"Hard conditioning data ({len(samples)} sampled points)",
            grid_shape=grid_shape,
            n_categories=n_cat,
        ),
        use_container_width=False,
    )

    if training_ready():
        st.success("**Statistical model ready.** Refit after changing sampling or algorithm settings.")
        fitted_model = st.session_state.get("model")
        if fitted_model is not None and hasattr(fitted_model, "variogram_for"):
            rows = []
            for category in fitted_model.categories:
                fitted = fitted_model.variogram_for(category)
                rows.append(
                    {
                        "Category": category,
                        "Proportion": fitted_model.proportions[str(category)],
                        "Model": fitted.model,
                        "Nugget": fitted.nugget,
                        "Sill": fitted.total_sill,
                        "Range X": fitted.range_x,
                        "Range Y": fitted.range_y,
                        "Fit RMSE": fitted.fit_rmse,
                        "Fallback": fitted.fallback,
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

    fit_model = st.button(
        "Fit statistical model",
        type="primary",
    )
    fit_log = st.empty()
    if fit_model:
        messages = []

        def append_fit_log(message: str) -> None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            messages.append(f"[{timestamp}] {message}")
            fit_log.code("\n".join(messages[-25:]), language="text")

        try:
            with st.spinner("Fitting sampled proportions and indicator variograms..."):
                result = algorithm.train(
                    samples,
                    samples.iloc[0:0].copy(),
                    samples,
                    tuple(grid_shape) if grid_shape is not None else None,
                    algo_config,
                    statistical_config,
                    log_callback=append_fit_log,
                )
                project_root = Path(__file__).resolve().parents[2]
                append_fit_log("Saving statistical model bundle...")
                model_path = save_model_bundle(
                    result.model,
                    result.meta,
                    project_root / "3_models",
                    statistical_config,
                    samples,
                    neighbor_pool_df=samples,
                )
        except Exception as exc:
            st.error(f"Statistical fitting failed: {exc}")
            st.exception(exc)
            st.stop()

        st.session_state.model = result.model
        st.session_state.model_meta = result.meta
        st.session_state.model_path = str(model_path)
        st.session_state.train_df = samples.copy()
        st.session_state.neighbor_pool_df = samples.copy()
        st.session_state.test_df = samples.iloc[0:0].copy()
        st.session_state.history = None
        st.session_state.live_training_history = None
        st.session_state.prediction_2d = None
        st.session_state.prediction_statistics = None
        st.session_state.simulations = []
        commit_training_fingerprint(statistical_fingerprint)
        st.success(
            f"Statistical model saved to `{model_path.name}` "
            f"({result.meta.training_seconds:.2f}s)."
        )
        st.rerun()
    st.stop()

if training_ready():
    st.info("**Model ready.** Change hyperparameters and click **Train model** to refresh.")
elif st.session_state.get("model_path") or st.session_state.history is not None:
    st.warning(
        "Training settings changed since the last run. "
        "Click **Train model** to refresh, then re-run prediction on Results."
    )

samples = st.session_state.samples_df
categories = st.session_state.categories

HIDDEN_ACTIVATIONS = ["relu", "tanh", "elu", "selu", "sigmoid"]
OUTPUT_ACTIVATIONS = ["softmax", "sigmoid", "linear"]

st.subheader("Hyperparameters")
st.caption(f"Algorithm: **{algorithm.name}** — {algorithm.feature_summary(algo_config)}")

with st.expander("Activation functions (ReLU vs softmax)", expanded=False):
    st.markdown(
        """
        The network uses **two different activations** on purpose:

        - **Hidden activation** (default: `relu`) — applied to internal layers. ReLU adds non-linearity
          so the model can learn complex patterns from neighbor features (dX, dY, D, V).
        - **Output activation** (default: `softmax`) — applied only to the last layer. Softmax turns
          outputs into class probabilities that sum to 1, which matches multi-class categorical
          training with `categorical_crossentropy`.

        ReLU is **not** a substitute for softmax on the output layer. Keep **softmax** unless you
        also change the loss function.
        """
    )

col1, col2 = st.columns(2)
with col1:
    test_ratio = st.slider(
        "Test ratio",
        0.05,
        0.5,
        0.2,
        0.05,
        help="Fraction of stratified samples reserved for validation. Recommended: **0.2** (20%).",
    )
    dropout = st.slider(
        "Dropout",
        0.0,
        0.5,
        0.2,
        0.05,
        help="Randomly drops hidden units during training to reduce overfitting. Recommended: **0.2**.",
    )
    batch_size = st.selectbox(
        "Batch size",
        [8, 16, 32, 64],
        index=2,
        help="Number of training rows per gradient update. Recommended: **16** (use 8 if memory is tight).",
    )
    hidden_activation = st.selectbox(
        "Hidden activation",
        HIDDEN_ACTIVATIONS,
        index=HIDDEN_ACTIVATIONS.index("relu"),
        help=(
            "Non-linearity for hidden dense layers. Recommended: **relu**. "
            "Try tanh or elu if training is unstable."
        ),
    )
    preview_interval = st.number_input(
        "Preview / log interval (epochs)",
        min_value=1,
        max_value=100,
        value=10,
        help=(
            "Update training curves, metrics, logs, and preliminary grid maps every N epochs. "
            "The final epoch is always reported when training stops."
        ),
    )
with col2:
    layers_str = st.text_input(
        "Hidden layer nodes (comma-separated)",
        "256,128,64,32",
        help=(
            "Number of neurons in each hidden layer, e.g. `256,128,64,32`. "
            "Recommended: **256,128,64** for ~150–1400 samples; use smaller layers for fewer samples."
        ),
    )
    optimizer = st.selectbox(
        "Optimizer",
        ["adadelta", "adam", "sgd"],
        index=1,
        help="Weight update algorithm. Recommended: **adadelta** (legacy default) or **adam** for faster convergence.",
    )
    out_activation = st.selectbox(
        "Output activation",
        OUTPUT_ACTIVATIONS,
        index=OUTPUT_ACTIVATIONS.index("softmax"),
        help=(
            "Activation on the final layer. Recommended: **softmax** for multi-class categorical "
            "prediction with categorical crossentropy. Other choices require a compatible loss."
        ),
    )
    suffix = st.text_input(
        "Run suffix",
        "R0",
        help="Short tag appended to saved model filenames to distinguish runs. Example: **R0**.",
    )

st.markdown("#### Convergence / stop criteria")
stop_mode = st.radio(
    "Primary stop criterion",
    [
        "Early stopping (validation accuracy)",
        "Target training accuracy",
    ],
    index=0,
    horizontal=True,
    help=(
        "Choose how training ends. **Target training accuracy** overrides max epochs and "
        "early stopping patience."
    ),
)
stop_on_train_accuracy = stop_mode == "Target training accuracy"

if stop_on_train_accuracy:
    st.info(
        "**Target training accuracy is active** and **overrides** max epochs and early stopping "
        "patience. Training stops as soon as train accuracy reaches the target below. "
        "Max epochs / patience controls are ignored (a large safety epoch cap still applies "
        "if the target is never reached)."
    )
else:
    st.caption(
        "Training stops when validation accuracy stops improving for the patience window, "
        "or when max epochs is reached — whichever comes first."
    )

sc1, sc2, sc3 = st.columns(3)
with sc1:
    epochs = st.number_input(
        "Max epochs",
        min_value=10,
        max_value=5000,
        value=1000,
        step=10,
        disabled=stop_on_train_accuracy,
        help=(
            "Maximum training passes when using early stopping. "
            "**Overridden** when target training accuracy is selected."
        ),
    )
with sc2:
    patience = st.number_input(
        "Early stopping patience",
        min_value=5,
        max_value=500,
        value=200,
        disabled=stop_on_train_accuracy,
        help=(
            "Stop if validation accuracy does not improve for this many epochs. "
            "**Overridden** when target training accuracy is selected."
        ),
    )
with sc3:
    target_train_accuracy = st.number_input(
        "Training accuracy target",
        min_value=0.50,
        max_value=1.00,
        value=0.95,
        step=0.01,
        format="%.2f",
        disabled=not stop_on_train_accuracy,
        help=(
            "Stop when **training** accuracy reaches this value (e.g. 0.95 = 95%). "
            "When selected as the primary criterion, this **overrides** max epochs and early stopping."
        ),
    )

if out_activation != "softmax":
    st.warning(
        "Output activation is not `softmax`. Ensure this matches the loss function "
        "(default: categorical_crossentropy expects softmax)."
    )

try:
    nodes = [int(x.strip()) for x in layers_str.split(",") if x.strip()]
except ValueError:
    st.error("Invalid layer specification.")
    st.stop()

ml_config = MLConfig(
    test_ratio=test_ratio,
    dropout=dropout,
    batch_size=int(batch_size),
    epochs=int(epochs),
    nodes_per_layer=nodes,
    optimizer=optimizer,
    early_stopping_patience=int(patience),
    stop_on_train_accuracy=bool(stop_on_train_accuracy),
    target_train_accuracy=float(target_train_accuracy),
    suffix=suffix,
    hidden_activation=hidden_activation,
    out_activation=out_activation,
    preview_interval=int(preview_interval),
)

set_current_training_fingerprint(make_training_fingerprint(ml_config))

train_df, test_df = split_samples(samples, ml_config.test_ratio, seed=st.session_state.random_seed)
config_error = algorithm.validate_config(algo_config, train_df)
if config_error:
    st.error(config_error)
    st.stop()

st.session_state.train_df = train_df
st.session_state.test_df = test_df
ti_samples_df = st.session_state.get("ti_samples_df")
ti_n = int(len(ti_samples_df)) if ti_samples_df is not None else 0
if ti_n:
    st.caption(
        f"Training labels: **{len(train_df):,}** target train + **{ti_n:,}** TI aux; "
        f"hard data for Results: **{len(train_df):,}** target train only "
        f"(TI Z = {st.session_state.get('ti_level')})."
    )
else:
    st.caption(
        f"Training labels: **{len(train_df):,}** train · **{len(test_df):,}** test "
        "(no training image)."
    )

grid_shape = st.session_state.categorized_2d.shape if st.session_state.categorized_2d is not None else None
n_cat = int(st.session_state.get("n_categories_effective") or st.session_state.categories)

# Keep sample scatters in a collapsed expander so Live training stays at a stable
# viewport height (expanding/collapsing these maps was shifting the page).
with st.expander(
    f"Training / test sample maps ({len(train_df)} train · {len(test_df)} test)",
    expanded=False,
):
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            sample_scatter(
                train_df,
                title=f"Training set ({len(train_df)} points)",
                grid_shape=grid_shape,
                n_categories=n_cat,
            ),
            use_container_width=False,
            key="training_set_scatter",
        )
    with c2:
        st.plotly_chart(
            sample_scatter(
                test_df,
                title=f"Test set ({len(test_df)} points)",
                grid_shape=grid_shape,
                n_categories=n_cat,
            ),
            use_container_width=False,
            key="test_set_scatter",
        )

st.subheader("Live training")
st.caption(
    f"Page height stays fixed while training so scrolling is not interrupted. "
    f"Preliminary maps refresh every {int(preview_interval)} epoch(s). "
    f"Stop keeps the weights with the best **training** accuracy."
)


@st.fragment
def _live_training_fragment() -> None:
    training_active = bool(st.session_state.get("training_active"))
    _, canvas_h = _ensure_preview_canvas_size()
    # Reserve fixed vertical space so fragment remounts never change page length.
    preview_box_h = int(canvas_h + 72)
    curves_box_h = int(LIVE_HISTORY_HEIGHT + 110)
    console_box_h = 320

    btn1, btn2 = st.columns(2)
    with btn1:
        start_train = st.button(
            "Train model",
            type="primary",
            disabled=training_active,
            key="train_model_btn",
        )
    with btn2:
        stop_train = st.button(
            "Stop training",
            disabled=not training_active,
            help="Stop after the current epoch and restore the best training-accuracy weights.",
            key="stop_train_btn",
        )

    st.markdown("#### 1. Preliminary prediction")
    with st.container(height=preview_box_h, border=False):
        preview_slot = st.empty()
        preview_caption = st.empty()

    st.markdown("#### 2. Accuracy and loss")
    with st.container(height=curves_box_h, border=False):
        metrics_slot = st.empty()
        live_chart = st.empty()

    st.markdown("#### 3. Training console")
    with st.container(height=console_box_h, border=False):
        progress = st.progress(0, text="Waiting to start training...")
        status = st.empty()
        log_panel = st.empty()

    # Always fill reserved panels immediately (placeholders if needed).
    _restore_live_training_panels(preview_slot, preview_caption, metrics_slot, live_chart)
    status.caption(_STATUS_RUNNING if training_active else _STATUS_IDLE)

    if "training_log" not in st.session_state:
        st.session_state.training_log = []

    def append_training_log(message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        st.session_state.training_log.append(line)
        _paint_training_log(log_panel, st.session_state.training_log)

    _paint_training_log(log_panel, st.session_state.training_log)

    if stop_train and training_active:
        request_manual_stop()
        status.caption(
            "Stop requested — finishing current epoch, then restoring best training-accuracy weights."
        )

    if start_train and not training_active:
        clear_manual_stop()
        st.session_state.training_active = True
        st.session_state.tr_warm_start = None
        st.session_state.training_log = []
        st.session_state.training_preview_frames = []
        st.session_state.training_preview_gif = None
        st.session_state.training_preview_png = None
        st.session_state.training_curves_png = None
        st.session_state.pop("training_preview_canvas_size", None)
        _ensure_preview_canvas_size()
        st.session_state._preview_png_frames = []
        st.session_state.training_preview_last_metrics = None
        st.session_state.live_training_history = {
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": [],
        }
        append_training_log(
            f"Train button clicked ({_LIVE_EPOCH_BATCH} epochs per UI refresh batch)."
        )
        append_training_log(
            f"Algorithm: {algorithm.name} ({algorithm.id}), config={algo_config}"
        )
        st.rerun(scope="fragment")

    if st.session_state.get("training_active"):
        truth_grid = st.session_state.categorized_2d
        samples_df = st.session_state.samples_df
        live_history = st.session_state.setdefault(
            "live_training_history",
            {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []},
        )
        preview_frames = st.session_state.setdefault("training_preview_frames", [])
        preview_png_frames = st.session_state.setdefault("_preview_png_frames", [])

        def on_epoch(epoch, total, logs, prediction=None):
            for key in live_history:
                if key in logs:
                    live_history[key].append(float(logs[key]))
            st.session_state.live_training_history = live_history
            progress.progress(min(epoch / max(total, 1), 1.0), text=f"Epoch {epoch}/{total}")
            _paint_metrics(metrics_slot, logs)
            curves_png = training_history_live_png(
                live_history,
                current_epoch=int(epoch),
                total_epochs=int(total),
            )
            st.session_state.training_curves_png = curves_png
            _paint_live_curves(
                live_chart,
                live_history,
                current_epoch=int(epoch),
                total_epochs=int(total),
            )
            if prediction is None:
                return

            logs_snapshot = {
                "accuracy": float(logs.get("accuracy", 0.0)),
                "val_accuracy": float(logs.get("val_accuracy", 0.0)),
                "loss": float(logs.get("loss", 0.0)),
                "val_loss": float(logs.get("val_loss", 0.0)),
            }
            grid_acc = (
                float((prediction == truth_grid).mean()) if truth_grid is not None else 0.0
            )
            frame = {
                "epoch": int(epoch),
                "total_epochs": int(total),
                "grid_acc": grid_acc,
                "prediction": prediction.copy(),
                "logs": logs_snapshot,
            }
            preview_frames.append(frame)
            st.session_state.training_preview_frames = preview_frames
            if truth_grid is not None and samples_df is not None:
                png = preview_maps_to_png(
                    truth_grid,
                    samples_df,
                    prediction,
                    int(epoch),
                    int(total),
                    n_categories=n_cat,
                    fast=True,
                )
                st.session_state.training_preview_png = png
                preview_png_frames.append(png)
                st.session_state._preview_png_frames = preview_png_frames
            _paint_preview_image(preview_slot, preview_caption, frame)

        status.caption(_STATUS_RUNNING)
        try:
            result = algorithm.train(
                train_df,
                test_df,
                train_df,
                tuple(truth_grid.shape) if truth_grid is not None else None,
                algo_config,
                ml_config,
                epoch_callback=on_epoch,
                log_callback=append_training_log,
                warm_start=st.session_state.get("tr_warm_start"),
                epochs_to_run=_LIVE_EPOCH_BATCH,
                ti_samples_df=ti_samples_df,
            )
        except Exception as exc:
            st.session_state.training_active = False
            st.session_state.tr_warm_start = None
            append_training_log(f"ERROR: {type(exc).__name__}: {exc}")
            status.error(f"Training failed: {exc}")
            st.exception(exc)
            return

        if result.incomplete:
            st.session_state.tr_warm_start = result.warm_start
            st.rerun(scope="fragment")

        # Finalize
        st.session_state.training_active = False
        st.session_state.tr_warm_start = None
        model = result.model
        history = result.history
        meta = result.meta
        meta.grid_shape = list(st.session_state.categorized_2d.shape)
        project_root = Path(__file__).resolve().parents[2]
        append_training_log("Saving model bundle...")
        model_path = save_model_bundle(
            model, meta, project_root / "3_models", ml_config, train_df
        )
        append_training_log(f"Model saved to {model_path.name}")
        st.session_state.model = model
        st.session_state.model_meta = meta
        st.session_state.model_path = str(model_path)
        st.session_state.train_df = train_df
        st.session_state.neighbor_pool_df = train_df
        st.session_state.history = history
        st.session_state.training_preview_last_metrics = (
            preview_frames[-1]["logs"] if preview_frames else None
        )
        pngs = list(st.session_state.pop("_preview_png_frames", []) or [])
        if pngs:
            try:
                st.session_state.training_preview_gif = assemble_gif_from_png_frames(
                    pngs, duration_ms=500
                )
                append_training_log(f"Built training preview GIF ({len(pngs)} frames).")
            except Exception as exc:
                st.session_state.training_preview_gif = None
                append_training_log(f"GIF build failed: {exc}")
        else:
            st.session_state.training_preview_gif = None
        if truth_grid is not None:
            st.session_state.prediction_2d = algorithm.predict_grid(
                model, truth_grid.shape, meta, train_df
            )
        commit_training_fingerprint(make_training_fingerprint(ml_config))
        progress.progress(1.0, text="Training complete")
        status.caption(
            f"Complete. Model saved to `{model_path.name}` ({meta.training_seconds:.1f}s)."
        )
        # Fragment-only rerun: avoid full-page remount that jumps scroll position.
        st.rerun(scope="fragment")

    elif st.session_state.history is not None:
        if not any(
            (st.session_state.get("live_training_history") or {}).get(key)
            for key in ("accuracy", "val_accuracy", "loss", "val_loss")
        ):
            final_png = training_history_live_png(
                st.session_state.history.history
                if hasattr(st.session_state.history, "history")
                else st.session_state.history
            )
            live_chart.image(final_png, width=LIVE_HISTORY_WIDTH)
        gif_bytes = st.session_state.get("training_preview_gif")
        frames = st.session_state.get("training_preview_frames") or []
        # Keep evolution optional so it cannot suddenly lengthen the page mid-session.
        with st.expander("Training evolution GIF (optional)", expanded=False):
            if gif_bytes and len(frames) > 1:
                st.image(gif_bytes, caption="Grid preview evolution across training epochs")
                st.download_button(
                    "Download training preview GIF",
                    data=gif_bytes,
                    file_name="training_preview_evolution.gif",
                    mime="image/gif",
                )
            else:
                st.caption("Available after a completed run with multiple preview frames.")


_live_training_fragment()
