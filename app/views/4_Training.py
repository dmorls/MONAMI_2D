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
    assemble_gif_from_png_frames,
    exhaustive_sample_prediction_maps,
    preview_maps_to_png,
    sample_scatter,
    training_history_live_plot,
    training_history_plot,
)

init_session_state()


def _render_training_previews() -> None:
    """Show persisted grid previews and training evolution GIF after a run."""
    frames = st.session_state.get("training_preview_frames") or []
    if not frames:
        return

    st.subheader("Grid previews")
    last = frames[-1]
    truth_grid = st.session_state.categorized_2d
    samples_df = st.session_state.samples_df
    if truth_grid is None or samples_df is None:
        return

    last_metrics = st.session_state.get("training_preview_last_metrics") or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train accuracy", f"{last_metrics.get('accuracy', 0.0):.3f}")
    m2.metric("Val accuracy", f"{last_metrics.get('val_accuracy', 0.0):.3f}")
    m3.metric("Train loss", f"{last_metrics.get('loss', 0.0):.3f}")
    m4.metric("Val loss", f"{last_metrics.get('val_loss', 0.0):.3f}")

    n_cat = int(st.session_state.get("n_categories_effective") or st.session_state.categories)
    st.plotly_chart(
        exhaustive_sample_prediction_maps(
            truth_grid,
            samples_df,
            last["prediction"],
            title=(
                f"Exhaustive / samples / prediction — epoch "
                f"{last['epoch']}/{last['total_epochs']}"
            ),
            n_categories=n_cat,
        ),
        use_container_width=False,
    )
    st.caption(
        f"Final preview at epoch {last['epoch']}: grid accuracy **{last['grid_acc']:.1%}** "
        f"({int((last['prediction'] == truth_grid).sum()):,}/{truth_grid.size:,} cells match). "
        f"Captured **{len(frames)}** preview frame(s) during training."
    )

    gif_bytes = st.session_state.get("training_preview_gif")
    if gif_bytes and len(frames) > 1:
        st.subheader("Training evolution")
        st.image(gif_bytes, caption="Grid preview evolution across training epochs")
        st.download_button(
            "Download training preview GIF",
            data=gif_bytes,
            file_name="training_preview_evolution.gif",
            mime="image/gif",
        )


st.title("4. ML training")

if not sampling_ready():
    st.warning("Complete sampling on the **Sampling** page first.")
    st.stop()

if not algorithm_ready():
    st.warning("Select and apply an algorithm on the **Algorithm** page first.")
    st.stop()

algorithm = get_algorithm(st.session_state.selected_algorithm_id)
algo_config = st.session_state.algorithm_config

if not algorithm.supports_dnn_training_page():
    st.info("This algorithm trains on the **Algorithm** page. DNN hyperparameters are not used.")
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

c1, c2 = st.columns(2)
grid_shape = st.session_state.categorized_2d.shape if st.session_state.categorized_2d is not None else None
n_cat = int(st.session_state.get("n_categories_effective") or st.session_state.categories)
with c1:
    st.plotly_chart(
        sample_scatter(
            train_df,
            title=f"Training set ({len(train_df)} points)",
            grid_shape=grid_shape,
            n_categories=n_cat,
        ),
        use_container_width=False,
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
    )

st.subheader("Live training")
st.caption(
    f"Training runs on the main thread one epoch at a time so you can press **Stop training** "
    f"between epochs. Charts refresh every {int(preview_interval)} epoch(s). "
    f"Stop keeps the weights with the best **training** accuracy so far."
)

training_active = bool(st.session_state.get("training_active"))

btn1, btn2 = st.columns(2)
with btn1:
    start_train = st.button(
        "Train model",
        type="primary",
        disabled=training_active,
    )
with btn2:
    stop_train = st.button(
        "Stop training",
        disabled=not training_active,
        help="Stop after the current epoch and restore the best training-accuracy weights.",
    )

live_chart = st.empty()
metrics_slot = st.empty()
preview_slot = st.empty()
preview_caption = st.empty()
progress = st.progress(0, text="Waiting to start training...")
status = st.empty()
log_panel = st.empty()

if "training_log" not in st.session_state:
    st.session_state.training_log = []


def append_training_log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    st.session_state.training_log.append(line)
    log_panel.code("\n".join(st.session_state.training_log[-25:]), language="text")


if st.session_state.training_log:
    log_panel.code("\n".join(st.session_state.training_log[-25:]), language="text")

if stop_train and training_active:
    request_manual_stop()
    status.warning(
        "Stop requested — finishing the current epoch, then restoring best training-accuracy weights."
    )

if start_train and not training_active:
    clear_manual_stop()
    st.session_state.training_active = True
    st.session_state.tr_warm_start = None
    st.session_state.training_log = []
    st.session_state.training_preview_frames = []
    st.session_state.training_preview_gif = None
    st.session_state.training_preview_last_metrics = None
    st.session_state.live_training_history = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
    }
    append_training_log("Train button clicked (main-thread epoch steps).")
    append_training_log(f"Algorithm: {algorithm.name} ({algorithm.id}), config={algo_config}")
    st.rerun()

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
        with metrics_slot.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train accuracy", f"{logs.get('accuracy', 0.0):.3f}")
            m2.metric("Val accuracy", f"{logs.get('val_accuracy', 0.0):.3f}")
            m3.metric("Train loss", f"{logs.get('loss', 0.0):.3f}")
            m4.metric("Val loss", f"{logs.get('val_loss', 0.0):.3f}")
        live_chart.plotly_chart(
            training_history_live_plot(
                live_history, current_epoch=epoch, total_epochs=total
            ),
            use_container_width=True,
        )
        if prediction is None:
            return
        logs_snapshot = {
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
            "loss": float(logs.get("loss", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
        }
        grid_acc = float((prediction == truth_grid).mean()) if truth_grid is not None else 0.0
        preview_frames.append(
            {
                "epoch": int(epoch),
                "total_epochs": int(total),
                "grid_acc": grid_acc,
                "prediction": prediction.copy(),
                "logs": logs_snapshot,
            }
        )
        st.session_state.training_preview_frames = preview_frames
        if truth_grid is not None and samples_df is not None:
            try:
                preview_png_frames.append(
                    preview_maps_to_png(
                        truth_grid, samples_df, prediction, epoch, total, n_categories=n_cat
                    )
                )
                st.session_state._preview_png_frames = preview_png_frames
            except Exception as exc:
                append_training_log(f"Preview frame capture skipped at epoch {epoch}: {exc}")
            preview_slot.plotly_chart(
                exhaustive_sample_prediction_maps(
                    truth_grid,
                    samples_df,
                    prediction,
                    title=f"Exhaustive / samples / prediction — epoch {epoch}/{total}",
                    n_categories=n_cat,
                ),
                use_container_width=False,
            )
            preview_caption.caption(
                f"Grid accuracy vs categorized truth at epoch {epoch}: **{grid_acc:.1%}** "
                f"({int((prediction == truth_grid).sum()):,}/{truth_grid.size:,} cells match)."
            )

    status.info("Training in progress (main thread). You can press **Stop training** between epochs.")
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
            epochs_to_run=1,
        )
    except Exception as exc:
        st.session_state.training_active = False
        st.session_state.tr_warm_start = None
        append_training_log(f"ERROR: {type(exc).__name__}: {exc}")
        status.error(f"Training failed: {exc}")
        st.exception(exc)
        st.stop()

    if result.incomplete:
        st.session_state.tr_warm_start = result.warm_start
        st.rerun()

    # Finalize
    st.session_state.training_active = False
    st.session_state.tr_warm_start = None
    model = result.model
    history = result.history
    meta = result.meta
    meta.grid_shape = list(st.session_state.categorized_2d.shape)
    project_root = Path(__file__).resolve().parents[2]
    append_training_log("Saving model bundle...")
    model_path = save_model_bundle(model, meta, project_root / "3_models", ml_config, train_df)
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
    pngs = st.session_state.pop("_preview_png_frames", [])
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
    status.success(f"Model saved to `{model_path}` ({meta.training_seconds:.1f}s)")
    st.rerun()

elif st.session_state.history is not None:
    st.subheader("Last training curves")
    if st.session_state.get("live_training_history"):
        live_chart.plotly_chart(
            training_history_live_plot(st.session_state.live_training_history),
            use_container_width=True,
        )
    else:
        st.plotly_chart(training_history_plot(st.session_state.history), use_container_width=True)
    _render_training_previews()
