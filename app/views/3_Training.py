"""Page 3: ML configuration and training."""

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
from app.state import init_session_state, sampling_ready, training_ready
from monami.config import MLConfig
from monami.ml import predict_grid, save_model_bundle, split_samples, train_model
from monami.viz import (
    exhaustive_sample_prediction_maps,
    sample_scatter,
    training_history_live_plot,
    training_history_plot,
)

init_session_state()

st.title("3. ML training")

if not sampling_ready():
    st.warning("Create samples on the **Sampling** page first.")
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
    _train_pool, _ = split_samples(samples, test_ratio, seed=st.session_state.random_seed)
    _max_neighbors = max(1, len(_train_pool) - 1)
    _default_n_neighbors = min(MLConfig().n_nearest, _max_neighbors)
    n_nearest = st.number_input(
        "Nearest neighbors (n)",
        min_value=1,
        max_value=_max_neighbors,
        value=_default_n_neighbors,
        help=(
            "Number of closest **training** samples used to build MONAMI features (dX, dY, D, V per neighbor). "
            "Test samples are not used as neighbors. Maximum is one less than the training pool size "
            f"(currently **{_max_neighbors}**). Legacy default: **10**."
        ),
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
    epochs = st.number_input(
        "Max epochs",
        min_value=10,
        max_value=5000,
        value=1000,
        step=10,
        help=(
            "Maximum training passes. Training often stops earlier via early stopping. "
            "Recommended: **250–2500** depending on dataset size."
        ),
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
    patience = st.number_input(
        "Early stopping patience",
        min_value=5,
        max_value=200,
        value=200,
        help=(
            "Stop training if validation accuracy does not improve for this many epochs. "
            "Recommended: **50**. Lower values stop sooner (e.g. 57/250 epochs is normal)."
        ),
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
    preview_interval = st.number_input(
        "Preview / log interval (epochs)",
        min_value=1,
        max_value=100,
        value=10,
        help=(
            "Update training curves, metrics, logs, and preliminary grid maps every N epochs. "
            "The final epoch is always reported, even with early stopping."
        ),
    )

st.caption(f"MONAMI input dimension: {4 * int(n_nearest)} features (dX, dY, D, V × {int(n_nearest)} neighbors)")

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
    suffix=suffix,
    n_nearest=int(n_nearest),
    hidden_activation=hidden_activation,
    out_activation=out_activation,
    preview_interval=int(preview_interval),
)

set_current_training_fingerprint(make_training_fingerprint(ml_config))

train_df, test_df = split_samples(samples, ml_config.test_ratio, seed=st.session_state.random_seed)
if int(n_nearest) >= len(train_df):
    st.error(
        f"`n` ({int(n_nearest)}) must be less than the training pool size ({len(train_df)}) "
        "when excluding the target point from its own neighbors."
    )
    st.stop()
st.session_state.train_df = train_df
st.session_state.test_df = test_df

c1, c2 = st.columns(2)
grid_shape = st.session_state.categorized_2d.shape if st.session_state.categorized_2d is not None else None
with c1:
    st.plotly_chart(
        sample_scatter(train_df, title=f"Training set ({len(train_df)} points)", grid_shape=grid_shape),
        use_container_width=False,
    )
with c2:
    st.plotly_chart(
        sample_scatter(test_df, title=f"Test set ({len(test_df)} points)", grid_shape=grid_shape),
        use_container_width=False,
    )

st.subheader("Live training")
st.caption(
    f"Progress updates every epoch. Charts, logs, and preliminary maps refresh every "
    f"{int(preview_interval)} epoch(s)."
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

if st.button("Train model", type="primary"):
    st.session_state.training_log = []
    append_training_log("Train button clicked.")
    append_training_log(
        f"Config: epochs={ml_config.epochs}, batch={ml_config.batch_size}, "
        f"layers={ml_config.nodes_per_layer}, hidden={ml_config.hidden_activation}, "
        f"output={ml_config.out_activation}"
    )
    status.info("Training started. Watch the log below and your Streamlit terminal for `[MONAMI]` messages.")
    progress.progress(0.01, text="Initializing training...")

    live_history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    truth_grid = st.session_state.categorized_2d
    samples_df = st.session_state.samples_df

    def on_epoch(epoch, total, logs, prediction=None):
        for key in live_history:
            if key in logs:
                live_history[key].append(float(logs[key]))
        progress.progress(min(epoch / total, 1.0), text=f"Epoch {epoch}/{total}")

        if prediction is None:
            return

        live_chart.plotly_chart(
            training_history_live_plot(
                live_history,
                current_epoch=epoch,
                total_epochs=total,
            ),
            use_container_width=True,
        )
        with metrics_slot.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train accuracy", f"{logs.get('accuracy', 0.0):.3f}")
            m2.metric("Val accuracy", f"{logs.get('val_accuracy', 0.0):.3f}")
            m3.metric("Train loss", f"{logs.get('loss', 0.0):.3f}")
            m4.metric("Val loss", f"{logs.get('val_loss', 0.0):.3f}")

        if truth_grid is not None and prediction is not None and samples_df is not None:
            grid_acc = float((prediction == truth_grid).mean())
            preview_slot.plotly_chart(
                exhaustive_sample_prediction_maps(
                    truth_grid,
                    samples_df,
                    prediction,
                    title=f"Exhaustive / samples / prediction — epoch {epoch}/{total}",
                ),
                use_container_width=False,
            )
            preview_caption.caption(
                f"Grid accuracy vs categorized truth at epoch {epoch}: **{grid_acc:.1%}** "
                f"({int((prediction == truth_grid).sum()):,}/{truth_grid.size:,} cells match)."
            )

    try:
        model, history, meta, _ = train_model(
            train_df,
            test_df,
            ml_config,
            grid_shape=tuple(truth_grid.shape) if truth_grid is not None else None,
            epoch_callback=on_epoch,
            log_callback=append_training_log,
        )
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
        st.session_state.live_training_history = live_history
        progress.progress(1.0, text="Training complete")
        status.success(f"Model saved to `{model_path}` ({meta.training_seconds:.1f}s)")
        live_chart.plotly_chart(training_history_plot(history), use_container_width=True)
        if truth_grid is not None:
            st.session_state.prediction_2d = predict_grid(model, truth_grid.shape, meta, train_df)
        commit_training_fingerprint(make_training_fingerprint(ml_config))
        st.rerun()
    except Exception as exc:
        append_training_log(f"ERROR: {type(exc).__name__}: {exc}")
        status.error(f"Training failed: {exc}")
        st.exception(exc)
        progress.progress(0, text="Training failed")

elif st.session_state.history is not None:
    st.subheader("Last training curves")
    if st.session_state.get("live_training_history"):
        live_chart.plotly_chart(
            training_history_live_plot(st.session_state.live_training_history),
            use_container_width=True,
        )
    else:
        st.plotly_chart(training_history_plot(st.session_state.history), use_container_width=True)
