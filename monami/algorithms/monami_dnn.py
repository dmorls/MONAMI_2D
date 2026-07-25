"""MONAMI n-nearest-neighbor feature DNN algorithm."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from monami.algorithms.base import Algorithm, TrainingResult
from monami.config import MLConfig
from monami.features import (
    build_feature_matrix_from_targets,
    build_training_matrix,
    compute_xy_scale,
    feature_dim,
    hybrid_feature_dim,
)
from monami.ml import ModelMeta, split_samples
from monami.simulation import sequential_simulate_grid
from monami.training_stop import (
    build_convergence_callbacks,
    effective_max_epochs,
    stop_mode_summary,
)
from monami.transform import numpy2d_to_easyformat

ALGORITHM_SPEC_PATH = Path(__file__).resolve().parents[1] / "algorithm"


def _import_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf

    major_minor = ".".join(tf.__version__.split(".")[:2])
    if major_minor >= "2.20":
        raise RuntimeError(
            f"TensorFlow {tf.__version__} hangs or crashes on macOS with PyArrow "
            "(mutex lock error). Install a compatible build: pip install 'tensorflow==2.19.1'"
        )
    return tf


def _training_log(log_callback: Optional[Callable[[str], None]], message: str) -> None:
    line = f"[MONAMI] {message}"
    print(line, flush=True)
    if log_callback is not None:
        log_callback(message)


def build_model(input_dim: int, n_classes: int, ml_config: MLConfig):
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(
        Dense(
            ml_config.nodes_per_layer[0],
            input_shape=(input_dim,),
            activation=ml_config.hidden_activation,
        )
    )
    for nodes in ml_config.nodes_per_layer[1:]:
        model.add(Dense(nodes, activation=ml_config.hidden_activation))
        model.add(Dropout(ml_config.dropout))
    model.add(Dense(n_classes, activation=ml_config.out_activation))
    model.compile(
        loss=ml_config.loss_function,
        optimizer=ml_config.optimizer,
        metrics=["accuracy"],
    )
    return model


def _n_nearest(algo_config: Dict[str, Any]) -> int:
    return int(algo_config.get("n_nearest", 10))


def _build_prediction_features(
    target_xy: np.ndarray,
    meta: ModelMeta,
    neighbor_pool_df: pd.DataFrame,
    *,
    include_target_xy: bool = False,
) -> np.ndarray:
    xy_scale = meta.xy_scale_array()
    if xy_scale is None:
        xy_scale = compute_xy_scale(neighbor_pool_df)
    n_nearest = meta.algorithm_config.get("n_nearest", meta.n_nearest)
    return build_feature_matrix_from_targets(
        target_xy,
        neighbor_pool_df,
        int(n_nearest),
        exclude_self=False,
        xy_scale=xy_scale,
        include_target_xy=include_target_xy,
    )


class MonamiDNNAlgorithm(Algorithm):
    id = "2_Relative_Position"
    name = "Relative Position"
    description = "Neighbor DNN: dX, dY, D, V from the n nearest training samples."
    # When True (Hybrid Position subclass), prepend normalized absolute X, Y.
    include_target_xy = False
    long_description = """
### Relative Position — MONAMI neighbor DNN

Spatial / pattern-oriented DNN that builds features from the **n nearest training
samples** around each target location (not from absolute coordinates).

**Training features (per neighbor)**
- **dX**, **dY** — normalized offsets from target to neighbor
- **D** — normalized distance
- **V** — neighbor category
- Total input dimension = `4 × n` (configure **n** below)

**Important details**
- Absolute **X / Y** of the target are **not** model inputs
- The neighbor pool is the **training split only** (test points are for validation labels)
- During sequential simulation, each newly drawn cell is added to the conditioning
  pool so later path cells see an evolving neighborhood

**Prediction / simulation**
- Most-likely map = argmax of the softmax given current neighbors
- Sequential simulation samples from the softmax and grows the conditioning set

See the expandable specification below for the full feature definition.
""".strip()

    def render_config_ui(
        self,
        st_module: Any,
        samples_df: pd.DataFrame,
        categorized_2d: np.ndarray,
        *,
        random_seed: int,
        default_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        default_config = default_config or {}
        train_pool, _ = split_samples(samples_df, 0.2, seed=random_seed)
        max_neighbors = max(1, len(train_pool) - 1)
        default_n = min(int(default_config.get("n_nearest", MLConfig().n_nearest)), max_neighbors)

        n_nearest = st_module.number_input(
            "Nearest neighbors (n)",
            min_value=1,
            max_value=max_neighbors,
            value=default_n,
            help=(
                "Number of closest **training** samples used to build MONAMI features "
                "(dX, dY, D, V per neighbor). Maximum is one less than the training pool size "
                f"(currently **{max_neighbors}**)."
            ),
        )

        if ALGORITHM_SPEC_PATH.exists():
            with st_module.expander("Algorithm specification", expanded=False):
                st_module.markdown(ALGORITHM_SPEC_PATH.read_text())

        return {"n_nearest": int(n_nearest)}

    def fingerprint(self, config: Dict[str, Any]) -> Tuple[Any, ...]:
        return (self.id, int(config.get("n_nearest", 10)))

    def supports_dnn_training_page(self) -> bool:
        return True

    def feature_summary(self, algo_config: Dict[str, Any]) -> str:
        n = _n_nearest(algo_config)
        if self.include_target_xy:
            return (
                f"Hybrid input dimension: {hybrid_feature_dim(n)} features "
                f"(X, Y + dX, dY, D, V × {n} neighbors)"
            )
        return f"MONAMI input dimension: {feature_dim(n)} features (dX, dY, D, V × {n} neighbors)"

    def prediction_description(self) -> str:
        if self.include_target_xy:
            return (
                "Argmax of the DNN softmax at every cell. Hybrid features combine "
                "normalized absolute (X, Y) with relative-position neighbors from the "
                "saved training-sample pool."
            )
        return (
            "Argmax of the DNN softmax at every cell. Relative-position neighbor "
            "features use the saved training-sample pool."
        )

    def simulation_description(self) -> str:
        if self.include_target_xy:
            return (
                "Random sequential path with training samples pinned. Hybrid features "
                "use normalized absolute (X, Y) plus relative neighbors from the growing "
                "hard-plus-simulated conditioning pool. A sample-proportion servo "
                "(Results page strength) gently steers each realization toward the "
                "training-sample histogram."
            )
        return (
            "Random sequential path with training samples pinned. Relative-position "
            "features use the growing hard-plus-simulated conditioning pool. A "
            "sample-proportion servo (Results page strength) gently steers each "
            "realization toward the training-sample histogram."
        )

    def validate_config(
        self,
        algo_config: Dict[str, Any],
        train_df: pd.DataFrame,
    ) -> Optional[str]:
        n = _n_nearest(algo_config)
        if n >= len(train_df):
            return (
                f"`n` ({n}) must be less than the training pool size ({len(train_df)}) "
                "when excluding the target point from its own neighbors."
            )
        return None

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        neighbor_pool_df: pd.DataFrame,
        grid_shape: Optional[Tuple[int, int]],
        algo_config: Dict[str, Any],
        ml_config: MLConfig,
        *,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        epoch_callback: Optional[Callable[[int, int, dict, Optional[np.ndarray]], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        warm_start: Any = None,
        epochs_to_run: Optional[int] = None,
    ) -> TrainingResult:
        from tensorflow.keras.callbacks import LambdaCallback
        from tensorflow.keras.utils import to_categorical

        from monami.training_stop import is_manual_stop_requested

        n_nearest = _n_nearest(algo_config)
        max_epochs = effective_max_epochs(ml_config)
        chunk = int(epochs_to_run) if epochs_to_run is not None else max_epochs

        if warm_start is None:
            _training_log(
                log_callback,
                f"Preparing data: {len(train_df)} train / {len(test_df)} test (validation only), "
                f"neighbor pool={len(neighbor_pool_df)} train samples, n_nearest={n_nearest}",
            )
            x_train, y_train_raw, x_test, y_test_raw, _, xy_scale = build_training_matrix(
                train_df,
                test_df,
                neighbor_pool_df,
                n_nearest,
                include_target_xy=self.include_target_xy,
            )
            input_dim = x_train.shape[1]
            if self.include_target_xy:
                feat_msg = (
                    f"Feature dimension: {input_dim} "
                    f"(X, Y + 4×{n_nearest}: dX, dY, D, V per neighbor)"
                )
            else:
                feat_msg = (
                    f"Feature dimension: {input_dim} "
                    f"(4×{n_nearest}: dX, dY, D, V per neighbor)"
                )
            _training_log(log_callback, feat_msg)

            classes = sorted(int(c) for c in set(y_train_raw) | set(y_test_raw))
            n_classes = len(classes)
            _training_log(log_callback, f"Found {n_classes} classes: {classes}")
            class_to_idx = {c: i for i, c in enumerate(classes)}

            _training_log(log_callback, "Importing TensorFlow/Keras (first run can take 1-2 min on Mac)...")
            tf = _import_tensorflow()
            _training_log(log_callback, f"TensorFlow {tf.__version__} loaded")
            _training_log(log_callback, f"Stop criteria: {stop_mode_summary(ml_config)}")

            y_train = to_categorical(
                np.array([class_to_idx[v] for v in y_train_raw]), num_classes=n_classes
            )
            y_test = to_categorical(
                np.array([class_to_idx[v] for v in y_test_raw]), num_classes=n_classes
            )

            _training_log(
                log_callback,
                f"Building model: layers={ml_config.nodes_per_layer}, dropout={ml_config.dropout}",
            )
            model = build_model(input_dim, n_classes, ml_config)
            _training_log(log_callback, f"Model ready. Starting fit (cap {max_epochs} epochs).")

            idx_to_class = {i: c for c, i in class_to_idx.items()}
            interim_meta = ModelMeta(
                n_classes=n_classes,
                categories=n_classes,
                grid_shape=list(grid_shape) if grid_shape else [],
                nodes_per_layer=list(ml_config.nodes_per_layer),
                dropout=ml_config.dropout,
                optimizer=ml_config.optimizer,
                loss_function=ml_config.loss_function,
                hidden_activation=ml_config.hidden_activation,
                out_activation=ml_config.out_activation,
                test_ratio=ml_config.test_ratio,
                training_seconds=0.0,
                model_filename="",
                n_nearest=n_nearest,
                feature_dim=input_dim,
                train_sample_count=len(train_df),
                neighbor_sample_count=len(neighbor_pool_df),
                xy_scale=xy_scale.tolist(),
                x_max=xy_scale.tolist(),
                class_to_idx={str(k): int(v) for k, v in class_to_idx.items()},
                idx_to_class={str(k): int(v) for k, v in idx_to_class.items()},
                algorithm_id=self.id,
                algorithm_config=dict(algo_config),
            )
            callbacks = build_convergence_callbacks(ml_config, log_callback=log_callback)
            history_acc: Dict[str, List[float]] = {
                "accuracy": [],
                "val_accuracy": [],
                "loss": [],
                "val_loss": [],
            }
            start_time = time.time()
            initial_epoch = 0
        else:
            _import_tensorflow()
            model = warm_start["model"]
            x_train = warm_start["x_train"]
            y_train = warm_start["y_train"]
            x_test = warm_start["x_test"]
            y_test = warm_start["y_test"]
            classes = warm_start["classes"]
            class_to_idx = warm_start["class_to_idx"]
            interim_meta = warm_start["interim_meta"]
            callbacks = warm_start["callbacks"]
            history_acc = warm_start["history_acc"]
            start_time = warm_start["start_time"]
            initial_epoch = int(warm_start["epoch"])
            n_nearest = warm_start["n_nearest"]
            input_dim = warm_start["input_dim"]
            xy_scale = np.asarray(warm_start["xy_scale"], dtype=float)
            neighbor_pool_df = warm_start["neighbor_pool_df"]
            grid_shape = warm_start.get("grid_shape") or grid_shape

        preview_every = max(1, int(ml_config.preview_interval))
        preview_enabled = grid_shape is not None and epoch_callback is not None
        last_epoch = [initial_epoch]
        last_logs: List[dict] = [{}]

        def _emit_report(epoch_num: int, logs: dict) -> None:
            _training_log(
                log_callback,
                f"Epoch {epoch_num}/{max_epochs} — "
                f"acc={logs.get('accuracy', 0.0):.4f}, val_acc={logs.get('val_accuracy', 0.0):.4f}, "
                f"loss={logs.get('loss', 0.0):.4f}, val_loss={logs.get('val_loss', 0.0):.4f}",
            )
            prediction = None
            if preview_enabled:
                _training_log(
                    log_callback,
                    f"Generating preliminary grid prediction at epoch {epoch_num}...",
                )
                prediction = self.predict_grid(model, grid_shape, interim_meta, neighbor_pool_df)
            if epoch_callback is not None:
                epoch_callback(epoch_num, max_epochs, logs, prediction)

        def _on_epoch_end(epoch, logs):
            epoch_num = epoch + 1
            logs = dict(logs or {})
            if "accuracy" not in logs and "acc" in logs:
                logs["accuracy"] = logs["acc"]
            if "val_accuracy" not in logs and "val_acc" in logs:
                logs["val_accuracy"] = logs["val_acc"]
            last_epoch[0] = epoch_num
            last_logs[0] = logs
            for key in history_acc:
                if key in logs:
                    history_acc[key].append(float(logs[key]))
            if progress_callback is not None:
                progress_callback(epoch_num, max_epochs)
            if epoch_num % preview_every == 0:
                _emit_report(epoch_num, logs)
            elif epoch_callback is not None:
                epoch_callback(epoch_num, max_epochs, logs, None)

        step_callbacks = list(callbacks) + [
            LambdaCallback(on_epoch_end=_on_epoch_end)
        ]

        target_epoch = min(initial_epoch + chunk, max_epochs)
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=target_epoch,
            initial_epoch=initial_epoch,
            batch_size=ml_config.batch_size,
            verbose=0,
            callbacks=step_callbacks,
        )

        current_epoch = int(last_epoch[0] or target_epoch)
        stopped = bool(getattr(model, "stop_training", False)) or is_manual_stop_requested()
        incomplete = (current_epoch < max_epochs) and not stopped and epochs_to_run is not None

        class _Hist:
            def __init__(self, h):
                self.history = h

        if incomplete:
            warm = {
                "model": model,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "interim_meta": interim_meta,
                "callbacks": callbacks,
                "history_acc": history_acc,
                "start_time": start_time,
                "epoch": current_epoch,
                "n_nearest": n_nearest,
                "input_dim": input_dim,
                "xy_scale": xy_scale.tolist() if hasattr(xy_scale, "tolist") else list(xy_scale),
                "neighbor_pool_df": neighbor_pool_df,
                "grid_shape": grid_shape,
                "train_df": train_df,
            }
            return TrainingResult(
                model=model,
                history=_Hist(history_acc),
                meta=interim_meta,
                classes=classes,
                incomplete=True,
                warm_start=warm,
                current_epoch=current_epoch,
                max_epochs=max_epochs,
            )

        if current_epoch and current_epoch % preview_every != 0 and last_logs[0]:
            _emit_report(current_epoch, last_logs[0])

        elapsed = time.time() - start_time
        _training_log(
            log_callback,
            f"Training finished in {elapsed:.1f}s ({current_epoch} epochs run)",
        )
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        meta = ModelMeta(
            n_classes=len(classes),
            categories=len(classes),
            grid_shape=[],
            nodes_per_layer=list(ml_config.nodes_per_layer),
            dropout=ml_config.dropout,
            optimizer=ml_config.optimizer,
            loss_function=ml_config.loss_function,
            hidden_activation=ml_config.hidden_activation,
            out_activation=ml_config.out_activation,
            test_ratio=ml_config.test_ratio,
            training_seconds=elapsed,
            model_filename="",
            n_nearest=n_nearest,
            feature_dim=int(input_dim),
            train_sample_count=len(train_df),
            neighbor_sample_count=len(neighbor_pool_df),
            xy_scale=list(interim_meta.xy_scale),
            x_max=list(interim_meta.xy_scale),
            class_to_idx={str(k): int(v) for k, v in class_to_idx.items()},
            idx_to_class={str(k): int(v) for k, v in idx_to_class.items()},
            algorithm_id=self.id,
            algorithm_config=dict(algo_config),
        )
        return TrainingResult(
            model=model,
            history=_Hist(history_acc),
            meta=meta,
            classes=classes,
            incomplete=False,
            current_epoch=current_epoch,
            max_epochs=max_epochs,
        )

    def predict_grid(
        self,
        model: Any,
        grid_shape: Tuple[int, int],
        meta: ModelMeta,
        neighbor_pool_df: pd.DataFrame,
    ) -> np.ndarray:
        rows, cols = grid_shape
        template = np.zeros((rows, cols))
        easy = numpy2d_to_easyformat(template)
        target_xy = easy[:, :2].astype(float)
        features = _build_prediction_features(
            target_xy,
            meta,
            neighbor_pool_df,
            include_target_xy=self.include_target_xy,
        )
        proba = model.predict(features, verbose=0)
        class_indices = np.argmax(proba, axis=1)
        mapped = meta.map_indices_to_classes(class_indices)
        return mapped.reshape(rows, cols)

    def evaluate_at_points(
        self,
        model: Any,
        meta: ModelMeta,
        points_df: pd.DataFrame,
        neighbor_pool_df: pd.DataFrame,
    ) -> np.ndarray:
        features = _build_prediction_features(
            points_df[["X", "Y"]].to_numpy(dtype=float),
            meta,
            neighbor_pool_df,
            include_target_xy=self.include_target_xy,
        )
        proba = model.predict(features, verbose=0)
        class_indices = np.argmax(proba, axis=1)
        return meta.map_indices_to_classes(class_indices)

    def simulate_grid(
        self,
        model: Any,
        meta: ModelMeta,
        hard_df: pd.DataFrame,
        grid_shape: Tuple[int, int],
        *,
        seed: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        correction_strength: float = 0.5,
    ) -> np.ndarray:
        return sequential_simulate_grid(
            model,
            meta,
            hard_df,
            grid_shape,
            seed=seed,
            progress_callback=progress_callback,
            include_target_xy=self.include_target_xy,
            correction_strength=float(correction_strength),
        )
