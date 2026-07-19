"""Default coordinate DNN: normalized (X, Y) → category V."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from monami.algorithms.base import Algorithm, TrainingResult
from monami.algorithms.monami_dnn import _import_tensorflow, build_model
from monami.config import MLConfig
from monami.features import compute_xy_scale
from monami.ml import ModelMeta
from monami.simulation import sequential_simulate_coord_grid
from monami.training_stop import (
    build_convergence_callbacks,
    effective_max_epochs,
    stop_mode_summary,
)
from monami.transform import numpy2d_to_easyformat


def _training_log(log_callback: Optional[Callable[[str], None]], message: str) -> None:
    line = f"[Default] {message}"
    print(line, flush=True)
    if log_callback is not None:
        log_callback(message)


def _xy_features(xy: np.ndarray, xy_scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(xy_scale, dtype=float).ravel()
    scale = np.where(scale == 0, 1.0, scale)
    return np.asarray(xy, dtype=float) / scale


def _xy_scale_from_meta(meta: ModelMeta, fallback_df: pd.DataFrame) -> np.ndarray:
    scale = meta.xy_scale_array()
    if scale is None:
        return compute_xy_scale(fallback_df)
    return scale


class DefaultDNNAlgorithm(Algorithm):
    id = "1_Absolute_Position"
    name = "Absolute Position"
    description = "Baseline DNN: normalized absolute X, Y → category V."
    long_description = """
### Absolute Position — coordinate DNN

A simple baseline that predicts the categorical value **V** from the cell’s own
location only.

**Training features**
- Normalized absolute **X** and **Y** (input dimension = 2)
- Label = category **V**

**What it does not use**
- No nearest-neighbor search
- No relative offsets, distances, or neighbor categories

**Prediction / simulation**
- At every grid cell, the network sees only that cell’s normalized `(X, Y)`
- Most-likely map = argmax of the softmax
- Sequential simulation samples a category from the softmax at each path cell;
  hard (training) data still pin known cells, but previously simulated values
  are **not** used as model inputs

Use this algorithm as a transparent reference against relative-position / neighbor methods.
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
        st_module.caption("No algorithm-specific parameters. Configure the DNN on the **Training** page.")
        return {}

    def fingerprint(self, config: Dict[str, Any]) -> Tuple[Any, ...]:
        return (self.id,)

    def supports_dnn_training_page(self) -> bool:
        return True

    def feature_summary(self, algo_config: Dict[str, Any]) -> str:
        return "Absolute Position input dimension: 2 features (X, Y)"

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

        max_epochs = effective_max_epochs(ml_config)
        chunk = int(epochs_to_run) if epochs_to_run is not None else max_epochs
        input_dim = 2

        if warm_start is None:
            _training_log(
                log_callback,
                f"Preparing data: {len(train_df)} train / {len(test_df)} test, features = X, Y",
            )
            xy_scale = compute_xy_scale(train_df)
            x_train = _xy_features(train_df[["X", "Y"]].to_numpy(dtype=float), xy_scale)
            x_test = _xy_features(test_df[["X", "Y"]].to_numpy(dtype=float), xy_scale)
            y_train_raw = train_df["V"].to_numpy()
            y_test_raw = test_df["V"].to_numpy()
            _training_log(log_callback, f"Feature dimension: {input_dim} (X, Y)")

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
                n_nearest=0,
                feature_dim=input_dim,
                train_sample_count=len(train_df),
                neighbor_sample_count=len(neighbor_pool_df),
                xy_scale=xy_scale.tolist(),
                x_max=xy_scale.tolist(),
                class_to_idx={str(k): int(v) for k, v in class_to_idx.items()},
                idx_to_class={str(i): int(c) for i, c in enumerate(classes)},
                algorithm_id=self.id,
                algorithm_config=dict(algo_config or {}),
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

        step_callbacks = list(callbacks) + [LambdaCallback(on_epoch_end=_on_epoch_end)]
        target_epoch = min(initial_epoch + chunk, max_epochs)
        model.fit(
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
                "xy_scale": xy_scale.tolist() if hasattr(xy_scale, "tolist") else list(xy_scale),
                "neighbor_pool_df": neighbor_pool_df,
                "grid_shape": grid_shape,
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
        _training_log(log_callback, f"Training finished in {elapsed:.1f}s ({current_epoch} epochs run)")
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
            n_nearest=0,
            feature_dim=input_dim,
            train_sample_count=len(train_df),
            neighbor_sample_count=len(neighbor_pool_df),
            xy_scale=list(interim_meta.xy_scale),
            x_max=list(interim_meta.xy_scale),
            class_to_idx={str(k): int(v) for k, v in class_to_idx.items()},
            idx_to_class={str(k): int(v) for k, v in idx_to_class.items()},
            algorithm_id=self.id,
            algorithm_config=dict(algo_config or {}),
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
        xy_scale = _xy_scale_from_meta(meta, neighbor_pool_df)
        features = _xy_features(target_xy, xy_scale)
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
        xy_scale = _xy_scale_from_meta(meta, neighbor_pool_df if neighbor_pool_df is not None else points_df)
        features = _xy_features(points_df[["X", "Y"]].to_numpy(dtype=float), xy_scale)
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
    ) -> np.ndarray:
        return sequential_simulate_coord_grid(
            model,
            meta,
            hard_df,
            grid_shape,
            seed=seed,
            progress_callback=progress_callback,
        )
