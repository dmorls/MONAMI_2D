"""Keras DNN training and prediction."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from monami.config import MLConfig
from monami.features import (
    build_feature_matrix,
    build_feature_matrix_from_targets,
    build_training_matrix,
    compute_xy_scale,
    feature_dim,
)
from monami.transform import numpy2d_to_easyformat


@dataclass
class ModelMeta:
    """Metadata saved alongside trained models."""

    n_classes: int
    categories: int
    grid_shape: List[int]
    nodes_per_layer: List[int]
    dropout: float
    optimizer: str
    loss_function: str
    hidden_activation: str
    test_ratio: float
    training_seconds: float
    model_filename: str
    n_nearest: int = 10
    feature_dim: int = 40
    train_sample_count: int = 0
    train_samples_file: str = ""
    neighbor_sample_count: int = 0
    neighbor_samples_file: str = ""
    out_activation: str = "softmax"
    xy_scale: List[float] = field(default_factory=list)
    class_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_class: Dict[str, int] = field(default_factory=dict)
    x_max: List[float] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        def _json_default(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        payload = json.dumps(asdict(self), indent=2, default=_json_default)
        path.write_text(payload)

    @classmethod
    def load(cls, path: Path) -> "ModelMeta":
        data = json.loads(path.read_text())
        n_nearest = data.get("n_nearest", 10)
        data.setdefault("feature_dim", feature_dim(n_nearest))
        data.setdefault("train_sample_count", 0)
        data.setdefault("train_samples_file", "")
        data.setdefault("neighbor_sample_count", 0)
        data.setdefault("neighbor_samples_file", "")
        data.setdefault("xy_scale", data.get("x_max", []))
        data.pop("include_target_xy", None)
        data.setdefault("out_activation", "softmax")
        data.setdefault("x_max", data.get("xy_scale", []))
        return cls(**data)

    def map_indices_to_classes(self, indices: np.ndarray) -> np.ndarray:
        mapping = {int(k): int(v) for k, v in self.idx_to_class.items()}
        return np.array([mapping.get(int(i), int(i)) for i in indices], dtype=float)

    def xy_scale_array(self) -> np.ndarray:
        if self.xy_scale:
            return np.array(self.xy_scale, dtype=float)
        return None


def split_samples(
    data: pd.DataFrame,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on categorical V."""
    return train_test_split(
        data,
        test_size=test_ratio,
        random_state=seed,
        stratify=data["V"],
    )


def train_samples_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + "_train.csv")


def neighbor_samples_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + "_samples.csv")


def load_train_samples(model_path: Path, meta: Optional[ModelMeta] = None) -> pd.DataFrame:
    path = train_samples_path(model_path)
    if not path.exists() and meta is not None and meta.train_samples_file:
        path = model_path.parent / meta.train_samples_file
    if not path.exists():
        raise FileNotFoundError(f"Training sample file not found for model {model_path.name}")
    return pd.read_csv(path)


def load_neighbor_pool(model_path: Path, meta: Optional[ModelMeta] = None) -> pd.DataFrame:
    """Load the sample set used for MONAMI neighbor lookup."""
    path = neighbor_samples_path(model_path)
    if not path.exists() and meta is not None and meta.neighbor_samples_file:
        path = model_path.parent / meta.neighbor_samples_file
    if not path.exists():
        return load_train_samples(model_path, meta)
    return pd.read_csv(path)


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
    model.add(Dense(ml_config.nodes_per_layer[0], input_shape=(input_dim,), activation=ml_config.hidden_activation))
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


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ml_config: MLConfig,
    neighbor_pool_df: Optional[pd.DataFrame] = None,
    grid_shape: Optional[tuple[int, int]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    epoch_callback: Optional[Callable[[int, int, dict, Optional[np.ndarray]], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
):
    """Train model and return (model, history, meta)."""
    n_nearest = ml_config.n_nearest
    if neighbor_pool_df is None:
        neighbor_pool_df = train_df

    _training_log(
        log_callback,
        f"Preparing data: {len(train_df)} train / {len(test_df)} test (validation only), "
        f"neighbor pool={len(neighbor_pool_df)} train samples, n_nearest={n_nearest}",
    )
    x_train, y_train_raw, x_test, y_test_raw, _, xy_scale = build_training_matrix(
        train_df, test_df, neighbor_pool_df, n_nearest
    )
    input_dim = x_train.shape[1]
    _training_log(
        log_callback,
        f"Feature dimension: {input_dim} (4×{n_nearest}: dX, dY, D, V per neighbor)",
    )

    classes = sorted(int(c) for c in set(y_train_raw) | set(y_test_raw))
    n_classes = len(classes)
    _training_log(log_callback, f"Found {n_classes} classes: {classes}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    _training_log(log_callback, "Importing TensorFlow/Keras (first run can take 1-2 min on Mac)...")

    tf = _import_tensorflow()

    from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
    from tensorflow.keras.utils import to_categorical

    _training_log(log_callback, f"TensorFlow {tf.__version__} loaded")

    y_train = to_categorical(np.array([class_to_idx[v] for v in y_train_raw]), num_classes=n_classes)
    y_test = to_categorical(np.array([class_to_idx[v] for v in y_test_raw]), num_classes=n_classes)

    _training_log(log_callback, f"Building model: layers={ml_config.nodes_per_layer}, dropout={ml_config.dropout}")
    model = build_model(input_dim, n_classes, ml_config)
    _training_log(log_callback, f"Model ready. Starting fit for up to {ml_config.epochs} epochs.")

    preview_every = max(1, int(ml_config.preview_interval))
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
    )
    preview_enabled = grid_shape is not None and epoch_callback is not None

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=ml_config.early_stopping_patience,
            restore_best_weights=True,
        )
    ]

    def _on_train_begin(logs=None):
        _training_log(log_callback, "Keras model.fit() started.")

    last_epoch = [0]
    last_logs: List[dict] = [{}]

    def _emit_report(epoch_num: int, logs: dict) -> None:
        _training_log(
            log_callback,
            f"Epoch {epoch_num}/{ml_config.epochs} — "
            f"acc={logs.get('accuracy', 0.0):.4f}, val_acc={logs.get('val_accuracy', 0.0):.4f}, "
            f"loss={logs.get('loss', 0.0):.4f}, val_loss={logs.get('val_loss', 0.0):.4f}",
        )
        prediction = None
        if preview_enabled:
            _training_log(log_callback, f"Generating preliminary grid prediction at epoch {epoch_num}...")
            prediction = predict_grid(model, grid_shape, interim_meta, neighbor_pool_df)
        if epoch_callback is not None:
            epoch_callback(epoch_num, ml_config.epochs, logs, prediction)

    def _on_epoch_end(epoch, logs):
        epoch_num = epoch + 1
        logs = dict(logs or {})
        if "accuracy" not in logs and "acc" in logs:
            logs["accuracy"] = logs["acc"]
        if "val_accuracy" not in logs and "val_acc" in logs:
            logs["val_accuracy"] = logs["val_acc"]

        last_epoch[0] = epoch_num
        last_logs[0] = logs

        if progress_callback is not None:
            progress_callback(epoch_num, ml_config.epochs)

        if epoch_num % preview_every == 0:
            _emit_report(epoch_num, logs)
        elif epoch_callback is not None:
            epoch_callback(epoch_num, ml_config.epochs, logs, None)

    def _on_train_end(logs=None):
        epoch_num = last_epoch[0]
        if epoch_num and epoch_num % preview_every != 0:
            _emit_report(epoch_num, last_logs[0])

    callbacks.append(
        LambdaCallback(on_train_begin=_on_train_begin, on_epoch_end=_on_epoch_end, on_train_end=_on_train_end)
    )

    start = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=ml_config.epochs,
        batch_size=ml_config.batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    elapsed = time.time() - start
    _training_log(log_callback, f"Training finished in {elapsed:.1f}s ({len(history.history.get('accuracy', []))} epochs run)")

    idx_to_class = {i: c for c, i in class_to_idx.items()}
    meta = ModelMeta(
        n_classes=n_classes,
        categories=n_classes,
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
        feature_dim=input_dim,
        train_sample_count=len(train_df),
        neighbor_sample_count=len(neighbor_pool_df),
        xy_scale=xy_scale.tolist(),
        x_max=xy_scale.tolist(),
        class_to_idx={str(k): int(v) for k, v in class_to_idx.items()},
        idx_to_class={str(k): int(v) for k, v in idx_to_class.items()},
    )
    return model, history, meta, classes


def _build_model_filename(ml_config: MLConfig, meta: ModelMeta) -> str:
    ratio = str(ml_config.test_ratio).replace(".", "")
    dropout = str(ml_config.dropout).replace(".", "")
    nodes = "_".join(str(n) for n in ml_config.nodes_per_layer)
    return (
        f"{meta.categories}_{ratio}_{ml_config.batch_size}_{ml_config.epochs}_{dropout}_"
        f"{ml_config.hidden_activation}_{ml_config.loss_function}_{ml_config.optimizer}_"
        f"{ml_config.suffix}_{nodes}"
    )


def save_model_bundle(
    model,
    meta: ModelMeta,
    folder: Path,
    ml_config: MLConfig,
    train_df: pd.DataFrame,
    neighbor_pool_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Save .h5 model, metadata JSON, train CSV, and neighbor-pool CSV."""
    if neighbor_pool_df is None:
        neighbor_pool_df = train_df
    folder.mkdir(parents=True, exist_ok=True)
    filename = _build_model_filename(ml_config, meta)
    meta.model_filename = filename
    model_path = folder / f"{filename}.h5"
    train_path = folder / f"{filename}_train.csv"
    neighbor_path = folder / f"{filename}_samples.csv"
    model.save(model_path)
    train_df.to_csv(train_path, index=False)
    neighbor_pool_df.to_csv(neighbor_path, index=False)
    meta.train_samples_file = train_path.name
    meta.train_sample_count = len(train_df)
    meta.neighbor_samples_file = neighbor_path.name
    meta.neighbor_sample_count = len(neighbor_pool_df)
    meta.save(folder / f"{filename}_meta.json")
    return model_path


def load_model_bundle(model_path: Path) -> tuple:
    """Load model, metadata, and neighbor pool from a model bundle."""
    _import_tensorflow()
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    meta = ModelMeta.load(meta_path)
    neighbor_pool_df = load_neighbor_pool(model_path, meta)
    return model, meta, neighbor_pool_df


def _build_prediction_features(
    target_xy: np.ndarray,
    meta: ModelMeta,
    neighbor_pool_df: pd.DataFrame,
) -> np.ndarray:
    xy_scale = meta.xy_scale_array()
    if xy_scale is None:
        xy_scale = compute_xy_scale(neighbor_pool_df)
    return build_feature_matrix_from_targets(
        target_xy,
        neighbor_pool_df,
        meta.n_nearest,
        exclude_self=False,
        xy_scale=xy_scale,
    )


def predict_grid(
    model,
    grid_shape: tuple[int, int],
    meta: ModelMeta,
    neighbor_pool_df: pd.DataFrame,
) -> np.ndarray:
    """Predict categorical values for every cell in a 2D grid."""
    rows, cols = grid_shape
    template = np.zeros((rows, cols))
    easy = numpy2d_to_easyformat(template)
    target_xy = easy[:, :2].astype(float)
    features = _build_prediction_features(target_xy, meta, neighbor_pool_df)
    proba = model.predict(features, verbose=0)
    class_indices = np.argmax(proba, axis=1)
    mapped = meta.map_indices_to_classes(class_indices)
    return mapped.reshape(rows, cols)


def evaluate_at_points(
    model,
    meta: ModelMeta,
    points_df: pd.DataFrame,
    neighbor_pool_df: pd.DataFrame,
) -> np.ndarray:
    """Predict classes at specific (X, Y) locations."""
    features = _build_prediction_features(
        points_df[["X", "Y"]].to_numpy(dtype=float),
        meta,
        neighbor_pool_df,
    )
    proba = model.predict(features, verbose=0)
    class_indices = np.argmax(proba, axis=1)
    return meta.map_indices_to_classes(class_indices)
