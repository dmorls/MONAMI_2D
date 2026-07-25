"""Shared ML infrastructure: metadata, splits, and model bundles."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from monami.config import MLConfig
from monami.features import feature_dim

DEFAULT_ALGORITHM_ID = "2_Relative_Position"
_RELATIVE_ALGORITHM_IDS = frozenset(
    {
        "2_Relative_Position",
        "2_Monami_NN",
        "monami_dnn",
        "4_Hybrid_Position",
    }
)


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
    algorithm_id: str = DEFAULT_ALGORITHM_ID
    algorithm_config: Dict[str, Any] = field(default_factory=dict)
    model_type: str = "keras"

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
        data.setdefault("algorithm_id", DEFAULT_ALGORITHM_ID)
        data.setdefault("model_type", "keras")
        # Normalize legacy algorithm ids stored in older bundles.
        legacy = {
            "default": "1_Absolute_Position",
            "1_Default": "1_Absolute_Position",
            "monami_dnn": "2_Relative_Position",
            "2_Monami_NN": "2_Relative_Position",
        }
        algo_id = legacy.get(data["algorithm_id"], data["algorithm_id"])
        data["algorithm_id"] = algo_id
        if "algorithm_config" not in data:
            data["algorithm_config"] = (
                {"n_nearest": n_nearest} if algo_id in _RELATIVE_ALGORITHM_IDS else {}
            )
        elif not data["algorithm_config"] and algo_id in _RELATIVE_ALGORITHM_IDS and n_nearest:
            data["algorithm_config"] = {"n_nearest": n_nearest}
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
    """Load the sample set used for neighbor lookup."""
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


def _build_model_filename(ml_config: MLConfig, meta: ModelMeta) -> str:
    if meta.model_type == "corrected_sis":
        suffix = str(ml_config.suffix or "SIS").strip() or "SIS"
        return f"ccsis_{meta.categories}cat_{meta.train_sample_count}samples_{suffix}"
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
    """Save a Keras or statistical model bundle plus metadata/sample CSVs."""
    if neighbor_pool_df is None:
        neighbor_pool_df = train_df
    folder.mkdir(parents=True, exist_ok=True)
    filename = _build_model_filename(ml_config, meta)
    candidate = filename
    version = 2
    extension = ".json" if meta.model_type == "corrected_sis" else ".h5"
    while (folder / f"{candidate}{extension}").exists():
        candidate = f"{filename}_{version}"
        version += 1
    filename = candidate
    meta.model_filename = filename
    model_path = folder / f"{filename}{extension}"
    train_path = folder / f"{filename}_train.csv"
    neighbor_path = folder / f"{filename}_samples.csv"
    if meta.model_type == "corrected_sis":
        from monami.sis import CorrectedSISModel

        if not isinstance(model, CorrectedSISModel):
            raise TypeError("corrected_sis bundles require a CorrectedSISModel")
        model_path.write_text(json.dumps(model.to_dict(), indent=2))
    else:
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
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    meta = ModelMeta.load(meta_path)
    if meta.model_type == "corrected_sis":
        from monami.sis import CorrectedSISModel

        model = CorrectedSISModel.from_dict(json.loads(model_path.read_text()))
    else:
        _import_tensorflow()
        from tensorflow.keras.models import load_model

        model = load_model(model_path)
    neighbor_pool_df = load_neighbor_pool(model_path, meta)
    return model, meta, neighbor_pool_df
