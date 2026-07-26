"""Base types and interface for pluggable prediction algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from monami.config import MLConfig
from monami.ml import ModelMeta


@dataclass
class TrainingResult:
    """Output of an algorithm training run."""

    model: Any
    history: Any
    meta: ModelMeta
    classes: List[int] = field(default_factory=list)
    # When True, training is mid-run (one-epoch Streamlit steps); keep calling train.
    incomplete: bool = False
    # Opaque resume payload for the next one-epoch step (model, data, callbacks, …).
    warm_start: Any = None
    current_epoch: int = 0
    max_epochs: int = 0


class Algorithm(ABC):
    """Full-pipeline algorithm: config UI, training, and grid prediction.

    New algorithms should use a sequential id/name prefix (``1_``, ``2_``, ``3_``, …)
    matching registration order on the Algorithm page.
    """

    id: str
    name: str
    description: str
    # Longer markdown shown below the algorithm selector on the Algorithm page.
    long_description: str = ""

    @abstractmethod
    def render_config_ui(
        self,
        st_module: Any,
        samples_df: pd.DataFrame,
        categorized_2d: np.ndarray,
        *,
        random_seed: int,
        default_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Render algorithm-specific controls and return a config dict."""

    @abstractmethod
    def fingerprint(self, config: Dict[str, Any]) -> Tuple[Any, ...]:
        """Hashable fingerprint for workflow invalidation."""

    @abstractmethod
    def supports_dnn_training_page(self) -> bool:
        """Whether DNN hyperparameters are configured on the Training page."""

    @abstractmethod
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
        ti_samples_df: Optional[pd.DataFrame] = None,
    ) -> TrainingResult:
        """Train and return model, history, and metadata.

        Optional ``warm_start`` / ``epochs_to_run`` support one-epoch Streamlit steps
        so training can yield between epochs (Stop button) without a TF background thread.

        Optional ``ti_samples_df`` supplies auxiliary training-image labels (DNN only).
        """

    @abstractmethod
    def predict_grid(
        self,
        model: Any,
        grid_shape: Tuple[int, int],
        meta: ModelMeta,
        neighbor_pool_df: pd.DataFrame,
    ) -> np.ndarray:
        """Predict categorical values for every cell in a 2D grid."""

    @abstractmethod
    def evaluate_at_points(
        self,
        model: Any,
        meta: ModelMeta,
        points_df: pd.DataFrame,
        neighbor_pool_df: pd.DataFrame,
    ) -> np.ndarray:
        """Predict classes at specific (X, Y) locations."""

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
        """Sequential categorical simulation for the full grid.

        ``correction_strength`` steers DNN draws toward sample proportions.
        Statistical algorithms may ignore it and use their own fitted setting.
        """
        raise NotImplementedError(f"{self.id} does not support sequential simulation")

    def feature_summary(self, algo_config: Dict[str, Any]) -> str:
        """Optional one-line summary for the Training page."""
        return ""

    def prediction_description(self) -> str:
        """User-facing explanation of deterministic grid prediction."""
        return "Deterministic full-grid category estimate from the fitted model."

    def simulation_description(self) -> str:
        """User-facing explanation of sequential simulation."""
        return (
            "Sequential categorical simulation over unsampled cells with hard "
            "conditioning data pinned."
        )

    def validate_config(
        self,
        algo_config: Dict[str, Any],
        train_df: pd.DataFrame,
    ) -> Optional[str]:
        """Return an error message if config is invalid, else None."""
        return None
