"""Workflow and ML configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class WorkflowConfig:
    """Project paths for the MONAMI categorical workflow."""

    project_root: Path = field(default_factory=lambda: Path("."))
    exh_folder: Path = field(default_factory=lambda: Path("1_original_exhaustive"))
    exh_file: str = "porosity_3d.txt"
    sample_folder: Path = field(default_factory=lambda: Path("2_samples"))
    model_folder: Path = field(default_factory=lambda: Path("3_models"))
    report_folder: Path = field(default_factory=lambda: Path("4_reports"))

    def exhaustive_path(self) -> Path:
        return self.project_root / self.exh_folder / self.exh_file


@dataclass
class MLConfig:
    """Deep learning hyperparameters."""

    test_ratio: float = 0.2
    dropout: float = 0.2
    epochs: int = 1000
    batch_size: int = 32
    nodes_per_layer: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    hidden_activation: str = "relu"
    out_activation: str = "softmax"
    loss_function: str = "categorical_crossentropy"
    optimizer: str = "adam"
    early_stopping_patience: int = 200
    # When True, stop when train accuracy >= target_train_accuracy (overrides
    # early stopping and max-epochs as the primary stop criteria).
    stop_on_train_accuracy: bool = False
    target_train_accuracy: float = 0.95
    suffix: str = "R0"
    n_nearest: int = 50
    preview_interval: int = 10
