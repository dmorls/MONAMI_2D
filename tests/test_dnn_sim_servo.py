"""Tests for DNN sequential-simulation sample-proportion servo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monami.ml import ModelMeta
from monami.simulation import (
    sequential_simulate_coord_grid,
    sequential_simulate_grid,
)


class _FlatProbaModel:
    """Stub model that always returns a uniform softmax over n_classes."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def __call__(self, features, training=False):
        n = int(np.asarray(features).shape[0])
        return np.full((n, self.n_classes), 1.0 / self.n_classes, dtype=float)

    def predict(self, features, verbose=0):
        return self(features)


def _meta(n_classes: int = 3, n_nearest: int = 2) -> ModelMeta:
    class_to_idx = {str(i): i for i in range(n_classes)}
    idx_to_class = {str(i): i for i in range(n_classes)}
    return ModelMeta(
        n_classes=n_classes,
        categories=n_classes,
        grid_shape=[8, 8],
        nodes_per_layer=[8],
        dropout=0.0,
        optimizer="adam",
        loss_function="categorical_crossentropy",
        hidden_activation="relu",
        test_ratio=0.2,
        training_seconds=0.0,
        model_filename="",
        n_nearest=n_nearest,
        feature_dim=4 * n_nearest,
        train_sample_count=12,
        neighbor_sample_count=12,
        xy_scale=[8.0, 8.0],
        x_max=[8.0, 8.0],
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        algorithm_id="2_Relative_Position",
        algorithm_config={"n_nearest": n_nearest},
    )


def _skewed_hard_df(n_classes: int = 3) -> pd.DataFrame:
    """Hard data dominated by category 0 (~75%), with a few of 1 and 2."""
    rows = []
    # Place unique hard cells on a small grid
    for i, v in enumerate([0] * 9 + [1] * 2 + [2] * 1):
        rows.append({"X": (i % 4) + 1, "Y": (i // 4) + 1, "V": v})
    return pd.DataFrame(rows)


def _field_proportions(grid: np.ndarray, n_classes: int) -> np.ndarray:
    vals = np.asarray(grid).ravel()
    counts = np.array([(vals == c).sum() for c in range(n_classes)], dtype=float)
    return counts / counts.sum()


def test_coord_sim_servo_strength_one_tracks_hard_proportions():
    meta = _meta()
    hard = _skewed_hard_df()
    model = _FlatProbaModel(meta.n_classes)
    target = np.array(
        [(hard["V"] == c).mean() for c in range(meta.n_classes)], dtype=float
    )

    grid_on = sequential_simulate_coord_grid(
        model,
        meta,
        hard,
        (8, 8),
        seed=7,
        correction_strength=1.0,
    )
    grid_off = sequential_simulate_coord_grid(
        model,
        meta,
        hard,
        (8, 8),
        seed=7,
        correction_strength=0.0,
    )

    p_on = _field_proportions(grid_on, meta.n_classes)
    p_off = _field_proportions(grid_off, meta.n_classes)
    # Full-strength servo should match hard proportions more closely than pure softmax.
    assert np.sum(np.abs(p_on - target)) < np.sum(np.abs(p_off - target))
    # Category 0 is the majority in hard data; servo should favor it in the field.
    assert p_on[0] > p_off[0]


def test_neighbor_sim_servo_uses_growing_pool_and_strength():
    meta = _meta(n_nearest=2)
    hard = _skewed_hard_df()
    model = _FlatProbaModel(meta.n_classes)
    target = np.array(
        [(hard["V"] == c).mean() for c in range(meta.n_classes)], dtype=float
    )

    grid_on = sequential_simulate_grid(
        model,
        meta,
        hard,
        (8, 8),
        seed=11,
        include_target_xy=False,
        correction_strength=1.0,
    )
    grid_off = sequential_simulate_grid(
        model,
        meta,
        hard,
        (8, 8),
        seed=11,
        include_target_xy=False,
        correction_strength=0.0,
    )

    p_on = _field_proportions(grid_on, meta.n_classes)
    p_off = _field_proportions(grid_off, meta.n_classes)
    assert np.sum(np.abs(p_on - target)) < np.sum(np.abs(p_off - target))

    # Hard data must remain pinned.
    for _, row in hard.iterrows():
        assert grid_on[int(row["Y"]) - 1, int(row["X"]) - 1] == float(row["V"])
        assert grid_off[int(row["Y"]) - 1, int(row["X"]) - 1] == float(row["V"])


def test_hybrid_flag_does_not_break_servo():
    meta = _meta(n_nearest=2)
    meta.feature_dim = 2 + 4 * 2
    meta.algorithm_id = "4_Hybrid_Position"
    hard = _skewed_hard_df()
    model = _FlatProbaModel(meta.n_classes)
    grid = sequential_simulate_grid(
        model,
        meta,
        hard,
        (8, 8),
        seed=3,
        include_target_xy=True,
        correction_strength=0.5,
    )
    assert grid.shape == (8, 8)
    assert not np.isnan(grid).any()
