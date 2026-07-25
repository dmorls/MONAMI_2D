"""Tests for Hybrid Position feature composition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monami.algorithms.hybrid_dnn import HybridPositionDNNAlgorithm
from monami.algorithms.registry import get_algorithm, list_algorithms
from monami.features import (
    absolute_xy_features,
    build_feature_matrix_from_targets,
    build_training_matrix,
    compute_xy_scale,
    feature_dim,
    hybrid_feature_dim,
    hybrid_feature_names,
)


def _toy_pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "X": [1.0, 2.0, 4.0, 5.0, 8.0],
            "Y": [1.0, 3.0, 2.0, 6.0, 4.0],
            "V": [0, 1, 0, 1, 0],
        }
    )


def test_hybrid_feature_dim_and_names():
    n = 3
    assert hybrid_feature_dim(n) == 2 + feature_dim(n)
    names = hybrid_feature_names(n)
    assert names[:2] == ["X", "Y"]
    assert len(names) == hybrid_feature_dim(n)
    assert names[2] == "dX_0"


def test_hybrid_prepends_absolute_xy_and_keeps_relative_block():
    pool = _toy_pool()
    xy_scale = compute_xy_scale(pool)
    targets = np.array([[3.0, 3.0], [6.0, 5.0]], dtype=float)
    n = 2

    relative = build_feature_matrix_from_targets(
        targets, pool, n, exclude_self=False, xy_scale=xy_scale, include_target_xy=False
    )
    hybrid = build_feature_matrix_from_targets(
        targets, pool, n, exclude_self=False, xy_scale=xy_scale, include_target_xy=True
    )

    assert hybrid.shape == (2, hybrid_feature_dim(n))
    expected_abs = absolute_xy_features(targets, xy_scale)
    np.testing.assert_allclose(hybrid[:, :2], expected_abs)
    np.testing.assert_allclose(hybrid[:, 2:], relative)


def test_build_training_matrix_hybrid_flag():
    pool = _toy_pool()
    train = pool.iloc[:4].copy()
    test = pool.iloc[4:].copy()
    n = 2
    x_train, _, x_test, _, names, xy_scale = build_training_matrix(
        train, test, train, n, include_target_xy=True
    )
    assert x_train.shape[1] == hybrid_feature_dim(n)
    assert x_test.shape[1] == hybrid_feature_dim(n)
    assert names == hybrid_feature_names(n)
    np.testing.assert_allclose(x_train[:, :2], absolute_xy_features(train[["X", "Y"]], xy_scale))


def test_hybrid_algorithm_registered():
    ids = [a.id for a in list_algorithms()]
    assert "4_Hybrid_Position" in ids
    algo = get_algorithm("4_Hybrid_Position")
    assert isinstance(algo, HybridPositionDNNAlgorithm)
    assert algo.include_target_xy is True
    assert algo.supports_dnn_training_page() is True
    summary = algo.feature_summary({"n_nearest": 5})
    assert "12" in summary or str(hybrid_feature_dim(5)) in summary
