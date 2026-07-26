"""Tests for optional training-image (aux Z-slice) labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monami.features import (
    absolute_xy_features,
    build_feature_matrix,
    build_training_matrix,
    compute_xy_scale,
)
from monami.transform import categorize_slice, compute_thresholds


def _grid(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((12, 10))


def test_ti_uses_shared_category_edges():
    target = _grid(1)
    ti = _grid(2) * 0.8 + 0.1
    edges = compute_thresholds(target.ravel(), 3, method="quantile")
    _, target_cat, applied = categorize_slice(target, 3, bin_edges=edges)
    _, ti_cat, ti_applied = categorize_slice(ti, 3, bin_edges=edges)
    np.testing.assert_allclose(applied, edges)
    np.testing.assert_allclose(ti_applied, edges)
    assert target_cat.shape == target.shape
    assert ti_cat.shape == ti.shape
    assert set(np.unique(ti_cat)).issubset(set(range(3)))


def test_training_matrix_stacks_ti_with_slice_local_neighbors():
    # Target pool clustered near origin; TI pool far away.
    train = pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Y": [1.0, 1.0, 2.0, 2.0, 3.0],
            "V": [0, 1, 0, 1, 0],
        }
    )
    test = pd.DataFrame({"X": [2.5], "Y": [2.5], "V": [1]})
    ti = pd.DataFrame(
        {
            "X": [20.0, 21.0, 22.0, 23.0, 24.0],
            "Y": [20.0, 21.0, 20.0, 21.0, 22.0],
            "V": [2, 2, 1, 2, 1],
        }
    )
    n = 2
    x_base, y_base, _, _, _, scale = build_training_matrix(
        train, test, train, n, include_target_xy=False
    )
    x_all, y_all, x_test, y_test, names, scale2 = build_training_matrix(
        train, test, train, n, include_target_xy=False, ti_samples_df=ti
    )
    np.testing.assert_allclose(scale, scale2)
    assert x_all.shape[0] == x_base.shape[0] + len(ti)
    assert y_all.shape[0] == y_base.shape[0] + len(ti)
    assert x_test.shape[0] == 1
    assert list(y_all[len(train) :]) == list(ti["V"].astype(int))

    # TI feature rows must match building TI against the TI pool (not target pool).
    x_ti_expected = build_feature_matrix(
        ti, ti, n, exclude_self=True, xy_scale=scale, include_target_xy=False
    )
    np.testing.assert_allclose(x_all[len(train) :], x_ti_expected)

    # Building TI targets against the target pool would differ (far neighbors).
    x_ti_wrong_pool = build_feature_matrix(
        ti, train, n, exclude_self=False, xy_scale=scale, include_target_xy=False
    )
    assert not np.allclose(x_all[len(train) :], x_ti_wrong_pool)
    assert names[0] == "dX_0"


def test_absolute_style_xy_stack_uses_target_scale():
    train = pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [1.0, 2.0, 3.0], "V": [0, 1, 0]})
    ti = pd.DataFrame({"X": [4.0, 5.0], "Y": [4.0, 5.0], "V": [1, 2]})
    scale = compute_xy_scale(train)
    x_train = absolute_xy_features(train[["X", "Y"]].to_numpy(dtype=float), scale)
    x_ti = absolute_xy_features(ti[["X", "Y"]].to_numpy(dtype=float), scale)
    stacked = np.vstack([x_train, x_ti])
    assert stacked.shape == (5, 2)
    np.testing.assert_allclose(stacked[:3], x_train)
    np.testing.assert_allclose(stacked[3:], x_ti)
