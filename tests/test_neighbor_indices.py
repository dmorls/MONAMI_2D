"""Tests for nearest-neighbor ranking used by Relative/Hybrid inspection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monami.features import nearest_neighbor_indices


def _toy_pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "X": [0.0, 1.0, 4.0, 5.0, 10.0],
            "Y": [0.0, 0.0, 0.0, 3.0, 0.0],
            "V": [0, 1, 0, 1, 0],
        }
    )


def _manual_rank(target_xy: np.ndarray, pool: pd.DataFrame, n: int) -> np.ndarray:
    """Brute-force nearest-first ranking with self-exclusion by distance."""
    xy = pool[["X", "Y"]].to_numpy(dtype=float)
    d = np.linalg.norm(xy - np.asarray(target_xy, dtype=float).reshape(1, 2), axis=1)
    order = np.argsort(d, kind="stable")
    keep = [i for i in order if d[i] > 1e-9]
    return np.asarray(keep[:n], dtype=int)


def test_nearest_neighbor_indices_matches_manual_ranking():
    pool = _toy_pool()
    focus = pool.loc[0, ["X", "Y"]].to_numpy(dtype=float)
    n = 3
    got = nearest_neighbor_indices(focus, pool, n, exclude_self=True)
    expected = _manual_rank(focus, pool, n)
    np.testing.assert_array_equal(got, expected)
    # Focus at (0,0): nearest others are (1,0), (4,0), (5,3) — not (10,0)
    np.testing.assert_array_equal(got, np.array([1, 2, 3]))


def test_nearest_neighbor_indices_excludes_self_at_pool_point():
    pool = _toy_pool()
    focus_idx = 2
    focus = pool.loc[focus_idx, ["X", "Y"]].to_numpy(dtype=float)
    got = nearest_neighbor_indices(focus, pool, 2, exclude_self=True)
    assert focus_idx not in set(got.tolist())
    np.testing.assert_array_equal(got, _manual_rank(focus, pool, 2))


def test_nearest_neighbor_indices_requires_enough_pool():
    pool = _toy_pool().iloc[:3].reset_index(drop=True)
    focus = pool.loc[0, ["X", "Y"]].to_numpy(dtype=float)
    with pytest.raises(ValueError, match="Need more than"):
        nearest_neighbor_indices(focus, pool, n_nearest=3, exclude_self=True)
