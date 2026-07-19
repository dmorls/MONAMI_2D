"""Classic sequential simulation helpers for trained DNNs."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from monami.features import build_feature_matrix_from_targets, compute_xy_scale
from monami.ml import ModelMeta


def _n_nearest(meta: ModelMeta) -> int:
    if meta.algorithm_config and "n_nearest" in meta.algorithm_config:
        return int(meta.algorithm_config["n_nearest"])
    return int(meta.n_nearest)


def _xy_scale(meta: ModelMeta, hard_df: pd.DataFrame) -> np.ndarray:
    scale = meta.xy_scale_array()
    if scale is None:
        return compute_xy_scale(hard_df)
    return scale


def _sample_class_index(proba_row: np.ndarray, rng: np.random.Generator) -> int:
    p = np.asarray(proba_row, dtype=float).ravel()
    p = np.maximum(p, 0.0)
    total = float(p.sum())
    if total <= 0.0 or not np.isfinite(total):
        p = np.ones(len(p), dtype=float) / max(len(p), 1)
    else:
        p = p / total
    return int(rng.choice(len(p), p=p))


def _predict_proba_row(model, features: np.ndarray) -> np.ndarray:
    """Prefer eager call for single-row inference speed when available."""
    try:
        proba = model(features, training=False)
        if hasattr(proba, "numpy"):
            proba = proba.numpy()
        else:
            proba = np.asarray(proba)
    except Exception:
        proba = model.predict(features, verbose=0)
    return np.asarray(proba)


def _pin_hard_data(
    hard_df: pd.DataFrame,
    grid_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if hard_df is None or len(hard_df) == 0:
        raise ValueError("Hard data DataFrame is empty; need training samples for conditioning.")
    missing = {"X", "Y", "V"} - set(hard_df.columns)
    if missing:
        raise ValueError(f"Hard data missing columns: {sorted(missing)}")

    n_rows, n_cols = grid_shape
    grid = np.full((n_rows, n_cols), np.nan, dtype=float)
    hard_mask = np.zeros((n_rows, n_cols), dtype=bool)

    hard = hard_df[["X", "Y", "V"]].copy()
    hard["X"] = hard["X"].astype(int)
    hard["Y"] = hard["Y"].astype(int)
    for _, row in hard.iterrows():
        x = int(row["X"])
        y = int(row["Y"])
        if 1 <= x <= n_cols and 1 <= y <= n_rows:
            grid[y - 1, x - 1] = float(row["V"])
            hard_mask[y - 1, x - 1] = True
    return grid, hard_mask, hard


def sequential_simulate_grid(
    model,
    meta: ModelMeta,
    hard_df: pd.DataFrame,
    grid_shape: Tuple[int, int],
    *,
    seed: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """
    Sequential categorical simulation with a random path (MONAMI neighbor features).

    Hard data (typically training samples) are fixed. Unsampled cells are visited
    in random order; each draw is sampled from the DNN softmax using the growing
    conditioning pool (hard data + previously simulated values).
    """
    n_nearest = _n_nearest(meta)
    if len(hard_df) < n_nearest:
        raise ValueError(
            f"Need at least {n_nearest} hard samples for neighbor features; got {len(hard_df)}"
        )

    xy_scale = _xy_scale(meta, hard_df)
    rng = np.random.default_rng(int(seed))
    grid, hard_mask, hard = _pin_hard_data(hard_df, grid_shape)

    unknown_y, unknown_x = np.where(~hard_mask)
    if len(unknown_y) == 0:
        return grid

    order = rng.permutation(len(unknown_y))
    path_y = unknown_y[order]
    path_x = unknown_x[order]

    cond_x = hard["X"].to_numpy(dtype=float).tolist()
    cond_y = hard["Y"].to_numpy(dtype=float).tolist()
    cond_v = hard["V"].to_numpy(dtype=float).tolist()

    n_path = len(path_y)
    report_every = max(1, n_path // 100)

    for i in range(n_path):
        gx = int(path_x[i]) + 1
        gy = int(path_y[i]) + 1

        neighbor_df = pd.DataFrame({"X": cond_x, "Y": cond_y, "V": cond_v})
        features = build_feature_matrix_from_targets(
            np.array([[gx, gy]], dtype=float),
            neighbor_df,
            n_nearest,
            exclude_self=False,
            xy_scale=xy_scale,
        )

        proba = _predict_proba_row(model, features)
        class_idx = _sample_class_index(proba[0], rng)
        drawn = float(meta.map_indices_to_classes(np.array([class_idx]))[0])

        grid[path_y[i], path_x[i]] = drawn
        cond_x.append(float(gx))
        cond_y.append(float(gy))
        cond_v.append(drawn)

        if progress_callback is not None and (i + 1 == n_path or (i + 1) % report_every == 0):
            progress_callback(i + 1, n_path)

    return grid


def sequential_simulate_coord_grid(
    model,
    meta: ModelMeta,
    hard_df: pd.DataFrame,
    grid_shape: Tuple[int, int],
    *,
    seed: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """
    Sequential categorical simulation with coordinate features only (X, Y → V).

    Hard data pin known cells. At each unknown cell the DNN is queried with that
    cell's normalized (X, Y); the conditioning pool is not used as model input.
    """
    xy_scale = _xy_scale(meta, hard_df)
    scale = np.where(xy_scale == 0, 1.0, xy_scale)
    rng = np.random.default_rng(int(seed))
    grid, hard_mask, _hard = _pin_hard_data(hard_df, grid_shape)

    unknown_y, unknown_x = np.where(~hard_mask)
    if len(unknown_y) == 0:
        return grid

    order = rng.permutation(len(unknown_y))
    path_y = unknown_y[order]
    path_x = unknown_x[order]

    n_path = len(path_y)
    report_every = max(1, n_path // 100)

    for i in range(n_path):
        gx = int(path_x[i]) + 1
        gy = int(path_y[i]) + 1
        features = np.array([[gx, gy]], dtype=float) / scale

        proba = _predict_proba_row(model, features)
        class_idx = _sample_class_index(proba[0], rng)
        drawn = float(meta.map_indices_to_classes(np.array([class_idx]))[0])

        grid[path_y[i], path_x[i]] = drawn

        if progress_callback is not None and (i + 1 == n_path or (i + 1) % report_every == 0):
            progress_callback(i + 1, n_path)

    return grid
