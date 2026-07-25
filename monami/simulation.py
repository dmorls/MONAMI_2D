"""Classic sequential simulation helpers for trained DNNs."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from monami.features import build_feature_matrix_from_targets, compute_xy_scale
from monami.ml import ModelMeta
from monami.sis import apply_proportion_servo


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


def _class_to_idx_map(meta: ModelMeta) -> dict[int, int]:
    return {int(k): int(v) for k, v in (meta.class_to_idx or {}).items()}


def _target_proportions_and_hard_counts(
    hard_values: np.ndarray,
    meta: ModelMeta,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample proportions and hard counts aligned to model class indices."""
    n_classes = int(meta.n_classes)
    class_to_idx = _class_to_idx_map(meta)
    counts = np.zeros(n_classes, dtype=float)
    for value in np.asarray(hard_values).ravel():
        idx = class_to_idx.get(int(value))
        if idx is not None and 0 <= idx < n_classes:
            counts[idx] += 1.0
    total = float(counts.sum())
    if total <= 0.0:
        target = np.ones(n_classes, dtype=float) / max(n_classes, 1)
    else:
        target = counts / total
    return target, counts.copy()


def _servo_corrected_proba(
    local_proba: np.ndarray,
    meta: ModelMeta,
    target_proportions: np.ndarray,
    current_counts: np.ndarray,
    total_cells: int,
    completed_cells: int,
    correction_strength: float,
) -> np.ndarray:
    local = np.asarray(local_proba, dtype=float).ravel()
    n_classes = int(meta.n_classes)
    if local.size != n_classes:
        # Fall back to raw local probs if shapes disagree.
        return local
    if float(correction_strength) <= 0.0:
        total = float(local.sum())
        return local / total if total > 0 else np.ones(n_classes) / n_classes
    return apply_proportion_servo(
        local,
        target_proportions,
        current_counts,
        total_cells,
        completed_cells,
        float(correction_strength),
    )


def sequential_simulate_grid(
    model,
    meta: ModelMeta,
    hard_df: pd.DataFrame,
    grid_shape: Tuple[int, int],
    *,
    seed: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    include_target_xy: bool = False,
    correction_strength: float = 0.5,
) -> np.ndarray:
    """
    Sequential categorical simulation with a random path (MONAMI neighbor features).

    Hard data (typically training samples) are fixed. Unsampled cells are visited
    in random order; each draw is sampled from the DNN softmax using the growing
    conditioning pool (hard data + previously simulated values).

    When ``include_target_xy`` is True (Hybrid Position), normalized absolute
    (X, Y) are prepended to the neighbor feature vector.

    ``correction_strength`` blends local softmax probs toward remaining sample
    proportions (same servo as Corrected SIS).
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

    n_rows, n_cols = grid_shape
    total_cells = int(n_rows * n_cols)
    hard_count = int(hard_mask.sum())
    target_proportions, current_counts = _target_proportions_and_hard_counts(
        hard["V"].to_numpy(),
        meta,
    )

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
            include_target_xy=include_target_xy,
        )

        proba = _predict_proba_row(model, features)
        corrected = _servo_corrected_proba(
            proba[0],
            meta,
            target_proportions,
            current_counts,
            total_cells,
            hard_count + i,
            correction_strength,
        )
        class_idx = _sample_class_index(corrected, rng)
        drawn = float(meta.map_indices_to_classes(np.array([class_idx]))[0])

        grid[path_y[i], path_x[i]] = drawn
        cond_x.append(float(gx))
        cond_y.append(float(gy))
        cond_v.append(drawn)
        current_counts[class_idx] += 1.0

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
    correction_strength: float = 0.5,
) -> np.ndarray:
    """
    Sequential categorical simulation with coordinate features only (X, Y → V).

    Hard data pin known cells. At each unknown cell the DNN is queried with that
    cell's normalized (X, Y); the conditioning pool is not used as model input.

    Global category counts still update for the sample-proportion servo.
    """
    xy_scale = _xy_scale(meta, hard_df)
    scale = np.where(xy_scale == 0, 1.0, xy_scale)
    rng = np.random.default_rng(int(seed))
    grid, hard_mask, hard = _pin_hard_data(hard_df, grid_shape)

    unknown_y, unknown_x = np.where(~hard_mask)
    if len(unknown_y) == 0:
        return grid

    order = rng.permutation(len(unknown_y))
    path_y = unknown_y[order]
    path_x = unknown_x[order]

    n_rows, n_cols = grid_shape
    total_cells = int(n_rows * n_cols)
    hard_count = int(hard_mask.sum())
    target_proportions, current_counts = _target_proportions_and_hard_counts(
        hard["V"].to_numpy(),
        meta,
    )

    n_path = len(path_y)
    report_every = max(1, n_path // 100)

    for i in range(n_path):
        gx = int(path_x[i]) + 1
        gy = int(path_y[i]) + 1
        features = np.array([[gx, gy]], dtype=float) / scale

        proba = _predict_proba_row(model, features)
        corrected = _servo_corrected_proba(
            proba[0],
            meta,
            target_proportions,
            current_counts,
            total_cells,
            hard_count + i,
            correction_strength,
        )
        class_idx = _sample_class_index(corrected, rng)
        drawn = float(meta.map_indices_to_classes(np.array([class_idx]))[0])

        grid[path_y[i], path_x[i]] = drawn
        current_counts[class_idx] += 1.0

        if progress_callback is not None and (i + 1 == n_path or (i + 1) % report_every == 0):
            progress_callback(i + 1, n_path)

    return grid
