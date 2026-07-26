"""MONAMI neighbor-feature construction for 2D categorical training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Direction, distance, and neighbor category only — no absolute X/Y coordinates.
NEIGHBOR_FIELDS = ("dX", "dY", "D", "V")
FEATURES_PER_NEIGHBOR = len(NEIGHBOR_FIELDS)
ABSOLUTE_XY_DIM = 2


def feature_names(n_nearest: int) -> list[str]:
    """Column names for the flattened neighbor feature vector."""
    names: list[str] = []
    for i in range(n_nearest):
        for field in NEIGHBOR_FIELDS:
            names.append(f"{field}_{i}")
    return names


def feature_dim(n_nearest: int) -> int:
    return n_nearest * FEATURES_PER_NEIGHBOR


def hybrid_feature_names(n_nearest: int) -> list[str]:
    """Column names for absolute XY prepended to neighbor features."""
    return ["X", "Y"] + feature_names(n_nearest)


def hybrid_feature_dim(n_nearest: int) -> int:
    return ABSOLUTE_XY_DIM + feature_dim(n_nearest)


def absolute_xy_features(
    target_xy: np.ndarray,
    xy_scale: np.ndarray,
) -> np.ndarray:
    """Normalize absolute target coordinates (same scaling as Absolute Position)."""
    target_xy = np.asarray(target_xy, dtype=float)
    if target_xy.ndim == 1:
        target_xy = target_xy.reshape(1, 2)
    scale = np.asarray(xy_scale, dtype=float).ravel()
    scale = np.where(scale == 0, 1.0, scale)
    return target_xy / scale


def compute_xy_scale(neighbor_df: pd.DataFrame) -> np.ndarray:
    """Max X/Y used to normalize relative offsets and distances."""
    xy = neighbor_df[["X", "Y"]].to_numpy(dtype=float)
    scale = xy.max(axis=0)
    return np.where(scale == 0, 1.0, scale)


def _require_columns(df: pd.DataFrame) -> None:
    missing = {"X", "Y", "V"} - set(df.columns)
    if missing:
        raise ValueError(f"Sample DataFrame missing columns: {sorted(missing)}")


def nearest_neighbor_indices(
    target_xy: np.ndarray,
    neighbor_df: pd.DataFrame,
    n_nearest: int,
    exclude_self: bool = True,
) -> np.ndarray:
    """Return pool row indices of the ``n_nearest`` neighbors (nearest-first).

    Uses the same KD-tree / self-exclusion rules as feature construction.
    """
    _require_columns(neighbor_df)
    if len(neighbor_df) <= n_nearest and exclude_self:
        raise ValueError(
            f"Need more than {n_nearest} neighbor samples when excluding self; got {len(neighbor_df)}"
        )
    if len(neighbor_df) < n_nearest and not exclude_self:
        raise ValueError(
            f"Need at least {n_nearest} neighbor samples; got {len(neighbor_df)}"
        )

    target_xy = np.asarray(target_xy, dtype=float).reshape(1, 2)
    pool_xy = neighbor_df[["X", "Y"]].to_numpy(dtype=float)
    tree = cKDTree(pool_xy)
    k_query = n_nearest + 1 if exclude_self else n_nearest
    k_query = min(k_query, len(neighbor_df))
    dists, idxs = tree.query(target_xy, k=k_query)
    row_idx = np.atleast_1d(idxs[0] if np.ndim(idxs) > 1 else idxs)
    row_dist = np.atleast_1d(dists[0] if np.ndim(dists) > 1 else dists)
    if exclude_self:
        keep = row_dist > 1e-9
        row_idx = row_idx[keep][:n_nearest]
    else:
        row_idx = row_idx[:n_nearest]
    if len(row_idx) < n_nearest:
        raise ValueError(
            f"Target ({target_xy[0, 0]}, {target_xy[0, 1]}) has only "
            f"{len(row_idx)} neighbors after exclusion; need {n_nearest}"
        )
    return np.asarray(row_idx, dtype=int)


def build_feature_row(
    target_x: float,
    target_y: float,
    neighbor_df: pd.DataFrame,
    n_nearest: int,
    exclude_self: bool = False,
    xy_scale: np.ndarray | None = None,
    include_target_xy: bool = False,
) -> np.ndarray:
    """Build one MONAMI feature row for a single target point."""
    matrix = build_feature_matrix_from_targets(
        np.array([[target_x, target_y]], dtype=float),
        neighbor_df,
        n_nearest,
        exclude_self=exclude_self,
        xy_scale=xy_scale,
        include_target_xy=include_target_xy,
    )
    return matrix[0]


def build_feature_matrix_from_targets(
    target_xy: np.ndarray,
    neighbor_df: pd.DataFrame,
    n_nearest: int,
    exclude_self: bool = False,
    xy_scale: np.ndarray | None = None,
    include_target_xy: bool = False,
) -> np.ndarray:
    """
    Build MONAMI features for many target points.

    ``target_xy`` shape: (N, 2). Returns shape (N, 4 * n_nearest), or
    (N, 2 + 4 * n_nearest) when ``include_target_xy`` is True.
    """
    _require_columns(neighbor_df)
    if len(neighbor_df) <= n_nearest and exclude_self:
        raise ValueError(
            f"Need more than {n_nearest} neighbor samples when excluding self; got {len(neighbor_df)}"
        )
    if len(neighbor_df) < n_nearest and not exclude_self:
        raise ValueError(
            f"Need at least {n_nearest} neighbor samples; got {len(neighbor_df)}"
        )

    target_xy = np.asarray(target_xy, dtype=float)
    if target_xy.ndim == 1:
        target_xy = target_xy.reshape(1, 2)

    if xy_scale is None:
        xy_scale = compute_xy_scale(neighbor_df)
    dist_scale = float(np.linalg.norm(xy_scale))

    pool_xy = neighbor_df[["X", "Y"]].to_numpy(dtype=float)
    pool_v = neighbor_df["V"].to_numpy(dtype=float)
    tree = cKDTree(pool_xy)

    k_query = n_nearest + 1 if exclude_self else n_nearest
    k_query = min(k_query, len(neighbor_df))
    dists, idxs = tree.query(target_xy, k=k_query)
    if k_query == 1:
        dists = np.atleast_2d(dists)
        idxs = np.atleast_2d(idxs)

    n_targets = target_xy.shape[0]
    out = np.zeros((n_targets, feature_dim(n_nearest)), dtype=float)

    for i in range(n_targets):
        row_idx = np.atleast_1d(idxs[i])
        row_dist = np.atleast_1d(dists[i])
        if exclude_self:
            keep = row_dist > 1e-9
            row_idx = row_idx[keep][:n_nearest]
        else:
            row_idx = row_idx[:n_nearest]

        if len(row_idx) < n_nearest:
            raise ValueError(
                f"Target ({target_xy[i, 0]}, {target_xy[i, 1]}) has only "
                f"{len(row_idx)} neighbors after exclusion; need {n_nearest}"
            )

        tx, ty = target_xy[i]
        nx = pool_xy[row_idx, 0]
        ny = pool_xy[row_idx, 1]
        dx = nx - tx
        dy = ny - ty
        d = np.sqrt(dx * dx + dy * dy)
        v = pool_v[row_idx]

        for j in range(n_nearest):
            base = j * FEATURES_PER_NEIGHBOR
            out[i, base] = dx[j] / xy_scale[0]
            out[i, base + 1] = dy[j] / xy_scale[1]
            out[i, base + 2] = d[j] / dist_scale
            out[i, base + 3] = v[j]

    if include_target_xy:
        abs_xy = absolute_xy_features(target_xy, xy_scale)
        out = np.concatenate([abs_xy, out], axis=1)
    return out


def build_feature_matrix(
    targets_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    n_nearest: int,
    exclude_self: bool = False,
    xy_scale: np.ndarray | None = None,
    include_target_xy: bool = False,
) -> np.ndarray:
    """Build feature matrix for rows in ``targets_df`` using ``neighbor_df`` as pool."""
    _require_columns(targets_df)
    target_xy = targets_df[["X", "Y"]].to_numpy(dtype=float)
    return build_feature_matrix_from_targets(
        target_xy,
        neighbor_df,
        n_nearest,
        exclude_self=exclude_self,
        xy_scale=xy_scale,
        include_target_xy=include_target_xy,
    )


def build_training_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    neighbor_pool_df: pd.DataFrame,
    n_nearest: int,
    include_target_xy: bool = False,
    ti_samples_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Build train/test feature matrices and category targets.

    Optional ``ti_samples_df`` appends auxiliary training rows whose neighbors
    come from the TI pool only. ``xy_scale`` is taken from the target neighbor
    pool so prediction on the target slice stays consistent.
    """
    xy_scale = compute_xy_scale(neighbor_pool_df)
    x_train = build_feature_matrix(
        train_df,
        neighbor_pool_df,
        n_nearest,
        exclude_self=True,
        xy_scale=xy_scale,
        include_target_xy=include_target_xy,
    )
    y_train = train_df["V"].astype(int).to_numpy()
    x_test = build_feature_matrix(
        test_df,
        neighbor_pool_df,
        n_nearest,
        exclude_self=False,
        xy_scale=xy_scale,
        include_target_xy=include_target_xy,
    )
    y_test = test_df["V"].astype(int).to_numpy()
    names = (
        hybrid_feature_names(n_nearest)
        if include_target_xy
        else feature_names(n_nearest)
    )

    if ti_samples_df is not None and len(ti_samples_df) > 0:
        if len(ti_samples_df) <= n_nearest:
            raise ValueError(
                f"Training image has {len(ti_samples_df)} samples; need more than "
                f"{n_nearest} neighbors for relative/hybrid features."
            )
        x_ti = build_feature_matrix(
            ti_samples_df,
            ti_samples_df,
            n_nearest,
            exclude_self=True,
            xy_scale=xy_scale,
            include_target_xy=include_target_xy,
        )
        y_ti = ti_samples_df["V"].astype(int).to_numpy()
        x_train = np.vstack([x_train, x_ti])
        y_train = np.concatenate([y_train, y_ti])

    return x_train, y_train, x_test, y_test, names, xy_scale
