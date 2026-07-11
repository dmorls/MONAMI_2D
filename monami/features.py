"""MONAMI neighbor-feature construction for 2D categorical training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Direction, distance, and neighbor category only — no absolute X/Y coordinates.
NEIGHBOR_FIELDS = ("dX", "dY", "D", "V")
FEATURES_PER_NEIGHBOR = len(NEIGHBOR_FIELDS)


def feature_names(n_nearest: int) -> list[str]:
    """Column names for the flattened neighbor feature vector."""
    names: list[str] = []
    for i in range(n_nearest):
        for field in NEIGHBOR_FIELDS:
            names.append(f"{field}_{i}")
    return names


def feature_dim(n_nearest: int) -> int:
    return n_nearest * FEATURES_PER_NEIGHBOR


def compute_xy_scale(neighbor_df: pd.DataFrame) -> np.ndarray:
    """Max X/Y used to normalize relative offsets and distances."""
    xy = neighbor_df[["X", "Y"]].to_numpy(dtype=float)
    scale = xy.max(axis=0)
    return np.where(scale == 0, 1.0, scale)


def _require_columns(df: pd.DataFrame) -> None:
    missing = {"X", "Y", "V"} - set(df.columns)
    if missing:
        raise ValueError(f"Sample DataFrame missing columns: {sorted(missing)}")


def build_feature_row(
    target_x: float,
    target_y: float,
    neighbor_df: pd.DataFrame,
    n_nearest: int,
    exclude_self: bool = False,
    xy_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Build one MONAMI feature row for a single target point."""
    matrix = build_feature_matrix_from_targets(
        np.array([[target_x, target_y]], dtype=float),
        neighbor_df,
        n_nearest,
        exclude_self=exclude_self,
        xy_scale=xy_scale,
    )
    return matrix[0]


def build_feature_matrix_from_targets(
    target_xy: np.ndarray,
    neighbor_df: pd.DataFrame,
    n_nearest: int,
    exclude_self: bool = False,
    xy_scale: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build MONAMI features for many target points.

    ``target_xy`` shape: (N, 2). Returns shape (N, 4 * n_nearest).
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

    return out


def build_feature_matrix(
    targets_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    n_nearest: int,
    exclude_self: bool = False,
    xy_scale: np.ndarray | None = None,
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
    )


def build_training_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    neighbor_pool_df: pd.DataFrame,
    n_nearest: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Build train/test feature matrices and category targets."""
    xy_scale = compute_xy_scale(neighbor_pool_df)
    x_train = build_feature_matrix(
        train_df,
        neighbor_pool_df,
        n_nearest,
        exclude_self=True,
        xy_scale=xy_scale,
    )
    y_train = train_df["V"].astype(int).to_numpy()
    x_test = build_feature_matrix(
        test_df,
        neighbor_pool_df,
        n_nearest,
        exclude_self=False,
        xy_scale=xy_scale,
    )
    y_test = test_df["V"].astype(int).to_numpy()
    names = feature_names(n_nearest)
    return x_train, y_train, x_test, y_test, names, xy_scale
