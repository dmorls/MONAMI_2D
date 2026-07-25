"""Corrected Sequential Indicator Simulation using sampled data only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from monami.geostats import (
    IndicatorVariogramModel,
    fit_indicator_variogram_models,
    indicator_covariance,
)


@dataclass
class CorrectedSISModel:
    """Serializable fitted state for corrected SIS."""

    categories: List[int]
    proportions: Dict[str, float]
    variograms: Dict[str, Dict[str, Any]]
    neighborhood_size: int = 24
    max_radius: Optional[float] = None
    correction_strength: float = 0.5
    n_lags: int = 15
    variogram_model: str = "auto"
    directional: bool = True
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CorrectedSISModel":
        return cls(**payload)

    def variogram_for(self, category: int) -> IndicatorVariogramModel:
        return IndicatorVariogramModel.from_dict(self.variograms[str(int(category))])

    def proportion_array(self) -> np.ndarray:
        values = np.array(
            [float(self.proportions[str(int(c))]) for c in self.categories],
            dtype=float,
        )
        total = float(values.sum())
        if total <= 0:
            return np.full(len(values), 1.0 / max(len(values), 1))
        return values / total


def fit_corrected_sis_model(
    samples_df: pd.DataFrame,
    *,
    categories: Optional[List[int]] = None,
    neighborhood_size: int = 24,
    max_radius: Optional[float] = None,
    correction_strength: float = 0.5,
    n_lags: int = 15,
    variogram_model: str = "auto",
    directional: bool = True,
) -> CorrectedSISModel:
    """Fit ccSIS proportions and indicator variograms from samples only."""
    if samples_df is None or len(samples_df) < 3:
        raise ValueError("At least three samples are required for corrected SIS")
    required = {"X", "Y", "V"}
    missing = required.difference(samples_df.columns)
    if missing:
        raise ValueError(f"Samples are missing required columns: {sorted(missing)}")
    if categories is None:
        categories = sorted(int(v) for v in samples_df["V"].unique())
    categories = [int(c) for c in categories]
    observed = set(int(v) for v in samples_df["V"].unique())
    absent = [c for c in categories if c not in observed]
    if absent:
        raise ValueError(
            "Corrected SIS cannot fit categories absent from the samples: "
            + ", ".join(str(c) for c in absent)
        )
    neighborhood_size = int(neighborhood_size)
    if neighborhood_size < 2:
        raise ValueError("Neighborhood size must be at least 2")
    correction_strength = float(correction_strength)
    if not 0.0 <= correction_strength <= 1.0:
        raise ValueError("Proportion correction strength must be between 0 and 1")

    counts = samples_df["V"].astype(int).value_counts()
    proportions = {
        str(c): float(counts.get(c, 0)) / float(len(samples_df))
        for c in categories
    }
    fitted = fit_indicator_variogram_models(
        samples_df,
        categories=categories,
        model=variogram_model,
        n_lags=int(n_lags),
        directional=bool(directional),
    )
    return CorrectedSISModel(
        categories=categories,
        proportions=proportions,
        variograms={str(c): fitted[c].to_dict() for c in categories},
        neighborhood_size=neighborhood_size,
        max_radius=float(max_radius) if max_radius else None,
        correction_strength=correction_strength,
        n_lags=int(n_lags),
        variogram_model=str(variogram_model),
        directional=bool(directional),
        sample_count=int(len(samples_df)),
    )


def _select_neighbor_indices(
    target_xy: np.ndarray,
    conditioning_xy: np.ndarray,
    *,
    neighborhood_size: int,
    max_radius: Optional[float],
    tree: Optional[cKDTree] = None,
) -> np.ndarray:
    if len(conditioning_xy) == 0:
        return np.empty(0, dtype=int)
    k = min(max(int(neighborhood_size), 1), len(conditioning_xy))
    active_tree = tree if tree is not None else cKDTree(conditioning_xy)
    upper = float(max_radius) if max_radius and max_radius > 0 else np.inf
    distances, indices = active_tree.query(target_xy, k=k, distance_upper_bound=upper)
    distances = np.atleast_1d(distances)
    indices = np.atleast_1d(indices).astype(int)
    valid = np.isfinite(distances) & (indices < len(conditioning_xy))
    return indices[valid]


def indicator_kriging_probabilities(
    target_xy: np.ndarray,
    conditioning_xy: np.ndarray,
    conditioning_values: np.ndarray,
    model: CorrectedSISModel,
    *,
    neighbor_indices: Optional[np.ndarray] = None,
    tree: Optional[cKDTree] = None,
) -> np.ndarray:
    """Compute normalized simple-indicator-kriging probabilities."""
    target = np.asarray(target_xy, dtype=float).reshape(2)
    coords = np.asarray(conditioning_xy, dtype=float).reshape(-1, 2)
    values = np.asarray(conditioning_values, dtype=int).ravel()
    priors = model.proportion_array()
    if len(coords) == 0:
        return priors

    coincident = np.flatnonzero(np.all(np.isclose(coords, target), axis=1))
    if len(coincident):
        exact = int(values[coincident[-1]])
        return np.array([1.0 if c == exact else 0.0 for c in model.categories])

    if neighbor_indices is None:
        neighbor_indices = _select_neighbor_indices(
            target,
            coords,
            neighborhood_size=model.neighborhood_size,
            max_radius=model.max_radius,
            tree=tree,
        )
    indices = np.asarray(neighbor_indices, dtype=int)
    if len(indices) < 2:
        return priors
    local_xy = coords[indices]
    local_values = values[indices]
    pair_offsets = local_xy[:, None, :] - local_xy[None, :, :]
    target_offsets = local_xy - target

    estimates: List[float] = []
    for position, category in enumerate(model.categories):
        variogram = model.variogram_for(category)
        covariance = indicator_covariance(pair_offsets, variogram)
        covariance.flat[:: len(local_xy) + 1] += max(
            variogram.total_sill * 1e-8,
            1e-10,
        )
        target_covariance = indicator_covariance(target_offsets, variogram)
        indicator = (local_values == int(category)).astype(float)
        residual = indicator - priors[position]
        try:
            weights = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(covariance, rcond=1e-10) @ residual
        estimate = priors[position] + float(target_covariance @ weights)
        estimates.append(estimate)

    probabilities = np.clip(np.asarray(estimates, dtype=float), 1e-9, 1.0)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0:
        return priors
    return probabilities / total


def apply_proportion_servo(
    local_probabilities: np.ndarray,
    target_proportions: np.ndarray,
    current_counts: np.ndarray,
    total_cells: int,
    completed_cells: int,
    strength: float,
) -> np.ndarray:
    """Blend local kriging probabilities with remaining global proportions."""
    local = np.asarray(local_probabilities, dtype=float)
    target = np.asarray(target_proportions, dtype=float)
    counts = np.asarray(current_counts, dtype=float)
    remaining = max(int(total_cells) - int(completed_cells), 1)
    desired_totals = target * float(total_cells)
    remaining_need = np.maximum(desired_totals - counts, 0.0)
    if remaining_need.sum() <= 0:
        quota = target
    else:
        quota = remaining_need / remaining_need.sum()
    alpha = float(np.clip(strength, 0.0, 1.0))
    if alpha <= 0:
        corrected = local
    elif alpha >= 1:
        corrected = quota
    else:
        # Geometric blending preserves zero/near-zero evidence more faithfully
        # than an arithmetic blend while gradually steering global counts.
        corrected = np.exp(
            (1.0 - alpha) * np.log(np.clip(local, 1e-12, 1.0))
            + alpha * np.log(np.clip(quota, 1e-12, 1.0))
        )
    if remaining == 1 and remaining_need.sum() > 0:
        corrected = np.zeros_like(corrected)
        corrected[int(np.argmax(remaining_need))] = 1.0
    total = float(corrected.sum())
    return corrected / total if total > 0 else target


def _validated_hard_data(
    hard_df: pd.DataFrame,
    grid_shape: Tuple[int, int],
    categories: List[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = map(int, grid_shape)
    coords = hard_df[["X", "Y"]].to_numpy(dtype=float)
    values = hard_df["V"].to_numpy(dtype=int)
    if np.any(coords[:, 0] < 1) or np.any(coords[:, 0] > cols):
        raise ValueError("Hard-data X coordinates fall outside the grid")
    if np.any(coords[:, 1] < 1) or np.any(coords[:, 1] > rows):
        raise ValueError("Hard-data Y coordinates fall outside the grid")
    if not set(values).issubset(set(categories)):
        raise ValueError("Hard data contains categories not present in the fitted model")

    keyed: Dict[tuple[int, int], int] = {}
    for (x, y), value in zip(coords, values):
        key = (int(x), int(y))
        if key in keyed and keyed[key] != int(value):
            raise ValueError(f"Conflicting hard data at coordinate {key}")
        keyed[key] = int(value)
    unique_coords = np.array(list(keyed.keys()), dtype=float)
    unique_values = np.array(list(keyed.values()), dtype=int)
    return unique_coords, unique_values, np.array(list(keyed.keys()), dtype=int)


def predict_indicator_grid(
    model: CorrectedSISModel,
    hard_df: pd.DataFrame,
    grid_shape: Tuple[int, int],
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Deterministic argmax indicator-kriging estimate conditioned on samples."""
    rows, cols = map(int, grid_shape)
    coords, values, hard_keys = _validated_hard_data(
        hard_df,
        grid_shape,
        model.categories,
    )
    tree = cKDTree(coords)
    grid = np.full((rows, cols), -1, dtype=int)
    for (x, y), value in zip(hard_keys, values):
        grid[y - 1, x - 1] = int(value)
    unknown = np.argwhere(grid < 0)
    notify_every = max(len(unknown) // 200, 1)
    for done, (row, col) in enumerate(unknown, start=1):
        target = np.array([col + 1, row + 1], dtype=float)
        probabilities = indicator_kriging_probabilities(
            target,
            coords,
            values,
            model,
            tree=tree,
        )
        grid[row, col] = int(model.categories[int(np.argmax(probabilities))])
        if progress_callback and (done % notify_every == 0 or done == len(unknown)):
            progress_callback(done, len(unknown))
    return grid


def _dynamic_neighbor_indices(
    target: np.ndarray,
    coords: np.ndarray,
    count: int,
    tree: cKDTree,
    tree_size: int,
    model: CorrectedSISModel,
) -> np.ndarray:
    k = min(max(model.neighborhood_size, 1), tree_size)
    upper = float(model.max_radius) if model.max_radius else np.inf
    distances, indices = tree.query(target, k=k, distance_upper_bound=upper)
    candidate_indices = np.atleast_1d(indices).astype(int)
    candidate_distances = np.atleast_1d(distances)
    valid = np.isfinite(candidate_distances) & (candidate_indices < tree_size)
    candidate_indices = candidate_indices[valid]
    candidate_distances = candidate_distances[valid]

    if count > tree_size:
        recent_indices = np.arange(tree_size, count, dtype=int)
        recent_distances = np.linalg.norm(coords[recent_indices] - target, axis=1)
        if model.max_radius:
            keep = recent_distances <= float(model.max_radius)
            recent_indices = recent_indices[keep]
            recent_distances = recent_distances[keep]
        candidate_indices = np.concatenate([candidate_indices, recent_indices])
        candidate_distances = np.concatenate([candidate_distances, recent_distances])

    if len(candidate_indices) <= model.neighborhood_size:
        return candidate_indices
    take = np.argpartition(candidate_distances, model.neighborhood_size - 1)[
        : model.neighborhood_size
    ]
    return candidate_indices[take]


def simulate_corrected_sis_grid(
    model: CorrectedSISModel,
    hard_df: pd.DataFrame,
    grid_shape: Tuple[int, int],
    *,
    seed: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Run corrected SIS with a growing conditioning pool and random path."""
    rows, cols = map(int, grid_shape)
    hard_coords, hard_values, hard_keys = _validated_hard_data(
        hard_df,
        grid_shape,
        model.categories,
    )
    total_cells = rows * cols
    grid = np.full((rows, cols), -1, dtype=int)
    for (x, y), value in zip(hard_keys, hard_values):
        grid[y - 1, x - 1] = int(value)

    unknown = np.argwhere(grid < 0)
    rng = np.random.default_rng(int(seed))
    path = unknown[rng.permutation(len(unknown))]
    coords = np.empty((total_cells, 2), dtype=float)
    values = np.empty(total_cells, dtype=int)
    hard_count = len(hard_coords)
    coords[:hard_count] = hard_coords
    values[:hard_count] = hard_values
    count = hard_count
    tree = cKDTree(coords[:count])
    tree_size = count
    rebuild_interval = max(32, min(256, model.neighborhood_size * 4))

    category_to_position = {c: i for i, c in enumerate(model.categories)}
    current_counts = np.zeros(len(model.categories), dtype=int)
    for value in hard_values:
        current_counts[category_to_position[int(value)]] += 1
    target_proportions = model.proportion_array()
    notify_every = max(len(path) // 200, 1)

    for done, (row, col) in enumerate(path, start=1):
        if count - tree_size >= rebuild_interval:
            tree = cKDTree(coords[:count])
            tree_size = count
        target = np.array([col + 1, row + 1], dtype=float)
        neighbor_indices = _dynamic_neighbor_indices(
            target,
            coords,
            count,
            tree,
            tree_size,
            model,
        )
        local = indicator_kriging_probabilities(
            target,
            coords[:count],
            values[:count],
            model,
            neighbor_indices=neighbor_indices,
        )
        probabilities = apply_proportion_servo(
            local,
            target_proportions,
            current_counts,
            total_cells,
            count,
            model.correction_strength,
        )
        position = int(rng.choice(len(model.categories), p=probabilities))
        category = int(model.categories[position])
        grid[row, col] = category
        coords[count] = target
        values[count] = category
        count += 1
        current_counts[position] += 1
        if progress_callback and (done % notify_every == 0 or done == len(path)):
            progress_callback(done, len(path))
    return grid
