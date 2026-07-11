"""Categorical geostatistics: indicator variograms."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from skgstat import DirectionalVariogram, Variogram
except ImportError:  # pragma: no cover
    DirectionalVariogram = None
    Variogram = None


# scikit-gstat azimuth: East (+X) = 0°, positive angles rotate clockwise toward +Y.
VARIOGRAM_DIRECTION_AZIMUTHS: Dict[str, tuple[str, float]] = {
    "x": ("X", 0.0),
    "y": ("Y", 90.0),
}


def _require_skgstat() -> None:
    if Variogram is None or DirectionalVariogram is None:
        raise ImportError("scikit-gstat is required for variogram computation. Install with: pip install scikit-gstat")


def sample_to_coordinates(samples_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract spatial coordinates and values from sample DataFrame."""
    x = samples_df["X"].to_numpy(dtype=float)
    y = samples_df["Y"].to_numpy(dtype=float)
    coords = np.column_stack((x, y))
    values = samples_df["V"].to_numpy(dtype=float)
    return coords, values


def _default_maxlag(coords: np.ndarray) -> float:
    ranges = coords.max(axis=0) - coords.min(axis=0)
    return float(max(ranges) / 2) if max(ranges) > 0 else 1.0


def resolve_variogram_directions(
    direction_keys: List[str],
    seed: int | None = None,
) -> List[tuple[str, float]]:
    """
    Map UI direction keys to ``(label, azimuth_deg)`` pairs.

    Supported keys: ``x``, ``y``, ``random``. A random direction uses ``seed`` for reproducibility.
    """
    rng = np.random.default_rng(seed)
    resolved: List[tuple[str, float]] = []
    for key in direction_keys:
        if key == "random":
            azimuth = float(rng.uniform(-180.0, 180.0))
            resolved.append((f"Random ({azimuth:.0f}°)", azimuth))
        elif key in VARIOGRAM_DIRECTION_AZIMUTHS:
            resolved.append(VARIOGRAM_DIRECTION_AZIMUTHS[key])
        else:
            raise ValueError(f"Unknown variogram direction: {key}")
    return resolved


def indicator_variogram(
    samples_df: pd.DataFrame,
    category: int,
    n_lags: int = 15,
    maxlag: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Experimental indicator variogram for a single category.

    Returns (lags, semivariance) arrays.
    """
    _require_skgstat()
    coords, values = sample_to_coordinates(samples_df)
    indicator = (values == category).astype(float)
    if maxlag is None:
        maxlag = _default_maxlag(coords)
    vario = Variogram(coords, indicator, n_lags=n_lags, maxlag=maxlag, normalize=False)
    return vario.bins, vario.experimental


def indicator_variogram_directional(
    samples_df: pd.DataFrame,
    category: int,
    azimuth: float,
    n_lags: int = 15,
    maxlag: Optional[float] = None,
    tolerance: float = 22.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Directional experimental indicator variogram for a single category.

    ``azimuth`` is in degrees (East = 0°, clockwise toward +Y).
    """
    _require_skgstat()
    coords, values = sample_to_coordinates(samples_df)
    indicator = (values == category).astype(float)
    if maxlag is None:
        maxlag = _default_maxlag(coords)
    vario = DirectionalVariogram(
        coords,
        indicator,
        n_lags=n_lags,
        maxlag=maxlag,
        normalize=False,
        azimuth=azimuth,
        tolerance=tolerance,
    )
    return vario.bins, vario.experimental


def indicator_variograms(
    samples_df: pd.DataFrame,
    categories: Optional[List[int]] = None,
    n_lags: int = 15,
    maxlag: Optional[float] = None,
) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    """Compute indicator variograms for multiple categories."""
    values = samples_df["V"].to_numpy()
    if categories is None:
        categories = sorted(int(v) for v in np.unique(values))
    return {
        cat: indicator_variogram(samples_df, cat, n_lags=n_lags, maxlag=maxlag)
        for cat in categories
    }


def indicator_variograms_by_direction(
    samples_df: pd.DataFrame,
    categories: Optional[List[int]] = None,
    directions: Optional[List[tuple[str, float]]] = None,
    n_lags: int = 15,
    maxlag: Optional[float] = None,
    tolerance: float = 22.5,
) -> Dict[str, Dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Compute directional indicator variograms for multiple categories and orientations."""
    values = samples_df["V"].to_numpy()
    if categories is None:
        categories = sorted(int(v) for v in np.unique(values))
    if directions is None:
        directions = [VARIOGRAM_DIRECTION_AZIMUTHS["x"], VARIOGRAM_DIRECTION_AZIMUTHS["y"]]

    coords, _ = sample_to_coordinates(samples_df)
    if maxlag is None:
        maxlag = _default_maxlag(coords)

    return {
        dir_label: {
            cat: indicator_variogram_directional(
                samples_df,
                cat,
                azimuth=azimuth,
                n_lags=n_lags,
                maxlag=maxlag,
                tolerance=tolerance,
            )
            for cat in categories
        }
        for dir_label, azimuth in directions
    }


def top_categories_by_frequency(samples_df: pd.DataFrame, n: int = 5) -> List[int]:
    """Return the ``n`` most frequent categories in the sample."""
    counts = samples_df["V"].value_counts()
    return [int(c) for c in counts.head(n).index]
