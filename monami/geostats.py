"""Categorical geostatistics: indicator variograms."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

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


SUPPORTED_VARIOGRAM_MODELS = ("spherical", "exponential", "gaussian")


@dataclass
class IndicatorVariogramModel:
    """Serializable variogram model fitted to one category indicator."""

    category: int
    model: str
    nugget: float
    partial_sill: float
    range_major: float
    range_x: float
    range_y: float
    fit_rmse: float
    directional: bool = False
    fallback: bool = False

    @property
    def total_sill(self) -> float:
        return float(self.nugget + self.partial_sill)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IndicatorVariogramModel":
        return cls(**payload)


def variogram_values(
    distance: np.ndarray | float,
    model: str,
    nugget: float,
    partial_sill: float,
    effective_range: float,
) -> np.ndarray:
    """Evaluate a nugget + structured semivariogram model.

    ``effective_range`` is the practical range for exponential/Gaussian models
    (approximately 95% of the partial sill), and the exact range for spherical.
    """
    h = np.asarray(distance, dtype=float)
    r = np.maximum(float(effective_range), 1e-9)
    u = np.maximum(h, 0.0) / r
    if model == "spherical":
        structure = np.where(u < 1.0, 1.5 * u - 0.5 * u**3, 1.0)
    elif model == "exponential":
        structure = 1.0 - np.exp(-3.0 * u)
    elif model == "gaussian":
        structure = 1.0 - np.exp(-3.0 * u**2)
    else:
        raise ValueError(f"Unsupported variogram model: {model}")
    gamma = float(nugget) + float(partial_sill) * structure
    # Coincident points have zero semivariance despite a fitted nugget.
    return np.where(h <= 1e-12, 0.0, gamma)


def anisotropic_distance(
    offsets: np.ndarray,
    range_x: float,
    range_y: float,
    reference_range: float,
) -> np.ndarray:
    """Map X/Y offsets to equivalent isotropic distances."""
    arr = np.asarray(offsets, dtype=float)
    rx = max(float(range_x), 1e-9)
    ry = max(float(range_y), 1e-9)
    ref = max(float(reference_range), 1e-9)
    return ref * np.sqrt((arr[..., 0] / rx) ** 2 + (arr[..., 1] / ry) ** 2)


def indicator_covariance(offsets: np.ndarray, model: IndicatorVariogramModel) -> np.ndarray:
    """Evaluate indicator covariance from coordinate offsets."""
    h = anisotropic_distance(
        offsets,
        model.range_x,
        model.range_y,
        model.range_major,
    )
    gamma = variogram_values(
        h,
        model.model,
        model.nugget,
        model.partial_sill,
        model.range_major,
    )
    return np.maximum(model.total_sill - gamma, 0.0)


def _finite_variogram_points(
    lags: np.ndarray,
    experimental: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(lags, dtype=float)
    y = np.asarray(experimental, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return x[mask], y[mask]


def _fit_variogram_candidate(
    lags: np.ndarray,
    experimental: np.ndarray,
    model: str,
    indicator_variance: float,
) -> tuple[float, float, float, float]:
    x, y = _finite_variogram_points(lags, experimental)
    if len(x) < 3:
        raise ValueError("At least three finite lag bins are required")
    maxlag = max(float(np.max(x)), 1.0)
    minlag = max(float(np.min(x)), 1e-6)
    variance = max(float(indicator_variance), 1e-4)

    def _curve(h, nugget, partial_sill, effective_range):
        return variogram_values(h, model, nugget, partial_sill, effective_range)

    sigma = np.sqrt(np.maximum(x / maxlag, 0.05))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        params, _ = curve_fit(
            _curve,
            x,
            y,
            p0=(min(float(np.nanmin(y)), variance * 0.2), variance, maxlag / 2.0),
            bounds=(
                (0.0, 1e-6, minlag / 4.0),
                (max(variance * 1.5, 0.25), max(variance * 3.0, 0.5), maxlag * 4.0),
            ),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=20_000,
        )
    fitted = _curve(x, *params)
    weights = 1.0 / np.maximum(x, minlag)
    rmse = float(np.sqrt(np.average((fitted - y) ** 2, weights=weights)))
    return float(params[0]), float(params[1]), float(params[2]), rmse


def _fit_directional_range(
    samples_df: pd.DataFrame,
    category: int,
    azimuth: float,
    n_lags: int,
    maxlag: Optional[float],
    tolerance: float,
    model_name: str,
    nugget: float,
    partial_sill: float,
    fallback_range: float,
) -> float:
    try:
        lags, experimental = indicator_variogram_directional(
            samples_df,
            category,
            azimuth=azimuth,
            n_lags=n_lags,
            maxlag=maxlag,
            tolerance=tolerance,
        )
        x, y = _finite_variogram_points(lags, experimental)
        if len(x) < 3:
            return float(fallback_range)

        def _curve(h, effective_range):
            return variogram_values(
                h,
                model_name,
                nugget,
                partial_sill,
                effective_range,
            )

        minlag = max(float(np.min(x)), 1e-6)
        max_x = max(float(np.max(x)), minlag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(
                _curve,
                x,
                y,
                p0=(fallback_range,),
                bounds=((minlag / 4.0,), (max_x * 4.0,)),
                maxfev=10_000,
            )
        return float(params[0])
    except Exception:
        return float(fallback_range)


def fit_indicator_variogram_model(
    samples_df: pd.DataFrame,
    category: int,
    *,
    model: str = "auto",
    n_lags: int = 15,
    maxlag: Optional[float] = None,
    directional: bool = True,
    tolerance: float = 22.5,
) -> IndicatorVariogramModel:
    """Fit one indicator variogram using sampled data only.

    With ``model='auto'``, spherical, exponential, and Gaussian candidates are
    compared by short-lag-weighted RMSE. Directional X/Y ranges are fitted only
    after the omnidirectional structure is stable; failures fall back safely.
    """
    if model != "auto" and model not in SUPPORTED_VARIOGRAM_MODELS:
        raise ValueError(f"Unknown variogram model: {model}")
    coords, values = sample_to_coordinates(samples_df)
    indicator = (values == int(category)).astype(float)
    proportion = float(np.mean(indicator))
    indicator_variance = proportion * (1.0 - proportion)
    fallback_range = max(_default_maxlag(coords) / 2.0, 1.0)
    candidates = SUPPORTED_VARIOGRAM_MODELS if model == "auto" else (model,)

    try:
        lags, experimental = indicator_variogram(
            samples_df,
            int(category),
            n_lags=int(n_lags),
            maxlag=maxlag,
        )
        fits = [
            (*_fit_variogram_candidate(lags, experimental, name, indicator_variance), name)
            for name in candidates
        ]
        nugget, partial_sill, effective_range, rmse, selected = min(
            fits,
            key=lambda item: item[3],
        )
        fallback = False
    except Exception:
        selected = candidates[0]
        nugget = 0.0
        partial_sill = max(indicator_variance, 1e-4)
        effective_range = fallback_range
        # JSON-safe sentinel; ``fallback`` carries the semantic meaning.
        rmse = -1.0
        fallback = True

    range_x = float(effective_range)
    range_y = float(effective_range)
    used_directional = False
    if directional and len(samples_df) >= max(20, n_lags * 2):
        range_x = _fit_directional_range(
            samples_df,
            int(category),
            0.0,
            int(n_lags),
            maxlag,
            tolerance,
            selected,
            nugget,
            partial_sill,
            effective_range,
        )
        range_y = _fit_directional_range(
            samples_df,
            int(category),
            90.0,
            int(n_lags),
            maxlag,
            tolerance,
            selected,
            nugget,
            partial_sill,
            effective_range,
        )
        used_directional = not (
            np.isclose(range_x, effective_range) and np.isclose(range_y, effective_range)
        )

    return IndicatorVariogramModel(
        category=int(category),
        model=str(selected),
        nugget=float(nugget),
        partial_sill=float(partial_sill),
        range_major=float(effective_range),
        range_x=float(range_x),
        range_y=float(range_y),
        fit_rmse=float(rmse),
        directional=bool(used_directional),
        fallback=bool(fallback),
    )


def fit_indicator_variogram_models(
    samples_df: pd.DataFrame,
    categories: Optional[List[int]] = None,
    **kwargs,
) -> Dict[int, IndicatorVariogramModel]:
    """Fit one valid indicator model per category."""
    if categories is None:
        categories = sorted(int(v) for v in samples_df["V"].unique())
    return {
        int(category): fit_indicator_variogram_model(
            samples_df,
            int(category),
            **kwargs,
        )
        for category in categories
    }


def category_proportions(
    values: np.ndarray,
    categories: List[int],
) -> np.ndarray:
    """Return normalized category frequencies in the requested order."""
    arr = np.asarray(values).astype(int).ravel()
    if len(arr) == 0:
        return np.zeros(len(categories), dtype=float)
    return np.array([np.mean(arr == int(c)) for c in categories], dtype=float)


def hard_data_reproduction_rate(
    grid: np.ndarray,
    samples_df: pd.DataFrame,
) -> float:
    """Fraction of hard sample locations reproduced exactly by a field."""
    if samples_df is None or len(samples_df) == 0:
        return 1.0
    rows, cols = grid.shape
    x = samples_df["X"].to_numpy(dtype=int) - 1
    y = samples_df["Y"].to_numpy(dtype=int) - 1
    valid = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)
    if not np.all(valid):
        return 0.0
    observed = samples_df["V"].to_numpy(dtype=int)
    return float(np.mean(np.asarray(grid)[y, x].astype(int) == observed))


def _sample_pair_variogram_rmse(
    grid: np.ndarray,
    variogram: IndicatorVariogramModel,
    *,
    n_lags: int,
    seed: int,
    max_pairs: int,
) -> float:
    rows, cols = grid.shape
    total = rows * cols
    if total < 2:
        return 0.0
    rng = np.random.default_rng(int(seed) + int(variogram.category) * 10_007)
    n_pairs = min(max(int(max_pairs), n_lags * 100), total * 20)
    first = rng.integers(0, total, size=n_pairs)
    second = rng.integers(0, total, size=n_pairs)
    distinct = first != second
    first = first[distinct]
    second = second[distinct]
    y1, x1 = np.divmod(first, cols)
    y2, x2 = np.divmod(second, cols)
    offsets = np.column_stack((x2 - x1, y2 - y1)).astype(float)
    equivalent_h = anisotropic_distance(
        offsets,
        variogram.range_x,
        variogram.range_y,
        variogram.range_major,
    )
    maxlag = max(rows, cols) / 2.0
    keep = equivalent_h <= maxlag
    if int(np.sum(keep)) < n_lags:
        keep = np.ones_like(equivalent_h, dtype=bool)
        maxlag = float(np.max(equivalent_h)) if len(equivalent_h) else 1.0
    equivalent_h = equivalent_h[keep]
    indicator = (np.asarray(grid).ravel().astype(int) == variogram.category).astype(float)
    semivariance = 0.5 * (indicator[first[keep]] - indicator[second[keep]]) ** 2
    edges = np.linspace(0.0, max(maxlag, 1e-6), int(n_lags) + 1)
    bin_index = np.digitize(equivalent_h, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    experimental = np.full(int(n_lags), np.nan)
    weights = np.zeros(int(n_lags), dtype=float)
    for idx in range(int(n_lags)):
        mask = bin_index == idx
        if np.any(mask):
            experimental[idx] = float(np.mean(semivariance[mask]))
            weights[idx] = float(np.sum(mask))
    valid = np.isfinite(experimental) & (weights > 0)
    if not np.any(valid):
        return -1.0
    expected = variogram_values(
        centers[valid],
        variogram.model,
        variogram.nugget,
        variogram.partial_sill,
        variogram.range_major,
    )
    return float(
        np.sqrt(
            np.average(
                (experimental[valid] - expected) ** 2,
                weights=weights[valid],
            )
        )
    )


def _grid_transition_matrix(
    grid: np.ndarray,
    categories: List[int],
    axis: str,
    lag: int = 1,
) -> np.ndarray:
    arr = np.asarray(grid).astype(int)
    lag = max(int(lag), 1)
    if axis == "x":
        if lag >= arr.shape[1]:
            return np.zeros((len(categories), len(categories)), dtype=float)
        source, target = arr[:, :-lag], arr[:, lag:]
    elif axis == "y":
        if lag >= arr.shape[0]:
            return np.zeros((len(categories), len(categories)), dtype=float)
        source, target = arr[:-lag, :], arr[lag:, :]
    else:
        raise ValueError("axis must be 'x' or 'y'")
    positions = {int(c): i for i, c in enumerate(categories)}
    counts = np.zeros((len(categories), len(categories)), dtype=float)
    for src, dst in zip(source.ravel(), target.ravel()):
        if int(src) in positions and int(dst) in positions:
            counts[positions[int(src)], positions[int(dst)]] += 1.0
    row_sum = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)


def _sample_transition_matrix(
    samples_df: pd.DataFrame,
    categories: List[int],
    axis: str,
    angular_tolerance_deg: float = 22.5,
) -> tuple[np.ndarray, int]:
    """Estimate directional transitions using each sample's nearest forward pair."""
    coords = samples_df[["X", "Y"]].to_numpy(dtype=float)
    values = samples_df["V"].to_numpy(dtype=int)
    positions = {int(c): i for i, c in enumerate(categories)}
    counts = np.zeros((len(categories), len(categories)), dtype=float)
    primary_distances: List[float] = []
    tan_tolerance = float(np.tan(np.deg2rad(angular_tolerance_deg)))
    for i, origin in enumerate(coords):
        delta = coords - origin
        if axis == "x":
            forward = (delta[:, 0] > 0) & (
                np.abs(delta[:, 1]) <= tan_tolerance * delta[:, 0]
            )
        elif axis == "y":
            forward = (delta[:, 1] > 0) & (
                np.abs(delta[:, 0]) <= tan_tolerance * delta[:, 1]
            )
        else:
            raise ValueError("axis must be 'x' or 'y'")
        candidates = np.flatnonzero(forward)
        if not len(candidates):
            continue
        distances = np.linalg.norm(delta[candidates], axis=1)
        j = int(candidates[int(np.argmin(distances))])
        src, dst = int(values[i]), int(values[j])
        if src in positions and dst in positions:
            counts[positions[src], positions[dst]] += 1.0
            primary_distances.append(
                float(abs(delta[j, 0] if axis == "x" else delta[j, 1]))
            )
    row_sum = counts.sum(axis=1, keepdims=True)
    matrix = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)
    lag = max(int(round(float(np.median(primary_distances)))), 1) if primary_distances else 1
    return matrix, lag


def corrected_sis_validation_metrics(
    grid: np.ndarray,
    samples_df: pd.DataFrame,
    model: Any,
    *,
    seed: int = 42,
    max_variogram_pairs: int = 100_000,
) -> Dict[str, Any]:
    """Measure realization fidelity against sample-derived ccSIS targets."""
    categories = [int(c) for c in model.categories]
    sample_p = category_proportions(samples_df["V"].to_numpy(), categories)
    field_p = category_proportions(grid, categories)
    difference = field_p - sample_p
    variogram_rmse = {
        str(category): _sample_pair_variogram_rmse(
            np.asarray(grid),
            model.variogram_for(category),
            n_lags=int(model.n_lags),
            seed=int(seed),
            max_pairs=int(max_variogram_pairs),
        )
        for category in categories
    }
    finite_vrmse = [value for value in variogram_rmse.values() if value >= 0]
    transition_errors: Dict[str, float] = {}
    transition_lags: Dict[str, int] = {}
    for axis in ("x", "y"):
        sample_matrix, lag = _sample_transition_matrix(samples_df, categories, axis)
        grid_matrix = _grid_transition_matrix(grid, categories, axis, lag=lag)
        supported_rows = sample_matrix.sum(axis=1) > 0
        transition_errors[axis] = (
            float(
                np.linalg.norm(
                    grid_matrix[supported_rows] - sample_matrix[supported_rows],
                    ord="fro",
                )
            )
            if np.any(supported_rows)
            else -1.0
        )
        transition_lags[axis] = int(lag)
    return {
        "hard_data_fidelity": hard_data_reproduction_rate(grid, samples_df),
        "sample_proportions": sample_p.tolist(),
        "field_proportions": field_p.tolist(),
        "proportion_l1": float(np.sum(np.abs(difference))),
        "proportion_rmse": float(np.sqrt(np.mean(difference**2))),
        "variogram_rmse_by_category": variogram_rmse,
        "variogram_rmse_mean": (
            float(np.mean(finite_vrmse)) if finite_vrmse else -1.0
        ),
        "transition_error_x": transition_errors["x"],
        "transition_error_y": transition_errors["y"],
        "transition_lag_x": transition_lags["x"],
        "transition_lag_y": transition_lags["y"],
    }
