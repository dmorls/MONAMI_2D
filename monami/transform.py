"""Array transforms: easy format, categorization."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd

CategorizationMethod = Literal["quantile", "equal_width", "custom"]


def numpy2d_to_easyformat(array_2d: np.ndarray) -> np.ndarray:
    """Convert 2D grid to (N, 3) array with 1-based X, Y and value V."""
    rows, cols = array_2d.shape
    y_idx, x_idx = np.indices((rows, cols))
    return np.column_stack((x_idx.ravel() + 1, y_idx.ravel() + 1, array_2d.ravel()))


def easyformat_to_numpy2d(easy: np.ndarray) -> np.ndarray:
    """Convert easy-format array back to 2D grid."""
    rows = int(easy[:, 1].max())
    cols = int(easy[:, 0].max())
    out = np.zeros((rows, cols), dtype=easy.dtype)
    x = easy[:, 0].astype(int) - 1
    y = easy[:, 1].astype(int) - 1
    out[y, x] = easy[:, 2]
    return out


def numpy2d_to_dataframe(easy: np.ndarray) -> pd.DataFrame:
    """Convert easy-format array to DataFrame with X, Y, V columns."""
    df = pd.DataFrame(easy, columns=["X", "Y", "V"])
    df["X"] = df["X"].astype(int)
    df["Y"] = df["Y"].astype(int)
    return df


def compute_quantile_thresholds(values: np.ndarray, n_categories: int) -> list[float]:
    """Return ``n_categories + 1`` bin edges using quantile breaks."""
    quantiles = np.linspace(0, 1, n_categories + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        vmin, vmax = float(values.min()), float(values.max())
        return [vmin, vmax]
    return edges.tolist()


def compute_equal_width_thresholds(
    values: np.ndarray,
    n_categories: int,
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[float]:
    """Return evenly spaced bin edges between ``vmin`` and ``vmax``."""
    lo = float(values.min()) if vmin is None else float(vmin)
    hi = float(values.max()) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-9
    return np.linspace(lo, hi, n_categories + 1).tolist()


def parse_threshold_text(text: str, fallback_values: np.ndarray) -> list[float]:
    """Parse comma-separated threshold string into sorted unique edges."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("Provide at least two threshold values separated by commas.")
    edges = sorted(float(p) for p in parts)
    if len(set(edges)) != len(edges):
        raise ValueError("Threshold values must be unique.")
    return edges


def thresholds_to_text(edges: Sequence[float], precision: int = 6) -> str:
    """Format bin edges for display in a text input."""
    return ", ".join(f"{float(v):.{precision}g}" for v in edges)


def compute_thresholds(
    values: np.ndarray,
    n_categories: int,
    method: CategorizationMethod = "quantile",
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[float]:
    """Compute categorization bin edges for the chosen method."""
    if method == "quantile":
        return compute_quantile_thresholds(values, n_categories)
    if method == "equal_width":
        return compute_equal_width_thresholds(values, n_categories, vmin=vmin, vmax=vmax)
    raise ValueError(f"Unsupported auto method: {method}")


def categorize(values_1d: np.ndarray, rows: int, columns: int, q_categories: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Quantile-based discretization into ``q_categories`` classes (legacy API)."""
    edges = compute_quantile_thresholds(values_1d, q_categories)
    return categorize_with_thresholds(values_1d, rows, columns, edges)


def categorize_with_thresholds(
    values_1d: np.ndarray,
    rows: int,
    columns: int,
    bin_edges: Sequence[float],
) -> tuple[pd.DataFrame, np.ndarray, list[float]]:
    """Discretize continuous values using explicit bin edges."""
    edges = list(bin_edges)
    n_categories = len(edges) - 1
    if n_categories < 1:
        raise ValueError("At least two thresholds are required to define one category.")

    labels = list(range(n_categories))
    categories = pd.cut(
        values_1d,
        bins=edges,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )
    bins = pd.cut(values_1d, bins=edges, include_lowest=True, duplicates="drop")
    categorized_df = pd.DataFrame({"categories": categories, "bins": bins})
    categorized_2d = np.reshape(categories.to_numpy(), (rows, columns))
    actual_edges = edges
    return categorized_df, categorized_2d.astype(float), actual_edges


def categorize_slice(
    slice_2d: np.ndarray,
    n_categories: int,
    method: CategorizationMethod = "quantile",
    bin_edges: Sequence[float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[float]]:
    """Discretize a 2D slice using quantile, equal-width, or custom thresholds."""
    rows, columns = slice_2d.shape
    flat = slice_2d.ravel()
    if bin_edges is None:
        edges = compute_thresholds(flat, n_categories, method=method, vmin=vmin, vmax=vmax)
    else:
        edges = list(bin_edges)
    return categorize_with_thresholds(flat, rows, columns, edges)
