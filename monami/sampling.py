"""Stratified random sampling on 2D grids."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _stratum_edges(length: int, n_strata: int) -> list[int]:
    """Partition ``[0, length)`` into ``n_strata`` contiguous index ranges."""
    edges = [int(j * length / n_strata) for j in range(n_strata + 1)]
    edges[-1] = length
    return edges


def sample_fraction_to_strata(
    height: int,
    width: int,
    sample_fraction: float,
) -> tuple[int, int, int]:
    """
    Choose stratified grid dimensions from a target fraction of exhaustive cells.

    Returns ``(n_h, n_v, n_samples)`` where ``n_samples = n_h * n_v``.
    ``sample_fraction`` is in ``(0, 1]`` (e.g. 0.05 for 5% of the image).
    """
    if height < 2 or width < 2:
        raise ValueError(f"Grid must be at least 2×2, got {height}×{width}")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")

    total = height * width
    target = max(4, min(total, int(round(sample_fraction * total))))

    aspect = width / height
    n_h = max(2, min(width, int(round(np.sqrt(target * aspect)))))
    n_v = max(2, min(height, max(2, int(round(target / n_h)))))

    while n_h * n_v < target and (n_h < width or n_v < height):
        if n_h < width and (n_v >= height or n_h <= n_v):
            n_h += 1
        elif n_v < height:
            n_v += 1
        else:
            break

    while n_h * n_v > target and n_h > 2 and n_v > 2:
        if n_h >= n_v and n_h > 2:
            n_h -= 1
        elif n_v > 2:
            n_v -= 1
        else:
            break

    n_h = max(2, min(n_h, width))
    n_v = max(2, min(n_v, height))
    return n_h, n_v, n_h * n_v


def stratified_sample(
    array_2d: np.ndarray,
    n_h: int,
    n_v: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Stratified random sampling returning (N, 4) array: index, X, Y, V.

    ``n_h`` and ``n_v`` are the number of horizontal and vertical strata.
    Strata are sized proportionally so the full grid is covered, including
    any remainder columns/rows on the right and bottom edges.
    """
    rng = np.random.default_rng(seed)
    height, width = array_2d.shape
    if n_h > width:
        raise ValueError(f"Horizontal strata {n_h} exceeds grid width {width}")
    if n_v > height:
        raise ValueError(f"Vertical strata {n_v} exceeds grid height {height}")

    x_edges = _stratum_edges(width, n_h)
    y_edges = _stratum_edges(height, n_v)
    n_samples = n_h * n_v
    out = np.zeros((n_samples, 4))
    k = 0
    for i in range(n_v):
        for j in range(n_h):
            x_lo, x_hi = x_edges[j], x_edges[j + 1]
            y_lo, y_hi = y_edges[i], y_edges[i + 1]
            sim_x = int(rng.integers(x_lo, x_hi)) if x_hi > x_lo else min(x_lo, width - 1)
            sim_y = int(rng.integers(y_lo, y_hi)) if y_hi > y_lo else min(y_lo, height - 1)
            out[k, 0] = k + 1
            out[k, 1] = sim_x + 1
            out[k, 2] = sim_y + 1
            out[k, 3] = array_2d[sim_y, sim_x]
            k += 1
    return out


def stratified_sample_dataframe(
    array_2d: np.ndarray,
    n_h: int,
    n_v: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return stratified sample as DataFrame with X, Y, V (1-based cell indices)."""
    sample_np = stratified_sample(array_2d, n_h, n_v, seed=seed)
    return pd.DataFrame({"X": sample_np[:, 1], "Y": sample_np[:, 2], "V": sample_np[:, 3]})
