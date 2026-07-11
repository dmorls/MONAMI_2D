"""Load SGEMS-style exhaustive 3D property files."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import numpy as np

FileLike = Union[str, Path, TextIO, BinaryIO, io.BytesIO]


@dataclass
class GridMeta:
    """Grid metadata parsed from file header."""

    nx: int
    ny: int
    nz: int
    property_name: str
    lines_jump: int

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """Volume shape as (n_levels, n_rows, n_columns) for level slicing."""
        return (self.nz, self.ny, self.nx)

    @property
    def slice_shape(self) -> tuple[int, int]:
        """Shape of a single XY slice (rows, columns)."""
        return (self.ny, self.nx)


def parse_header_lines(header_lines: list[str]) -> GridMeta:
    """Parse SGEMS-style header: ``nx*ny*nz``, property count, property name."""
    dim_match = re.match(r"(\d+)\*(\d+)\*(\d+)", header_lines[0].strip())
    if not dim_match:
        raise ValueError(f"Cannot parse grid dimensions from: {header_lines[0]!r}")
    nx, ny, nz = (int(dim_match.group(i)) for i in range(1, 4))
    property_name = header_lines[2].strip() if len(header_lines) > 2 else "v"
    return GridMeta(nx=nx, ny=ny, nz=nz, property_name=property_name, lines_jump=3)


def _read_header(source: FileLike) -> tuple[GridMeta, FileLike | str | Path]:
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8", errors="replace") as handle:
            header = [handle.readline() for _ in range(3)]
        meta = parse_header_lines(header)
        return meta, source
    if hasattr(source, "read"):
        pos = source.tell() if hasattr(source, "tell") else None
        header = [source.readline() for _ in range(3)]
        if isinstance(header[0], bytes):
            header = [line.decode("utf-8", errors="replace") for line in header]
        meta = parse_header_lines(header)
        return meta, source
    raise TypeError(f"Unsupported source type: {type(source)}")


def load_from_path_or_bytes(source: FileLike, lines_jump: int | None = None) -> tuple[np.ndarray, GridMeta]:
    """Load 3D volume and metadata from path or uploaded bytes."""
    meta, data_source = _read_header(source)
    if lines_jump is not None:
        meta.lines_jump = lines_jump

    values = np.loadtxt(data_source, dtype=np.float32, skiprows=meta.lines_jump)
    expected = meta.nx * meta.ny * meta.nz
    if values.size != expected:
        raise ValueError(f"Expected {expected} values, got {values.size}")
    volume = values.reshape(meta.shape_zyx)
    return volume, meta


def load_exhaustive_3d(
    exhaustive_file: FileLike,
    lines_jump: int | None = None,
    rows: int | None = None,
    columns: int | None = None,
    levels: int | None = None,
) -> np.ndarray:
    """
    Load 3D exhaustive array.

    Legacy API: when ``rows``, ``columns``, ``levels`` are supplied they are interpreted
    as (ny, nx, nz) and mapped to volume shape (levels, columns, rows).
    """
    if rows is not None and columns is not None and levels is not None:
        skip = lines_jump if lines_jump is not None else 3
        values = np.loadtxt(exhaustive_file, dtype=np.float32, skiprows=skip)
        return values.reshape((rows, columns, levels))
    volume, _ = load_from_path_or_bytes(exhaustive_file, lines_jump)
    return volume


def extract_slice(volume: np.ndarray, level: int, meta: GridMeta | None = None) -> np.ndarray:
    """Return 2D slice at ``level`` with shape (rows, columns)."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    if level < 0 or level >= volume.shape[0]:
        raise IndexError(f"Level {level} out of range for volume with {volume.shape[0]} levels")
    return volume[level, :, :].copy()


def save_slice_numpy(path: Path, slice_2d: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, slice_2d, delimiter=",")
