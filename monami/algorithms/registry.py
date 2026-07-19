"""Algorithm registry for pluggable prediction pipelines."""

from __future__ import annotations

from typing import Dict, List

from monami.algorithms.base import Algorithm
from monami.algorithms.default_dnn import DefaultDNNAlgorithm
from monami.algorithms.monami_dnn import MonamiDNNAlgorithm

# Bootstrap still uses the relative-position (neighbor) algorithm.
DEFAULT_ALGORITHM_ID = "2_Relative_Position"

# Older saved models / sessions may still store previous ids.
_LEGACY_ALGORITHM_IDS: Dict[str, str] = {
    "default": "1_Absolute_Position",
    "1_Default": "1_Absolute_Position",
    "monami_dnn": "2_Relative_Position",
    "2_Monami_NN": "2_Relative_Position",
}

# Relative-position / neighbor-feature algorithms (need n_nearest, growing pool, etc.).
RELATIVE_POSITION_IDS = frozenset(
    {
        "2_Relative_Position",
        "2_Monami_NN",
        "monami_dnn",
    }
)

# Register Absolute Position first, then Relative Position. Next new algo: 3_…
_ALGORITHMS: Dict[str, Algorithm] = {}
for _algo in (DefaultDNNAlgorithm(), MonamiDNNAlgorithm()):
    _ALGORITHMS[_algo.id] = _algo


def resolve_algorithm_id(algorithm_id: str) -> str:
    """Map legacy ids to current numbered ids."""
    return _LEGACY_ALGORITHM_IDS.get(str(algorithm_id), str(algorithm_id))


def uses_relative_position(algorithm_id: str) -> bool:
    """Whether the algorithm uses neighbor / relative-position features."""
    return resolve_algorithm_id(algorithm_id) in RELATIVE_POSITION_IDS


def register(algorithm: Algorithm) -> None:
    """Register an algorithm instance by its ``id`` (use next ``N_`` prefix)."""
    _ALGORITHMS[algorithm.id] = algorithm


def get_algorithm(algorithm_id: str) -> Algorithm:
    """Return a registered algorithm or raise KeyError."""
    algorithm_id = resolve_algorithm_id(algorithm_id)
    if algorithm_id not in _ALGORITHMS:
        raise KeyError(
            f"Unknown algorithm '{algorithm_id}'. Available: {list(_ALGORITHMS.keys())}"
        )
    return _ALGORITHMS[algorithm_id]


def list_algorithms() -> List[Algorithm]:
    """Return all registered algorithms in registration order."""
    return list(_ALGORITHMS.values())
