"""Pluggable prediction algorithms."""

from monami.algorithms.base import Algorithm, TrainingResult
from monami.algorithms.registry import (
    DEFAULT_ALGORITHM_ID,
    get_algorithm,
    list_algorithms,
    register,
    resolve_algorithm_id,
    uses_relative_position,
)

__all__ = [
    "Algorithm",
    "TrainingResult",
    "DEFAULT_ALGORITHM_ID",
    "get_algorithm",
    "list_algorithms",
    "register",
    "resolve_algorithm_id",
    "uses_relative_position",
]
