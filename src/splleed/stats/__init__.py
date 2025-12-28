"""Statistical analysis utilities for benchmark results."""

from .confidence import ConfidenceInterval, compute_ci

__all__ = [
    "ConfidenceInterval",
    "compute_ci",
]
