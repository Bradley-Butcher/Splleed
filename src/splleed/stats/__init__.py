"""Statistical analysis utilities for benchmark results."""

from .confidence import ConfidenceInterval, aggregate_trial_values, compute_ci

__all__ = [
    "ConfidenceInterval",
    "compute_ci",
    "aggregate_trial_values",
]
