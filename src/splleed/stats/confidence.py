"""Confidence interval computation for benchmark statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# T-distribution critical values for common degrees of freedom at 95% confidence
# Values for two-tailed test: t_(alpha/2, df) where alpha = 0.05
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    50: 2.009,
    100: 1.984,
}

# T-distribution critical values at 99% confidence
_T_CRITICAL_99 = {
    1: 63.657,
    2: 9.925,
    3: 5.841,
    4: 4.604,
    5: 4.032,
    6: 3.707,
    7: 3.499,
    8: 3.355,
    9: 3.250,
    10: 3.169,
    15: 2.947,
    20: 2.845,
    25: 2.787,
    30: 2.750,
    40: 2.704,
    50: 2.678,
    100: 2.626,
}

# T-distribution critical values at 90% confidence
_T_CRITICAL_90 = {
    1: 6.314,
    2: 2.920,
    3: 2.353,
    4: 2.132,
    5: 2.015,
    6: 1.943,
    7: 1.895,
    8: 1.860,
    9: 1.833,
    10: 1.812,
    15: 1.753,
    20: 1.725,
    25: 1.708,
    30: 1.697,
    40: 1.684,
    50: 1.676,
    100: 1.660,
}


def _get_t_critical(df: int, confidence_level: float) -> float:
    """
    Get t-distribution critical value for given degrees of freedom and confidence level.

    Uses lookup table for common values, falls back to normal approximation for large df.

    Args:
        df: Degrees of freedom (n - 1)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Critical t-value for two-tailed test
    """
    # Select appropriate table based on confidence level
    if confidence_level >= 0.99:
        table = _T_CRITICAL_99
        z_fallback = 2.576
    elif confidence_level >= 0.95:
        table = _T_CRITICAL_95
        z_fallback = 1.96
    else:
        table = _T_CRITICAL_90
        z_fallback = 1.645

    # For large df, use normal approximation
    if df > 100:
        return z_fallback

    # Find closest df in table
    if df in table:
        return table[df]

    # Linear interpolation between closest values
    sorted_dfs = sorted(table.keys())
    for i, table_df in enumerate(sorted_dfs):
        if table_df > df:
            if i == 0:
                return table[sorted_dfs[0]]
            lower_df = sorted_dfs[i - 1]
            upper_df = table_df
            # Linear interpolation
            t_lower = table[lower_df]
            t_upper = table[upper_df]
            ratio = (df - lower_df) / (upper_df - lower_df)
            return t_lower + ratio * (t_upper - t_lower)

    return table[sorted_dfs[-1]]


@dataclass
class ConfidenceInterval:
    """A statistical value with confidence interval."""

    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_samples: int
    confidence_level: float = 0.95

    @property
    def margin(self) -> float:
        """Half-width of the confidence interval."""
        return (self.ci_upper - self.ci_lower) / 2

    def __str__(self) -> str:
        """Format as 'mean +/- margin'."""
        if self.n_samples <= 1:
            return f"{self.mean:.2f}"
        return f"{self.mean:.2f} +/- {self.margin:.2f}"

    def format(self, precision: int = 2) -> str:
        """Format with custom precision."""
        if self.n_samples <= 1:
            return f"{self.mean:.{precision}f}"
        return f"{self.mean:.{precision}f} +/- {self.margin:.{precision}f}"


def compute_ci(
    values: Sequence[float] | np.ndarray,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Compute confidence interval for a set of values.

    Uses t-distribution for small samples (n < 30), which is more accurate
    than normal approximation when sample size is limited.

    Args:
        values: Sample values (e.g., from multiple benchmark trials)
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        ConfidenceInterval with mean, bounds, std, and metadata
    """
    if isinstance(values, list):
        values = np.array(values)

    n = len(values)

    if n == 0:
        return ConfidenceInterval(
            mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            std=0.0,
            n_samples=0,
            confidence_level=confidence_level,
        )

    if n == 1:
        val = float(values[0])
        return ConfidenceInterval(
            mean=val,
            ci_lower=val,
            ci_upper=val,
            std=0.0,
            n_samples=1,
            confidence_level=confidence_level,
        )

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))  # Sample standard deviation

    # Standard error of the mean
    sem = std / math.sqrt(n)

    # Get t-critical value
    df = n - 1
    t_crit = _get_t_critical(df, confidence_level)

    # Compute margin of error
    margin = t_crit * sem

    return ConfidenceInterval(
        mean=mean,
        ci_lower=mean - margin,
        ci_upper=mean + margin,
        std=std,
        n_samples=n,
        confidence_level=confidence_level,
    )


def aggregate_trial_values(
    trial_values: Sequence[Sequence[float]],
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Aggregate values across multiple trials.

    For each trial, computes the mean, then computes CI across trial means.

    Args:
        trial_values: List of value lists, one per trial
        confidence_level: Confidence level for CI

    Returns:
        ConfidenceInterval across trials
    """
    if not trial_values:
        return compute_ci([], confidence_level)

    # Compute mean for each trial
    trial_means = [float(np.mean(values)) for values in trial_values if len(values) > 0]

    return compute_ci(trial_means, confidence_level)
