"""Tests for statistics and confidence interval computation."""

import numpy as np

from splleed.stats import ConfidenceInterval, compute_ci


class TestConfidenceInterval:
    """Tests for ConfidenceInterval dataclass."""

    def test_margin_calculation(self):
        """Test that margin is half the CI width."""
        ci = ConfidenceInterval(
            mean=100.0,
            ci_lower=90.0,
            ci_upper=110.0,
            std=5.0,
            n_samples=10,
        )
        assert ci.margin == 10.0

    def test_str_format(self):
        """Test string formatting."""
        ci = ConfidenceInterval(
            mean=100.0,
            ci_lower=90.0,
            ci_upper=110.0,
            std=5.0,
            n_samples=10,
        )
        assert str(ci) == "100.00 +/- 10.00"

    def test_str_format_single_sample(self):
        """Test string formatting with single sample (no CI)."""
        ci = ConfidenceInterval(
            mean=100.0,
            ci_lower=100.0,
            ci_upper=100.0,
            std=0.0,
            n_samples=1,
        )
        assert str(ci) == "100.00"

    def test_format_precision(self):
        """Test custom precision formatting."""
        ci = ConfidenceInterval(
            mean=100.123,
            ci_lower=90.0,
            ci_upper=110.246,
            std=5.0,
            n_samples=10,
        )
        assert ci.format(precision=1) == "100.1 +/- 10.1"


class TestComputeCI:
    """Tests for compute_ci function."""

    def test_empty_values(self):
        """Test with empty input."""
        ci = compute_ci([])
        assert ci.mean == 0.0
        assert ci.n_samples == 0

    def test_single_value(self):
        """Test with single value - no CI possible."""
        ci = compute_ci([100.0])
        assert ci.mean == 100.0
        assert ci.ci_lower == 100.0
        assert ci.ci_upper == 100.0
        assert ci.std == 0.0
        assert ci.n_samples == 1

    def test_two_values(self):
        """Test with two values."""
        ci = compute_ci([90.0, 110.0])
        assert ci.mean == 100.0
        assert ci.n_samples == 2
        # With 2 samples, CI should be wide
        assert ci.ci_lower < ci.mean
        assert ci.ci_upper > ci.mean

    def test_mean_calculation(self):
        """Test that mean is correctly calculated."""
        values = [10, 20, 30, 40, 50]
        ci = compute_ci(values)
        assert ci.mean == 30.0

    def test_ci_contains_mean(self):
        """Test that CI always contains the mean."""
        values = [100, 105, 95, 110, 90]
        ci = compute_ci(values)
        assert ci.ci_lower <= ci.mean <= ci.ci_upper

    def test_numpy_array_input(self):
        """Test that numpy arrays work."""
        values = np.array([10, 20, 30, 40, 50])
        ci = compute_ci(values)
        assert ci.mean == 30.0

    def test_confidence_level(self):
        """Test different confidence levels."""
        values = [100, 105, 95, 110, 90]

        ci_95 = compute_ci(values, confidence_level=0.95)
        ci_99 = compute_ci(values, confidence_level=0.99)

        # 99% CI should be wider than 95% CI
        assert (ci_99.ci_upper - ci_99.ci_lower) > (ci_95.ci_upper - ci_95.ci_lower)

    def test_larger_sample_narrower_ci(self):
        """Test that larger samples produce narrower CIs."""
        # Generate samples with same variance
        np.random.seed(42)
        small_sample = list(np.random.normal(100, 10, 5))
        large_sample = list(np.random.normal(100, 10, 50))

        ci_small = compute_ci(small_sample)
        ci_large = compute_ci(large_sample)

        small_width = ci_small.ci_upper - ci_small.ci_lower
        large_width = ci_large.ci_upper - ci_large.ci_lower

        # Larger sample should have narrower CI (on average)
        # Note: This isn't guaranteed for any single random draw,
        # but with seed 42 it should hold
        assert large_width < small_width
