"""Tests for metrics aggregation."""

import pytest

from splleed.config.base import SLOConfig
from splleed.metrics.aggregator import (
    LatencyStats,
    aggregate_results,
    compute_goodput,
)
from splleed.metrics.types import RequestResult, Token


def make_result(
    success: bool = True,
    ttft: float = 0.1,
    itl: list[float] | None = None,
    total_time: float = 1.0,
    num_tokens: int = 10,
    error: str | None = None,
) -> RequestResult:
    """Helper to create RequestResult for testing.

    Creates tokens positioned to produce the desired ttft and itl
    when computed by RequestResult properties.
    """
    start_time = 0.0

    if success:
        # If itl is provided, it determines token count
        if itl is not None:
            num_tokens = len(itl) + 1

        # Default ITL values if not provided
        actual_itl = itl if itl is not None else [0.05] * (num_tokens - 1)

        # Build tokens to produce desired ttft and itl
        tokens = []
        current_time = start_time + ttft  # First token at start_time + ttft
        for i in range(num_tokens):
            tokens.append(Token(text=f"t{i}", timestamp=current_time))
            if i < len(actual_itl):
                current_time += actual_itl[i]
    else:
        tokens = []

    return RequestResult(
        success=success,
        start_time=start_time,
        end_time=total_time,
        tokens=tokens,
        error=error,
    )


class TestLatencyStats:
    """Tests for LatencyStats computation."""

    def test_from_values_basic(self):
        """Test basic statistics computation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = LatencyStats.from_values(values)

        assert stats.mean == 3.0
        assert stats.median == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.p50 == 3.0

    def test_from_values_empty(self):
        """Test with empty values."""
        stats = LatencyStats.from_values([])

        assert stats.mean == 0.0
        assert stats.median == 0.0
        assert stats.p99 == 0.0

    def test_from_values_single(self):
        """Test with single value."""
        stats = LatencyStats.from_values([5.0])

        assert stats.mean == 5.0
        assert stats.median == 5.0
        assert stats.min == 5.0
        assert stats.max == 5.0

    def test_percentiles(self):
        """Test percentile calculations."""
        # 100 values from 1 to 100
        values = list(range(1, 101))
        stats = LatencyStats.from_values([float(v) for v in values])

        assert stats.p50 == pytest.approx(50.5, rel=0.1)
        assert stats.p95 == pytest.approx(95.05, rel=0.1)
        assert stats.p99 == pytest.approx(99.01, rel=0.1)


class TestComputeGoodput:
    """Tests for goodput calculation."""

    def test_all_meeting_slo(self):
        """Test when all requests meet SLO."""
        results = [
            make_result(ttft=0.05, total_time=0.5),  # 50ms TTFT, 500ms E2E
            make_result(ttft=0.08, total_time=0.4),  # 80ms TTFT, 400ms E2E
        ]
        slo = SLOConfig(ttft_ms=100, e2el_ms=1000)

        assert compute_goodput(results, slo) == 100.0

    def test_none_meeting_slo(self):
        """Test when no requests meet SLO."""
        results = [
            make_result(ttft=0.15, total_time=1.5),  # 150ms TTFT
            make_result(ttft=0.20, total_time=2.0),  # 200ms TTFT
        ]
        slo = SLOConfig(ttft_ms=100)

        assert compute_goodput(results, slo) == 0.0

    def test_partial_meeting_slo(self):
        """Test when some requests meet SLO."""
        results = [
            make_result(ttft=0.05),  # Meets 100ms SLO
            make_result(ttft=0.15),  # Exceeds 100ms SLO
            make_result(ttft=0.08),  # Meets 100ms SLO
            make_result(ttft=0.20),  # Exceeds 100ms SLO
        ]
        slo = SLOConfig(ttft_ms=100)

        assert compute_goodput(results, slo) == 50.0

    def test_multiple_slo_criteria(self):
        """Test with multiple SLO criteria (all must be met)."""
        results = [
            make_result(ttft=0.05, total_time=0.5),  # Meets both
            make_result(ttft=0.05, total_time=1.5),  # Meets TTFT, fails E2E
            make_result(ttft=0.15, total_time=0.5),  # Fails TTFT, meets E2E
        ]
        slo = SLOConfig(ttft_ms=100, e2el_ms=1000)

        # Only first request meets both criteria
        assert compute_goodput(results, slo) == pytest.approx(33.33, rel=0.1)

    def test_empty_results(self):
        """Test with no results."""
        slo = SLOConfig(ttft_ms=100)
        assert compute_goodput([], slo) == 0.0

    def test_all_failed_requests(self):
        """Test with only failed requests."""
        results = [
            make_result(success=False, error="timeout"),
            make_result(success=False, error="error"),
        ]
        slo = SLOConfig(ttft_ms=100)
        assert compute_goodput(results, slo) == 0.0

    def test_tpot_slo(self):
        """Test TPOT SLO checking."""
        # Create results with specific ITL patterns
        fast_result = make_result(itl=[0.01, 0.01, 0.01])  # 10ms TPOT
        slow_result = make_result(itl=[0.1, 0.1, 0.1])  # 100ms TPOT

        slo = SLOConfig(tpot_ms=50)

        assert compute_goodput([fast_result], slo) == 100.0
        assert compute_goodput([slow_result], slo) == 0.0
        assert compute_goodput([fast_result, slow_result], slo) == 50.0


class TestAggregateResults:
    """Tests for result aggregation."""

    def test_basic_aggregation(self):
        """Test basic result aggregation."""
        results = [
            make_result(ttft=0.1, total_time=1.0, num_tokens=10),
            make_result(ttft=0.15, total_time=1.2, num_tokens=12),
            make_result(ttft=0.12, total_time=0.9, num_tokens=8),
        ]

        agg = aggregate_results(
            results=results,
            concurrency=2,
            total_time=2.0,
        )

        assert agg.concurrency == 2
        assert agg.num_requests == 3
        assert agg.num_successful == 3
        assert agg.num_failed == 0
        assert agg.total_time_sec == 2.0

        # Check throughput
        total_tokens = 10 + 12 + 8  # 30
        assert agg.throughput_tokens_per_sec == total_tokens / 2.0
        assert agg.throughput_requests_per_sec == 3 / 2.0

        # Check TTFT is in milliseconds
        assert agg.ttft_mean_ms == pytest.approx(123.33, rel=0.1)  # (100+150+120)/3

    def test_with_failures(self):
        """Test aggregation with some failures."""
        results = [
            make_result(ttft=0.1, total_time=1.0, num_tokens=10),
            make_result(success=False, error="timeout"),
            make_result(ttft=0.15, total_time=1.2, num_tokens=12),
        ]

        agg = aggregate_results(
            results=results,
            concurrency=2,
            total_time=2.0,
        )

        assert agg.num_requests == 3
        assert agg.num_successful == 2
        assert agg.num_failed == 1

        # Throughput should only count successful requests
        assert agg.throughput_tokens_per_sec == 22 / 2.0  # 10 + 12 tokens

    def test_with_slo(self):
        """Test aggregation with SLO for goodput."""
        results = [
            make_result(ttft=0.05, total_time=0.5),  # Meets SLO
            make_result(ttft=0.15, total_time=0.8),  # Fails TTFT SLO
            make_result(ttft=0.08, total_time=1.5),  # Fails E2E SLO
            make_result(ttft=0.06, total_time=0.4),  # Meets SLO
        ]

        slo = SLOConfig(ttft_ms=100, e2el_ms=1000)

        agg = aggregate_results(
            results=results,
            concurrency=4,
            total_time=2.0,
            slo=slo,
        )

        assert agg.goodput_pct == 50.0  # 2 out of 4 meet both criteria

    def test_include_raw(self):
        """Test including raw results."""
        results = [make_result()]

        agg = aggregate_results(
            results=results,
            concurrency=1,
            total_time=1.0,
            include_raw=True,
        )

        assert agg.raw_results is not None
        assert len(agg.raw_results) == 1

    def test_exclude_raw(self):
        """Test excluding raw results."""
        results = [make_result()]

        agg = aggregate_results(
            results=results,
            concurrency=1,
            total_time=1.0,
            include_raw=False,
        )

        assert agg.raw_results is None

    def test_empty_results(self):
        """Test with empty results."""
        agg = aggregate_results(
            results=[],
            concurrency=1,
            total_time=1.0,
        )

        assert agg.num_requests == 0
        assert agg.num_successful == 0
        assert agg.throughput_tokens_per_sec == 0.0
        assert agg.ttft_mean_ms == 0.0

    def test_itl_aggregation(self):
        """Test that ITL is aggregated across all tokens."""
        results = [
            make_result(itl=[0.01, 0.02, 0.03]),  # 3 ITL values
            make_result(itl=[0.04, 0.05]),  # 2 ITL values
        ]

        agg = aggregate_results(
            results=results,
            concurrency=1,
            total_time=1.0,
        )

        # Mean of [0.01, 0.02, 0.03, 0.04, 0.05] = 0.03 = 30ms
        assert agg.itl_mean_ms == pytest.approx(30.0, rel=0.1)


class TestRequestResultProperties:
    """Tests for RequestResult computed properties."""

    def test_total_time(self):
        """Test total_time property."""
        result = RequestResult(
            success=True,
            start_time=1.0,
            end_time=2.5,
            tokens=[],
        )
        assert result.total_time == 1.5

    def test_output_tokens(self):
        """Test output_tokens property."""
        result = make_result(num_tokens=15)
        assert result.output_tokens == 15

    def test_output_text(self):
        """Test output_text property."""
        result = make_result(num_tokens=3)
        assert result.output_text == "t0t1t2"

    def test_tpot(self):
        """Test tpot property (mean of ITLs)."""
        result = make_result(itl=[0.1, 0.2, 0.3])
        assert result.tpot == pytest.approx(0.2, rel=0.01)

    def test_tpot_empty_itl(self):
        """Test tpot with no ITL values."""
        result = make_result(num_tokens=1, itl=[])
        assert result.tpot is None

    def test_tokens_per_sec(self):
        """Test tokens_per_sec property."""
        result = make_result(num_tokens=10, total_time=2.0)
        assert result.tokens_per_sec == 5.0

    def test_tokens_per_sec_no_tokens(self):
        """Test tokens_per_sec with no tokens."""
        result = make_result(success=False)
        assert result.tokens_per_sec is None

    def test_ttft_computed(self):
        """Test ttft is computed from token timestamps."""
        result = make_result(ttft=0.05)
        assert result.ttft == pytest.approx(0.05, rel=0.01)

    def test_ttft_no_tokens(self):
        """Test ttft is None when no tokens."""
        result = make_result(success=False)
        assert result.ttft is None

    def test_itl_computed(self):
        """Test itl is computed from token timestamps."""
        result = make_result(itl=[0.01, 0.02, 0.03])
        assert len(result.itl) == 3
        assert result.itl[0] == pytest.approx(0.01, rel=0.01)
        assert result.itl[1] == pytest.approx(0.02, rel=0.01)
        assert result.itl[2] == pytest.approx(0.03, rel=0.01)

    def test_itl_single_token(self):
        """Test itl is empty with single token."""
        result = make_result(num_tokens=1, itl=[])
        assert result.itl == []
