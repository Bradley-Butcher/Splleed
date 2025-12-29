"""Metrics aggregation and statistics computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from splleed.stats import ConfidenceInterval, compute_ci

if TYPE_CHECKING:
    from splleed.config.base import SLOConfig

from .types import ConcurrencyResult, ConcurrencyResultWithCI, RequestResult, TrialResult


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def from_values(cls, values: list[float]) -> LatencyStats:
        """Compute stats from a list of values."""
        if not values:
            return cls(
                mean=0.0,
                median=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                p50=0.0,
                p95=0.0,
                p99=0.0,
            )

        arr = np.array(values)
        return cls(
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
        )


def compute_goodput(
    results: list[RequestResult],
    slo: SLOConfig,
) -> float:
    """
    Compute goodput percentage - fraction of requests meeting SLO.

    Args:
        results: List of successful request results
        slo: SLO configuration with thresholds

    Returns:
        Percentage of requests meeting all SLO thresholds (0-100)
    """
    if not results:
        return 0.0

    successful = [r for r in results if r.success]
    if not successful:
        return 0.0

    meeting_slo = 0

    for result in successful:
        meets = True

        # Check TTFT SLO
        if slo.ttft_ms is not None and result.ttft is not None:
            ttft_ms = result.ttft * 1000
            if ttft_ms > slo.ttft_ms:
                meets = False

        # Check TPOT SLO
        if slo.tpot_ms is not None and result.tpot is not None:
            tpot_ms = result.tpot * 1000
            if tpot_ms > slo.tpot_ms:
                meets = False

        # Check E2E latency SLO
        if slo.e2el_ms is not None:
            e2el_ms = result.total_time * 1000
            if e2el_ms > slo.e2el_ms:
                meets = False

        if meets:
            meeting_slo += 1

    return (meeting_slo / len(successful)) * 100


def aggregate_results(
    results: list[RequestResult],
    concurrency: int,
    total_time: float,
    slo: SLOConfig | None = None,
    include_raw: bool = False,
) -> ConcurrencyResult:
    """
    Aggregate request results into summary statistics.

    Args:
        results: List of request results
        concurrency: Concurrency level used
        total_time: Total benchmark time in seconds
        slo: Optional SLO configuration for goodput calculation
        include_raw: Whether to include raw results

    Returns:
        Aggregated ConcurrencyResult
    """
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # Collect latency values (in seconds, will convert to ms at the end)
    ttft_values = [r.ttft for r in successful if r.ttft is not None]
    e2el_values = [r.total_time for r in successful]

    # ITL - flatten all inter-token latencies
    itl_values: list[float] = []
    for r in successful:
        itl_values.extend(r.itl)

    # TPOT - mean ITL per request
    tpot_values = [r.tpot for r in successful if r.tpot is not None]

    # Compute stats
    ttft_stats = LatencyStats.from_values(ttft_values)
    itl_stats = LatencyStats.from_values(itl_values)
    e2el_stats = LatencyStats.from_values(e2el_values)
    tpot_stats = LatencyStats.from_values(tpot_values)

    # Compute throughput
    total_output_tokens = sum(r.output_tokens for r in successful)
    tokens_per_sec = total_output_tokens / total_time if total_time > 0 else 0.0
    requests_per_sec = len(successful) / total_time if total_time > 0 else 0.0

    # Compute goodput if SLO provided
    goodput_pct = None
    if slo is not None:
        goodput_pct = compute_goodput(successful, slo)

    # Convert seconds to milliseconds for output
    def to_ms(val: float) -> float:
        return val * 1000

    return ConcurrencyResult(
        concurrency=concurrency,
        num_requests=len(results),
        num_successful=len(successful),
        num_failed=len(failed),
        total_time_sec=total_time,
        throughput_requests_per_sec=requests_per_sec,
        throughput_tokens_per_sec=tokens_per_sec,
        ttft_p50_ms=to_ms(ttft_stats.p50),
        ttft_p95_ms=to_ms(ttft_stats.p95),
        ttft_p99_ms=to_ms(ttft_stats.p99),
        ttft_mean_ms=to_ms(ttft_stats.mean),
        itl_p50_ms=to_ms(itl_stats.p50),
        itl_p95_ms=to_ms(itl_stats.p95),
        itl_p99_ms=to_ms(itl_stats.p99),
        itl_mean_ms=to_ms(itl_stats.mean),
        tpot_mean_ms=to_ms(tpot_stats.mean),
        e2el_p50_ms=to_ms(e2el_stats.p50),
        e2el_p95_ms=to_ms(e2el_stats.p95),
        e2el_p99_ms=to_ms(e2el_stats.p99),
        e2el_mean_ms=to_ms(e2el_stats.mean),
        goodput_pct=goodput_pct,
        raw_results=results if include_raw else None,
    )


def aggregate_trials(
    trial_results: list[TrialResult],
    confidence_level: float = 0.95,
) -> list[ConcurrencyResultWithCI]:
    """
    Aggregate results across multiple trials into confidence intervals.

    Args:
        trial_results: Results from multiple independent trials
        confidence_level: Confidence level for CI computation (e.g., 0.95 for 95% CI)

    Returns:
        List of aggregated results with confidence intervals for each concurrency level
    """
    if not trial_results:
        return []

    concurrency_levels = [cr.concurrency for cr in trial_results[0].concurrency_results]
    aggregated: list[ConcurrencyResultWithCI] = []

    for idx, concurrency in enumerate(concurrency_levels):
        trial_data = [tr.concurrency_results[idx] for tr in trial_results]

        def ci(field: str, data: list[ConcurrencyResult] = trial_data) -> ConfidenceInterval:
            values = [getattr(td, field) for td in data]
            return compute_ci(values, confidence_level)

        # Handle goodput specially - filter None values
        goodput_values = [td.goodput_pct for td in trial_data if td.goodput_pct is not None]
        goodput = compute_ci(goodput_values, confidence_level) if goodput_values else None

        aggregated.append(
            ConcurrencyResultWithCI(
                concurrency=concurrency,
                # Counts - summed across trials
                num_requests=sum(td.num_requests for td in trial_data),
                num_successful=sum(td.num_successful for td in trial_data),
                num_failed=sum(td.num_failed for td in trial_data),
                # Throughput
                throughput_tokens_per_sec=ci("throughput_tokens_per_sec"),
                throughput_requests_per_sec=ci("throughput_requests_per_sec"),
                # TTFT
                ttft_p50_ms=ci("ttft_p50_ms"),
                ttft_p95_ms=ci("ttft_p95_ms"),
                ttft_p99_ms=ci("ttft_p99_ms"),
                ttft_mean_ms=ci("ttft_mean_ms"),
                # ITL
                itl_p50_ms=ci("itl_p50_ms"),
                itl_p95_ms=ci("itl_p95_ms"),
                itl_p99_ms=ci("itl_p99_ms"),
                itl_mean_ms=ci("itl_mean_ms"),
                # TPOT
                tpot_mean_ms=ci("tpot_mean_ms"),
                # E2E Latency
                e2el_p50_ms=ci("e2el_p50_ms"),
                e2el_p95_ms=ci("e2el_p95_ms"),
                e2el_p99_ms=ci("e2el_p99_ms"),
                e2el_mean_ms=ci("e2el_mean_ms"),
                # Goodput (special handling for None)
                goodput_pct=goodput,
            )
        )

    return aggregated
