"""Core metric types for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

    from splleed.environment import EnvironmentInfo
    from splleed.stats import ConfidenceInterval


@dataclass
class Token:
    """A single generated token with timing information."""

    text: str
    timestamp: float  # time.perf_counter() when received


@dataclass
class RequestResult:
    """Result of a single generation request."""

    success: bool
    start_time: float
    end_time: float

    # Populated on success
    tokens: list[Token] = field(default_factory=list)
    ttft: float | None = None  # Time to first token (seconds)
    itl: list[float] = field(default_factory=list)  # Inter-token latencies (seconds)

    # Populated on failure
    error: str | None = None

    @property
    def total_time(self) -> float:
        """Total request time in seconds."""
        return self.end_time - self.start_time

    @property
    def output_tokens(self) -> int:
        """Number of tokens generated."""
        return len(self.tokens)

    @property
    def output_text(self) -> str:
        """Concatenated output text."""
        return "".join(t.text for t in self.tokens)

    @property
    def tpot(self) -> float | None:
        """Time per output token (mean of ITLs)."""
        if not self.itl:
            return None
        return sum(self.itl) / len(self.itl)

    @property
    def tokens_per_sec(self) -> float | None:
        """Tokens per second for this request."""
        if self.output_tokens == 0 or self.total_time == 0:
            return None
        return self.output_tokens / self.total_time


@dataclass
class ConcurrencyResult:
    """Aggregated results for a single concurrency level."""

    concurrency: int
    num_requests: int
    num_successful: int
    num_failed: int

    # Throughput metrics
    total_time_sec: float
    throughput_requests_per_sec: float
    throughput_tokens_per_sec: float

    # Latency metrics (in milliseconds)
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_mean_ms: float

    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    itl_mean_ms: float

    tpot_mean_ms: float

    e2el_p50_ms: float  # End-to-end latency
    e2el_p95_ms: float
    e2el_p99_ms: float
    e2el_mean_ms: float

    # Goodput (if SLO configured)
    goodput_pct: float | None = None

    # Raw results (if include_raw=True)
    raw_results: list[RequestResult] | None = None


@dataclass
class TrialResult:
    """Results from a single benchmark trial (for multi-trial runs)."""

    trial_index: int
    concurrency_results: list[ConcurrencyResult]


@dataclass
class ConcurrencyResultWithCI:
    """
    Aggregated results with confidence intervals (when trials > 1).

    All latency/throughput values are ConfidenceInterval objects containing
    mean, ci_lower, ci_upper, and std.
    """

    concurrency: int
    num_requests: int  # Total across all trials
    num_successful: int
    num_failed: int

    # Throughput metrics with CI
    throughput_tokens_per_sec: ConfidenceInterval
    throughput_requests_per_sec: ConfidenceInterval

    # Latency metrics with CI (in milliseconds)
    ttft_p50_ms: ConfidenceInterval
    ttft_p95_ms: ConfidenceInterval
    ttft_p99_ms: ConfidenceInterval
    ttft_mean_ms: ConfidenceInterval

    itl_p50_ms: ConfidenceInterval
    itl_p95_ms: ConfidenceInterval
    itl_p99_ms: ConfidenceInterval
    itl_mean_ms: ConfidenceInterval

    tpot_mean_ms: ConfidenceInterval

    e2el_p50_ms: ConfidenceInterval
    e2el_p95_ms: ConfidenceInterval
    e2el_p99_ms: ConfidenceInterval
    e2el_mean_ms: ConfidenceInterval

    # Goodput with CI (if SLO configured)
    goodput_pct: ConfidenceInterval | None = None


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    # Metadata
    engine: str
    model: str
    timestamp: str
    gpu: str | None  # Keep for backwards compatibility
    config: dict[str, Any]

    # Results per concurrency level (first trial or single trial)
    results: list[ConcurrencyResult]

    # Extended fields for Phase 1
    environment: EnvironmentInfo | None = None
    n_trials: int = 1

    # Multi-trial data (populated when trials > 1)
    trial_results: list[TrialResult] | None = None
    aggregated_results: list[ConcurrencyResultWithCI] | None = None

    def print(self, console: Console | None = None) -> None:
        """Print results as a Rich table."""
        from splleed.reporters.console import print_results

        print_results(self, console)

    def to_json(self, indent: int = 2) -> str:
        """Serialize results to JSON string."""
        from splleed.reporters.json import to_json

        return to_json(self, indent)

    def to_csv(self) -> str:
        """Serialize results to CSV string."""
        from splleed.reporters.csv import to_csv

        return to_csv(self)

    def save(self, path: Path | str) -> None:
        """
        Save results to file.

        Format is inferred from file extension:
        - .json: JSON format
        - .csv: CSV format
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".csv":
            path.write_text(self.to_csv())
        else:
            path.write_text(self.to_json())
