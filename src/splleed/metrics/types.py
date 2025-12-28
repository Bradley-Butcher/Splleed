"""Core metric types for benchmark results."""

from dataclasses import dataclass, field
from typing import Any


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
class BenchmarkResults:
    """Complete benchmark results."""

    # Metadata
    engine: str
    model: str
    timestamp: str
    gpu: str | None
    config: dict[str, Any]

    # Results per concurrency level
    results: list[ConcurrencyResult]
