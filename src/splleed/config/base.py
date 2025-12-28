"""Configuration classes for splleed."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SamplingParams(BaseModel):
    """Per-request generation parameters."""

    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class ArrivalPattern(BaseModel):
    """Request arrival distribution for serve mode."""

    type: Literal["poisson", "gamma", "constant"] = "poisson"
    rate: float = Field(default=10.0, description="Requests per second")
    burstiness: float = Field(default=1.0, description="Gamma shape parameter (1.0 = Poisson)")


class SLOConfig(BaseModel):
    """Service Level Objective thresholds (in milliseconds)."""

    ttft_ms: float | None = Field(default=None, description="Max acceptable TTFT")
    tpot_ms: float | None = Field(default=None, description="Max acceptable time per output token")
    e2el_ms: float | None = Field(default=None, description="Max acceptable end-to-end latency")


class BenchmarkConfig(BaseModel):
    """Benchmark execution configuration."""

    mode: Literal["throughput", "latency", "serve", "startup"] = "serve"
    concurrency: list[int] = Field(default=[1, 2, 4, 8], description="Concurrency levels to test")
    warmup: int = Field(default=2, description="Number of warmup runs")
    runs: int = Field(default=3, description="Number of benchmark runs per concurrency level")
    trials: int = Field(
        default=1,
        ge=1,
        description="Number of independent benchmark trials for confidence intervals",
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for CI computation (0.95 = 95% CI)",
    )

    # For serve mode
    arrival: ArrivalPattern | None = None
    slo: SLOConfig | None = None
