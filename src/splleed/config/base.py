"""Configuration classes for splleed."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SamplingParams(BaseModel):
    """Per-request generation parameters."""

    max_tokens: int = 128
    min_tokens: int | None = None
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
