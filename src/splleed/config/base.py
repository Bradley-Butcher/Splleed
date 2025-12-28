"""Base configuration classes for splleed."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SamplingConfig(BaseModel):
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

    # For serve mode
    arrival: ArrivalPattern | None = None
    slo: SLOConfig | None = None


class DatasetConfig(BaseModel):
    """Dataset configuration for benchmark prompts."""

    type: Literal["sharegpt", "random", "jsonl", "inline"] = "inline"
    path: Path | None = Field(default=None, description="Path to dataset file")
    prompts: list[str] | None = Field(default=None, description="Inline prompts")
    num_samples: int = Field(default=1000, description="Number of samples to use")
    input_len_range: tuple[int, int] | None = Field(
        default=None, description="Filter prompts by token length (min, max)"
    )
    output_len: int = Field(default=128, description="Expected output length for random dataset")


class OutputConfig(BaseModel):
    """Output configuration for benchmark results."""

    path: Path = Field(default=Path("results.json"), description="Output file path")
    format: Literal["json", "csv", "markdown"] = "json"
    include_raw: bool = Field(default=False, description="Include per-request raw data")


class CloudConfig(BaseModel):
    """SkyPilot cloud configuration."""

    enabled: bool = False
    gpu: str = Field(default="A100:1", description="GPU type and count")
    provider: str | None = Field(default=None, description="Cloud provider (aws, gcp, lambda, any)")
    spot: bool = Field(default=True, description="Use spot instances")
    region: str | None = None
    disk_size: int = Field(default=100, description="Disk size in GB")


class SplleedConfig(BaseModel):
    """Root configuration for splleed benchmarks."""

    # backend: BackendConfig  # Added after backends are defined to avoid circular import
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    cloud: CloudConfig | None = None
