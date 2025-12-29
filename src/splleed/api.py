"""Public Python API for splleed benchmarks."""

from __future__ import annotations

import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from splleed.backends import BackendConfig, get_backend
from splleed.config.base import ArrivalPattern, SamplingParams, SLOConfig
from splleed.loaders import load_hf_dataset
from splleed.metrics.types import BenchmarkResults
from splleed.runner.orchestrator import BenchmarkOrchestrator

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from datasets import Dataset  # type: ignore[import-not-found]

    from splleed.backends.base import Backend


class Benchmark(BaseModel):
    """
    LLM inference benchmark configuration.

    Example:
        >>> b = Benchmark(
        ...     backend=VLLMConfig(model="Qwen/Qwen2.5-0.5B-Instruct"),
        ...     prompts=["What is 2+2?", "Write a poem."],
        ...     concurrency=[1, 2, 4],
        ... )
        >>> results = await b.run()
    """

    # Backend
    backend: BackendConfig

    # Data sources (one required)
    prompts: list[str] | None = None
    dataset: str | None = None
    prompt_column: str = "prompt"
    num_samples: Annotated[int | None, Field(gt=0)] = None

    # Benchmark settings
    mode: Literal["latency", "throughput", "serve"] = "throughput"
    concurrency: Annotated[list[int], Field(min_length=1)] = [1]
    warmup: Annotated[int, Field(ge=0)] = 2
    runs: Annotated[int, Field(ge=1)] = 10
    trials: Annotated[int, Field(ge=1)] = 1
    confidence_level: Annotated[float, Field(gt=0, lt=1)] = 0.95

    # Serve mode
    arrival_rate: Annotated[float | None, Field(gt=0)] = None
    arrival_pattern: Literal["poisson", "gamma", "constant"] = "poisson"

    # Sampling
    sampling: SamplingParams = Field(default_factory=SamplingParams)

    # SLO thresholds
    slo: SLOConfig | None = None

    # Output
    output_file: Path | None = None

    @model_validator(mode="after")
    def validate_data_source(self) -> Benchmark:
        """Ensure exactly one data source is provided."""
        if self.prompts is None and self.dataset is None:
            raise ValueError("Either 'prompts' or 'dataset' must be provided")
        if self.prompts is not None and self.dataset is not None:
            raise ValueError("Cannot specify both 'prompts' and 'dataset'")
        return self

    @model_validator(mode="after")
    def validate_latency_mode(self) -> Benchmark:
        """Warn if latency mode is used with concurrency > 1."""
        if self.mode == "latency" and (len(self.concurrency) > 1 or max(self.concurrency) > 1):
            warnings.warn(
                "mode='latency' runs requests sequentially and ignores the concurrency "
                "parameter. Use mode='throughput' or mode='serve' to test different "
                "concurrency levels.",
                UserWarning,
                stacklevel=2,
            )
        return self

    async def run(
        self,
        processor: Callable[[Dataset], Dataset] | None = None,
        *,
        _backend: Backend | None = None,
    ) -> BenchmarkResults:
        """
        Execute the benchmark.

        Args:
            processor: Optional function to process HF dataset before use.
            _backend: Injected backend instance (for testing only).

        Returns:
            BenchmarkResults with metrics and statistics.
        """
        prompts = self._resolve_prompts(processor)

        async with self._backend_session(_backend) as backend:
            results = await self._execute(backend, prompts)

        if self.output_file:
            results.save(self.output_file)

        return results

    # --- Private helpers ---

    def _resolve_prompts(
        self,
        processor: Callable[[Dataset], Dataset] | None,
    ) -> list[str]:
        """Get prompts from direct list or HF dataset."""
        if self.prompts is not None:
            return self.prompts

        assert self.dataset is not None
        return load_hf_dataset(
            dataset_name=self.dataset,
            processor=processor,
            prompt_column=self.prompt_column,
            num_samples=self.num_samples,
        )

    @property
    def arrival(self) -> ArrivalPattern | None:
        """Build ArrivalPattern from flat fields for serve mode."""
        if self.mode != "serve":
            return None
        if self.arrival_rate is not None:
            return ArrivalPattern(type=self.arrival_pattern, rate=self.arrival_rate)
        return ArrivalPattern(type=self.arrival_pattern)

    @asynccontextmanager
    async def _backend_session(
        self,
        injected: Backend | None = None,
    ) -> AsyncIterator[Backend]:
        """Context manager for backend lifecycle."""
        if injected is not None:
            yield injected
            return

        backend = get_backend(self.backend)

        if self.backend.endpoint:
            await backend.connect(self.backend.endpoint)
        else:
            await backend.start()

        try:
            yield backend
        finally:
            await backend.shutdown()

    async def _execute(self, backend: Backend, prompts: list[str]) -> BenchmarkResults:
        """Run the benchmark orchestrator."""
        orchestrator = BenchmarkOrchestrator(self, prompts)
        return await orchestrator.run(backend)
