"""Throughput benchmark strategy - send all requests at once."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .base import BenchmarkStrategy

if TYPE_CHECKING:
    from splleed.backends.base import Backend, GenerateRequest
    from splleed.config.base import BenchmarkConfig, SamplingConfig
    from splleed.datasets.base import Dataset
    from splleed.metrics.types import RequestResult
    from splleed.runner.executor import RequestExecutor


class ThroughputStrategy(BenchmarkStrategy):
    """
    Throughput benchmark strategy.

    Sends all requests as fast as possible with limited concurrency.
    Measures maximum achievable throughput.
    """

    def __init__(self, sampling: SamplingConfig | None = None) -> None:
        """
        Initialize throughput strategy.

        Args:
            sampling: Sampling configuration for generation
        """
        self.sampling = sampling

    async def run(
        self,
        executor: RequestExecutor,
        backend: Backend,
        dataset: Dataset,
        config: BenchmarkConfig,
    ) -> list[RequestResult]:
        """
        Run throughput benchmark.

        Args:
            executor: Request executor for timing
            backend: Inference backend
            dataset: Dataset of prompts
            config: Benchmark configuration

        Returns:
            List of request results
        """
        from splleed.backends.base import GenerateRequest

        # Get samples from dataset
        samples = dataset.sample(config.runs * 10)  # Get enough samples for all runs

        # Build generation requests
        max_tokens = self.sampling.max_tokens if self.sampling else 128
        temperature = self.sampling.temperature if self.sampling else 0.0

        requests = [
            GenerateRequest(
                prompt=sample.prompt,
                max_tokens=sample.expected_output_len or max_tokens,
                temperature=temperature,
            )
            for sample in samples
        ]

        # Run with concurrency limit
        # For throughput, we use the highest concurrency level
        concurrency = max(config.concurrency) if config.concurrency else 1

        semaphore = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []

        async def run_request(request: GenerateRequest) -> RequestResult:
            async with semaphore:
                return await executor.execute(backend, request)

        # Execute all requests concurrently
        tasks = [run_request(req) for req in requests]
        results = await asyncio.gather(*tasks)

        return list(results)


class LatencyStrategy(BenchmarkStrategy):
    """
    Latency benchmark strategy.

    Sends requests one at a time (concurrency=1) to measure
    best-case latency without queuing effects.
    """

    def __init__(self, sampling: SamplingConfig | None = None) -> None:
        """
        Initialize latency strategy.

        Args:
            sampling: Sampling configuration for generation
        """
        self.sampling = sampling

    async def run(
        self,
        executor: RequestExecutor,
        backend: Backend,
        dataset: Dataset,
        config: BenchmarkConfig,
    ) -> list[RequestResult]:
        """
        Run latency benchmark.

        Args:
            executor: Request executor for timing
            backend: Inference backend
            dataset: Dataset of prompts
            config: Benchmark configuration

        Returns:
            List of request results
        """
        from splleed.backends.base import GenerateRequest

        # Get samples from dataset
        samples = dataset.sample(config.runs)

        # Build generation requests
        max_tokens = self.sampling.max_tokens if self.sampling else 128
        temperature = self.sampling.temperature if self.sampling else 0.0

        results: list[RequestResult] = []

        for sample in samples:
            request = GenerateRequest(
                prompt=sample.prompt,
                max_tokens=sample.expected_output_len or max_tokens,
                temperature=temperature,
            )
            result = await executor.execute(backend, request)
            results.append(result)

        return results
