"""Throughput and latency benchmark strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BenchmarkStrategy

if TYPE_CHECKING:
    from splleed.backends.base import Backend
    from splleed.config.base import BenchmarkConfig, SamplingParams
    from splleed.metrics.types import RequestResult
    from splleed.runner.executor import RequestExecutor


class ThroughputStrategy(BenchmarkStrategy):
    """
    Throughput benchmark strategy.

    Sends all requests as fast as possible with limited concurrency.
    Measures maximum achievable throughput.
    """

    def __init__(self, sampling: SamplingParams | None = None) -> None:
        self.sampling = sampling

    async def run(
        self,
        executor: RequestExecutor,
        backend: Backend,
        prompts: list[str],
        config: BenchmarkConfig,
    ) -> list[RequestResult]:
        from splleed.backends.base import GenerateRequest
        from splleed.runner.executor import execute_concurrent

        max_tokens = self.sampling.max_tokens if self.sampling else 128
        temperature = self.sampling.temperature if self.sampling else 0.0

        requests = [
            GenerateRequest(prompt=p, max_tokens=max_tokens, temperature=temperature)
            for p in prompts
        ]

        concurrency = max(config.concurrency) if config.concurrency else 1
        return await execute_concurrent(executor, backend, requests, concurrency)


class LatencyStrategy(BenchmarkStrategy):
    """
    Latency benchmark strategy.

    Sends requests one at a time (concurrency=1) to measure
    best-case latency without queuing effects.
    """

    def __init__(self, sampling: SamplingParams | None = None) -> None:
        self.sampling = sampling

    async def run(
        self,
        executor: RequestExecutor,
        backend: Backend,
        prompts: list[str],
        config: BenchmarkConfig,
    ) -> list[RequestResult]:
        from splleed.backends.base import GenerateRequest

        max_tokens = self.sampling.max_tokens if self.sampling else 128
        temperature = self.sampling.temperature if self.sampling else 0.0

        results: list[RequestResult] = []

        for prompt in prompts:
            request = GenerateRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = await executor.execute(backend, request)
            results.append(result)

        return results
