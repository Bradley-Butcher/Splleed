"""Serve benchmark strategy - simulate realistic request arrival patterns."""

from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING

from .base import BenchmarkStrategy

if TYPE_CHECKING:
    from splleed.api import Benchmark
    from splleed.backends.base import Backend, GenerateRequest
    from splleed.config.base import ArrivalPattern, SamplingParams
    from splleed.metrics.types import RequestResult
    from splleed.runner.executor import RequestExecutor


def generate_arrival_times(
    num_requests: int,
    pattern: ArrivalPattern,
    seed: int | None = None,
) -> list[float]:
    """
    Generate request arrival times based on pattern.

    Args:
        num_requests: Number of requests to generate times for
        pattern: Arrival pattern configuration
        seed: Random seed for reproducibility

    Returns:
        List of relative arrival times in seconds (starting at 0)
    """
    rng = random.Random(seed)

    if pattern.type == "constant":
        interval = 1.0 / pattern.rate
        return [i * interval for i in range(num_requests)]

    elif pattern.type == "poisson":
        times = [0.0]
        for _ in range(num_requests - 1):
            interval = rng.expovariate(pattern.rate)
            times.append(times[-1] + interval)
        return times

    elif pattern.type == "gamma":
        times = [0.0]
        shape = pattern.burstiness
        scale = 1.0 / (pattern.rate * shape)
        for _ in range(num_requests - 1):
            interval = rng.gammavariate(shape, scale)
            times.append(times[-1] + interval)
        return times

    else:
        raise ValueError(f"Unknown arrival pattern type: {pattern.type}")


class ServeStrategy(BenchmarkStrategy):
    """
    Serve benchmark strategy.

    Simulates realistic serving conditions with request arrival patterns.
    """

    def __init__(
        self,
        sampling: SamplingParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.sampling = sampling
        self.seed = seed

    async def run(
        self,
        executor: RequestExecutor,
        backend: Backend,
        prompts: list[str],
        config: Benchmark,
    ) -> list[RequestResult]:
        from splleed.backends.base import GenerateRequest
        from splleed.config.base import ArrivalPattern

        max_tokens = self.sampling.max_tokens if self.sampling else 128
        min_tokens = self.sampling.min_tokens if self.sampling else None
        temperature = self.sampling.temperature if self.sampling else 0.0

        requests = [
            GenerateRequest(
                prompt=p,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                temperature=temperature,
            )
            for p in prompts
        ]

        num_requests = len(requests)
        pattern = config.arrival or ArrivalPattern()
        arrival_times = generate_arrival_times(num_requests, pattern, self.seed)

        concurrency = config.concurrency[0] if config.concurrency else 8
        semaphore = asyncio.Semaphore(concurrency)

        results: list[RequestResult | None] = [None] * num_requests
        start_time = time.perf_counter()

        async def run_at_time(idx: int, request: GenerateRequest, arrival: float) -> None:
            elapsed = time.perf_counter() - start_time
            if arrival > elapsed:
                await asyncio.sleep(arrival - elapsed)
            async with semaphore:
                results[idx] = await executor.execute(backend, request)

        tasks = [
            run_at_time(i, req, arrival)
            for i, (req, arrival) in enumerate(zip(requests, arrival_times, strict=True))
        ]
        await asyncio.gather(*tasks)

        return [r for r in results if r is not None]


class StartupStrategy(BenchmarkStrategy):
    """Startup benchmark strategy - measures server startup time."""

    async def run(
        self,
        executor: RequestExecutor,
        backend: Backend,
        prompts: list[str],
        config: Benchmark,
    ) -> list[RequestResult]:
        # Startup timing is handled by the orchestrator
        return []
