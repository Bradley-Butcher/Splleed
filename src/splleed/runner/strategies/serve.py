"""Serve benchmark strategy - simulate realistic request arrival patterns."""

from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING

from .base import BenchmarkStrategy

if TYPE_CHECKING:
    from splleed.backends.base import Backend, GenerateRequest
    from splleed.config.base import ArrivalPattern, BenchmarkConfig, SamplingConfig
    from splleed.datasets.base import Dataset
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
        # Constant rate - evenly spaced
        interval = 1.0 / pattern.rate
        return [i * interval for i in range(num_requests)]

    elif pattern.type == "poisson":
        # Poisson process - exponential inter-arrival times
        times = [0.0]
        for _ in range(num_requests - 1):
            interval = rng.expovariate(pattern.rate)
            times.append(times[-1] + interval)
        return times

    elif pattern.type == "gamma":
        # Gamma distribution for burstiness control
        # burstiness=1.0 is equivalent to Poisson
        # burstiness<1.0 is more regular
        # burstiness>1.0 is more bursty
        times = [0.0]
        shape = pattern.burstiness
        # Scale chosen so mean rate matches pattern.rate
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

    Simulates realistic serving conditions with:
    - Request arrival patterns (Poisson, constant, gamma)
    - Concurrency limits
    - Multiple concurrency levels tested

    This measures latency under load, which is more representative
    of production serving scenarios.
    """

    def __init__(
        self,
        sampling: SamplingConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize serve strategy.

        Args:
            sampling: Sampling configuration for generation
            seed: Random seed for arrival times
        """
        self.sampling = sampling
        self.seed = seed

    async def run(
        self,
        executor: RequestExecutor,
        backend: Backend,
        dataset: Dataset,
        config: BenchmarkConfig,
    ) -> list[RequestResult]:
        """
        Run serve benchmark at a single concurrency level.

        Args:
            executor: Request executor for timing
            backend: Inference backend
            dataset: Dataset of prompts
            config: Benchmark configuration

        Returns:
            List of request results
        """
        from splleed.backends.base import GenerateRequest
        from splleed.config.base import ArrivalPattern

        # Get samples from dataset
        num_requests = config.runs
        samples = dataset.sample(num_requests)

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

        # Get arrival pattern
        pattern = config.arrival or ArrivalPattern()
        arrival_times = generate_arrival_times(num_requests, pattern, self.seed)

        # Concurrency limit (use first level for single run)
        concurrency = config.concurrency[0] if config.concurrency else 8
        semaphore = asyncio.Semaphore(concurrency)

        results: list[RequestResult | None] = [None] * num_requests
        start_time = time.perf_counter()

        async def run_at_time(idx: int, request: GenerateRequest, arrival: float) -> None:
            # Wait until scheduled arrival time
            elapsed = time.perf_counter() - start_time
            if arrival > elapsed:
                await asyncio.sleep(arrival - elapsed)

            # Execute with concurrency limit
            async with semaphore:
                results[idx] = await executor.execute(backend, request)

        # Launch all requests according to schedule
        tasks = [
            run_at_time(i, req, arrival)
            for i, (req, arrival) in enumerate(zip(requests, arrival_times, strict=True))
        ]
        await asyncio.gather(*tasks)

        return [r for r in results if r is not None]


class StartupStrategy(BenchmarkStrategy):
    """
    Startup benchmark strategy.

    Measures time to start the inference server.
    Only applicable to managed backends.
    """

    async def run(
        self,
        executor: RequestExecutor,
        backend: Backend,
        dataset: Dataset,
        config: BenchmarkConfig,
    ) -> list[RequestResult]:
        """
        Run startup benchmark.

        This strategy is special - it measures server startup time,
        not generation latency. Returns empty list as there are no
        generation results.
        """
        # Startup timing is handled by the orchestrator
        # This is a placeholder for the strategy interface
        return []
