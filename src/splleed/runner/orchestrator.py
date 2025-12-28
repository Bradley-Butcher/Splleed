"""Benchmark orchestrator - coordinates the full benchmark run."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from splleed.datasets import get_dataset
from splleed.metrics import aggregate_results
from splleed.metrics.types import BenchmarkResults, ConcurrencyResult
from splleed.runner.executor import RequestExecutor
from splleed.runner.strategies import (
    LatencyStrategy,
    ServeStrategy,
    StartupStrategy,
    ThroughputStrategy,
)

if TYPE_CHECKING:
    from splleed.backends.base import Backend
    from splleed.config.loader import FullConfig

logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """
    Orchestrates the full benchmark workflow.

    Coordinates:
    - Backend lifecycle (start/connect/shutdown)
    - Warmup runs
    - Running benchmarks at different concurrency levels
    - Metrics aggregation
    - Result reporting
    """

    def __init__(self, config: FullConfig) -> None:
        """
        Initialize orchestrator.

        Args:
            config: Full benchmark configuration
        """
        self.config = config
        self.executor = RequestExecutor()

    def _get_strategy(self):
        """Get the appropriate benchmark strategy."""
        mode = self.config.benchmark.mode

        if mode == "throughput":
            return ThroughputStrategy(self.config.sampling)
        elif mode == "latency":
            return LatencyStrategy(self.config.sampling)
        elif mode == "serve":
            return ServeStrategy(self.config.sampling)
        elif mode == "startup":
            return StartupStrategy()
        else:
            raise ValueError(f"Unknown benchmark mode: {mode}")

    async def _run_warmup(self, backend: Backend, strategy, dataset) -> None:
        """Run warmup iterations."""
        warmup_count = self.config.benchmark.warmup
        if warmup_count <= 0:
            return

        logger.info(f"Running {warmup_count} warmup iterations...")

        for i in range(warmup_count):
            logger.debug(f"Warmup iteration {i + 1}/{warmup_count}")
            await strategy.run(
                self.executor,
                backend,
                dataset,
                self.config.benchmark,
            )

        logger.info("Warmup complete")

    async def run(self, backend: Backend) -> BenchmarkResults:
        """
        Run the full benchmark.

        Args:
            backend: Initialized inference backend

        Returns:
            Complete benchmark results
        """
        # Load dataset
        dataset = get_dataset(self.config.dataset)
        logger.info(f"Loaded dataset with {len(dataset)} samples")

        # Get strategy
        strategy = self._get_strategy()
        logger.info(f"Using {strategy.__class__.__name__} strategy")

        # Run warmup
        await self._run_warmup(backend, strategy, dataset)

        # Run benchmarks at each concurrency level
        concurrency_results: list[ConcurrencyResult] = []

        for concurrency in self.config.benchmark.concurrency:
            logger.info(f"Running benchmark at concurrency={concurrency}")

            # Create modified config for this concurrency level
            bench_config = self.config.benchmark.model_copy()
            bench_config.concurrency = [concurrency]

            # Time the benchmark run
            start_time = time.perf_counter()

            results = await strategy.run(
                self.executor,
                backend,
                dataset,
                bench_config,
            )

            total_time = time.perf_counter() - start_time

            # Aggregate results
            agg = aggregate_results(
                results=results,
                concurrency=concurrency,
                total_time=total_time,
                slo=self.config.benchmark.slo,
                include_raw=self.config.output.include_raw,
            )
            concurrency_results.append(agg)

            logger.info(
                f"  Completed: {agg.num_successful}/{agg.num_requests} requests, "
                f"{agg.throughput_tokens_per_sec:.1f} tokens/s"
            )

        # Build final results
        return BenchmarkResults(
            engine=self.config.backend.type,
            model=getattr(self.config.backend, "model", None) or "unknown",
            timestamp=datetime.now(UTC).isoformat(),
            gpu=self._get_gpu_info(),
            config=self.config.model_dump(),
            results=concurrency_results,
        )

    def _get_gpu_info(self) -> str | None:
        """Try to get GPU information."""
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass
        return None


async def run_benchmark(config: FullConfig) -> BenchmarkResults:
    """
    Run a complete benchmark from configuration.

    This is the main entry point for running benchmarks.

    Args:
        config: Full benchmark configuration

    Returns:
        Complete benchmark results
    """
    from splleed.backends import get_backend

    # Create backend
    backend = get_backend(config.backend)

    # Initialize backend (connect or start)
    if hasattr(backend, "initialize"):
        await backend.initialize()
    elif config.backend.endpoint:
        await backend.connect(config.backend.endpoint)
    else:
        await backend.start()

    try:
        # Run benchmark
        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(backend)
        return results

    finally:
        # Cleanup
        if hasattr(backend, "shutdown"):
            await backend.shutdown()
