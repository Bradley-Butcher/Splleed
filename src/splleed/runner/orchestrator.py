"""Benchmark orchestrator - coordinates the full benchmark run."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from splleed.environment import capture_environment, format_gpu_info
from splleed.metrics import aggregate_results, aggregate_trials
from splleed.metrics.types import (
    BenchmarkResults,
    ConcurrencyResult,
    TrialResult,
)
from splleed.runner.executor import RequestExecutor
from splleed.runner.strategies import (
    LatencyStrategy,
    ServeStrategy,
    StartupStrategy,
    ThroughputStrategy,
)

if TYPE_CHECKING:
    from splleed.api import Benchmark
    from splleed.backends.base import Backend

logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """
    Orchestrates the full benchmark workflow.

    Coordinates:
    - Warmup runs
    - Multiple independent trials (for statistical rigor)
    - Running benchmarks at different concurrency levels
    - Metrics aggregation with confidence intervals
    """

    def __init__(
        self,
        benchmark: Benchmark,
        prompts: list[str],
        *,
        include_raw: bool = False,
    ) -> None:
        self.benchmark = benchmark
        self.prompts = prompts
        self.include_raw = include_raw
        self.executor = RequestExecutor()

    def _get_strategy(self):
        """Get the appropriate benchmark strategy."""
        mode = self.benchmark.mode
        sampling = self.benchmark.sampling

        if mode == "throughput":
            return ThroughputStrategy(sampling)
        elif mode == "latency":
            return LatencyStrategy(sampling)
        elif mode == "serve":
            return ServeStrategy(sampling)
        elif mode == "startup":
            return StartupStrategy()
        else:
            raise ValueError(f"Unknown benchmark mode: {mode}")

    async def _run_warmup(self, backend: Backend, strategy) -> None:
        """Run warmup iterations."""
        warmup_count = self.benchmark.warmup
        if warmup_count <= 0:
            return

        logger.info(f"Running {warmup_count} warmup iterations...")

        for _ in tqdm(range(warmup_count), desc="Warmup", leave=False):
            await strategy.run(
                self.executor,
                backend,
                self.prompts,
                self.benchmark,
            )

        logger.info("Warmup complete")

    async def _run_single_benchmark(
        self,
        backend: Backend,
        strategy,
    ) -> list[ConcurrencyResult]:
        """Run benchmark at all concurrency levels (single trial)."""
        concurrency_results: list[ConcurrencyResult] = []

        pbar = tqdm(self.benchmark.concurrency, desc="Benchmarking", leave=False)
        for concurrency in pbar:
            pbar.set_postfix(concurrency=concurrency)
            logger.info(f"Running benchmark at concurrency={concurrency}")

            # Create modified config for this concurrency level
            bench_config = self.benchmark.model_copy()
            bench_config.concurrency = [concurrency]

            start_time = time.perf_counter()

            results = await strategy.run(
                self.executor,
                backend,
                self.prompts,
                bench_config,
            )

            total_time = time.perf_counter() - start_time

            agg = aggregate_results(
                results=results,
                concurrency=concurrency,
                total_time=total_time,
                slo=self.benchmark.slo,
                include_raw=self.include_raw,
            )
            concurrency_results.append(agg)

            logger.info(
                f"  Completed: {agg.num_successful}/{agg.num_requests} requests, "
                f"{agg.throughput_tokens_per_sec:.1f} tokens/s"
            )

        return concurrency_results

    def _build_config_dict(self) -> dict[str, Any]:
        """Build config dict for results."""
        return self.benchmark.model_dump()

    async def run(self, backend: Backend) -> BenchmarkResults:
        """Run the full benchmark."""
        backend_config = self.benchmark.backend
        environment = capture_environment(
            backend_type=backend_config.type,
            model_name=getattr(backend_config, "model", None),
            quantization=getattr(backend_config, "quantization", None),
            dtype=getattr(backend_config, "dtype", None),
        )
        logger.info(f"Environment: {environment.format_summary()}")
        logger.info(f"Benchmarking with {len(self.prompts)} prompts")

        strategy = self._get_strategy()
        logger.info(f"Using {strategy.__class__.__name__} strategy")

        n_trials = self.benchmark.trials

        if n_trials == 1:
            await self._run_warmup(backend, strategy)
            concurrency_results = await self._run_single_benchmark(backend, strategy)

            return BenchmarkResults(
                engine=backend_config.type,
                model=getattr(backend_config, "model", None) or "unknown",
                timestamp=datetime.now(UTC).isoformat(),
                gpu=self._get_gpu_info(),
                config=self._build_config_dict(),
                results=concurrency_results,
                environment=environment,
                n_trials=1,
            )

        # Multiple trials
        logger.info(f"Running {n_trials} independent trials for statistical rigor")
        trial_results: list[TrialResult] = []

        for trial_idx in tqdm(range(n_trials), desc="Trials"):
            await self._run_warmup(backend, strategy)
            concurrency_results = await self._run_single_benchmark(backend, strategy)
            trial_results.append(
                TrialResult(trial_index=trial_idx, concurrency_results=concurrency_results)
            )

        logger.info("Aggregating results across trials...")
        aggregated = aggregate_trials(trial_results, self.benchmark.confidence_level)

        return BenchmarkResults(
            engine=backend_config.type,
            model=getattr(backend_config, "model", None) or "unknown",
            timestamp=datetime.now(UTC).isoformat(),
            gpu=self._get_gpu_info(),
            config=self._build_config_dict(),
            results=trial_results[0].concurrency_results,
            environment=environment,
            n_trials=n_trials,
            trial_results=trial_results,
            aggregated_results=aggregated,
        )

    def _get_gpu_info(self) -> str | None:
        """Get GPU information string (for backwards compatibility)."""
        try:
            from splleed.environment import detect_gpus

            gpus = detect_gpus()
            if gpus:
                return format_gpu_info(gpus)
        except Exception:
            pass
        return None
