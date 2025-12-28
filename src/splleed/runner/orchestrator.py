"""Benchmark orchestrator - coordinates the full benchmark run."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from splleed.config.base import BenchmarkConfig, SamplingParams
from splleed.environment import capture_environment, format_gpu_info
from splleed.metrics import aggregate_results
from splleed.metrics.types import (
    BenchmarkResults,
    ConcurrencyResult,
    ConcurrencyResultWithCI,
    TrialResult,
)
from splleed.runner.executor import RequestExecutor
from splleed.runner.strategies import (
    LatencyStrategy,
    ServeStrategy,
    StartupStrategy,
    ThroughputStrategy,
)
from splleed.stats import ConfidenceInterval, compute_ci

if TYPE_CHECKING:
    from splleed.backends import BackendConfig
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
        *,
        backend_config: BackendConfig,
        prompts: list[str],
        benchmark: BenchmarkConfig,
        sampling: SamplingParams,
        include_raw: bool = False,
    ) -> None:
        self.backend_config = backend_config
        self.prompts = prompts
        self.benchmark = benchmark
        self.sampling = sampling
        self.include_raw = include_raw
        self.executor = RequestExecutor()

    def _get_strategy(self):
        """Get the appropriate benchmark strategy."""
        mode = self.benchmark.mode

        if mode == "throughput":
            return ThroughputStrategy(self.sampling)
        elif mode == "latency":
            return LatencyStrategy(self.sampling)
        elif mode == "serve":
            return ServeStrategy(self.sampling)
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

        for i in range(warmup_count):
            logger.debug(f"Warmup iteration {i + 1}/{warmup_count}")
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

        for concurrency in self.benchmark.concurrency:
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

    def _aggregate_trials(
        self,
        trial_results: list[TrialResult],
        confidence_level: float,
    ) -> list[ConcurrencyResultWithCI]:
        """Aggregate results across trials into CIs."""
        if not trial_results:
            return []

        concurrency_levels = [cr.concurrency for cr in trial_results[0].concurrency_results]
        aggregated: list[ConcurrencyResultWithCI] = []

        for idx, concurrency in enumerate(concurrency_levels):
            trial_data = [tr.concurrency_results[idx] for tr in trial_results]

            def ci(values: list[float]) -> ConfidenceInterval:
                return compute_ci(values, confidence_level)

            aggregated.append(
                ConcurrencyResultWithCI(
                    concurrency=concurrency,
                    num_requests=sum(td.num_requests for td in trial_data),
                    num_successful=sum(td.num_successful for td in trial_data),
                    num_failed=sum(td.num_failed for td in trial_data),
                    throughput_tokens_per_sec=ci(
                        [td.throughput_tokens_per_sec for td in trial_data]
                    ),
                    throughput_requests_per_sec=ci(
                        [td.throughput_requests_per_sec for td in trial_data]
                    ),
                    ttft_p50_ms=ci([td.ttft_p50_ms for td in trial_data]),
                    ttft_p95_ms=ci([td.ttft_p95_ms for td in trial_data]),
                    ttft_p99_ms=ci([td.ttft_p99_ms for td in trial_data]),
                    ttft_mean_ms=ci([td.ttft_mean_ms for td in trial_data]),
                    itl_p50_ms=ci([td.itl_p50_ms for td in trial_data]),
                    itl_p95_ms=ci([td.itl_p95_ms for td in trial_data]),
                    itl_p99_ms=ci([td.itl_p99_ms for td in trial_data]),
                    itl_mean_ms=ci([td.itl_mean_ms for td in trial_data]),
                    tpot_mean_ms=ci([td.tpot_mean_ms for td in trial_data]),
                    e2el_p50_ms=ci([td.e2el_p50_ms for td in trial_data]),
                    e2el_p95_ms=ci([td.e2el_p95_ms for td in trial_data]),
                    e2el_p99_ms=ci([td.e2el_p99_ms for td in trial_data]),
                    e2el_mean_ms=ci([td.e2el_mean_ms for td in trial_data]),
                    goodput_pct=ci(
                        [td.goodput_pct for td in trial_data if td.goodput_pct is not None]
                    )
                    if any(td.goodput_pct is not None for td in trial_data)
                    else None,
                )
            )

        return aggregated

    def _build_config_dict(self) -> dict[str, Any]:
        """Build config dict for results."""
        return {
            "backend": self.backend_config.model_dump(),
            "benchmark": self.benchmark.model_dump(),
            "sampling": self.sampling.model_dump(),
        }

    async def run(self, backend: Backend) -> BenchmarkResults:
        """Run the full benchmark."""
        environment = capture_environment(
            backend_type=self.backend_config.type,
            model_name=getattr(self.backend_config, "model", None),
            quantization=getattr(self.backend_config, "quantization", None),
            dtype=getattr(self.backend_config, "dtype", None),
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
                engine=self.backend_config.type,
                model=getattr(self.backend_config, "model", None) or "unknown",
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

        for trial_idx in range(n_trials):
            logger.info(f"=== Trial {trial_idx + 1}/{n_trials} ===")
            await self._run_warmup(backend, strategy)
            concurrency_results = await self._run_single_benchmark(backend, strategy)
            trial_results.append(
                TrialResult(trial_index=trial_idx, concurrency_results=concurrency_results)
            )

        logger.info("Aggregating results across trials...")
        aggregated = self._aggregate_trials(trial_results, self.benchmark.confidence_level)

        return BenchmarkResults(
            engine=self.backend_config.type,
            model=getattr(self.backend_config, "model", None) or "unknown",
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
