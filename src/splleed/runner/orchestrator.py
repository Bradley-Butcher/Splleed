"""Benchmark orchestrator - coordinates the full benchmark run."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from splleed.datasets import analyze_dataset, get_dataset, print_dataset_analysis
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
from splleed.tokenizers import get_tokenizer

if TYPE_CHECKING:
    from splleed.backends.base import Backend
    from splleed.config.loader import FullConfig
    from splleed.datasets.base import Dataset
    from splleed.tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """
    Orchestrates the full benchmark workflow.

    Coordinates:
    - Backend lifecycle (start/connect/shutdown)
    - Tokenizer initialization (optional)
    - Dataset analysis (optional)
    - Warmup runs
    - Multiple independent trials (for statistical rigor)
    - Running benchmarks at different concurrency levels
    - Metrics aggregation with confidence intervals
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
        self.tokenizer: Tokenizer | None = None

        # Initialize tokenizer if configured
        if config.tokenizer.enabled:
            default_model = getattr(config.backend, "model", None)
            self.tokenizer = get_tokenizer(config.tokenizer, default_model)
            if self.tokenizer:
                logger.info(f"Tokenizer loaded: {self.tokenizer.name}")

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

    async def _run_warmup(self, backend: Backend, strategy, dataset: Dataset) -> None:
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

    async def _run_single_benchmark(
        self,
        backend: Backend,
        strategy,
        dataset: Dataset,
    ) -> list[ConcurrencyResult]:
        """
        Run benchmark at all concurrency levels (single trial).

        Returns:
            List of ConcurrencyResult, one per concurrency level
        """
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

        return concurrency_results

    def _aggregate_trials(
        self,
        trial_results: list[TrialResult],
        confidence_level: float,
    ) -> list[ConcurrencyResultWithCI]:
        """
        Aggregate results across trials into CIs.

        Args:
            trial_results: List of TrialResult from each trial
            confidence_level: Confidence level for CI computation

        Returns:
            List of ConcurrencyResultWithCI with aggregated metrics
        """
        if not trial_results:
            return []

        # Get concurrency levels from first trial
        concurrency_levels = [cr.concurrency for cr in trial_results[0].concurrency_results]

        aggregated: list[ConcurrencyResultWithCI] = []

        for idx, concurrency in enumerate(concurrency_levels):
            # Collect values from each trial for this concurrency level
            trial_data = [tr.concurrency_results[idx] for tr in trial_results]

            # Helper to compute CI from list of values
            def ci(values: list[float]) -> ConfidenceInterval:
                return compute_ci(values, confidence_level)

            aggregated.append(
                ConcurrencyResultWithCI(
                    concurrency=concurrency,
                    num_requests=sum(td.num_requests for td in trial_data),
                    num_successful=sum(td.num_successful for td in trial_data),
                    num_failed=sum(td.num_failed for td in trial_data),
                    # Throughput
                    throughput_tokens_per_sec=ci(
                        [td.throughput_tokens_per_sec for td in trial_data]
                    ),
                    throughput_requests_per_sec=ci(
                        [td.throughput_requests_per_sec for td in trial_data]
                    ),
                    # TTFT
                    ttft_p50_ms=ci([td.ttft_p50_ms for td in trial_data]),
                    ttft_p95_ms=ci([td.ttft_p95_ms for td in trial_data]),
                    ttft_p99_ms=ci([td.ttft_p99_ms for td in trial_data]),
                    ttft_mean_ms=ci([td.ttft_mean_ms for td in trial_data]),
                    # ITL
                    itl_p50_ms=ci([td.itl_p50_ms for td in trial_data]),
                    itl_p95_ms=ci([td.itl_p95_ms for td in trial_data]),
                    itl_p99_ms=ci([td.itl_p99_ms for td in trial_data]),
                    itl_mean_ms=ci([td.itl_mean_ms for td in trial_data]),
                    # TPOT
                    tpot_mean_ms=ci([td.tpot_mean_ms for td in trial_data]),
                    # E2E
                    e2el_p50_ms=ci([td.e2el_p50_ms for td in trial_data]),
                    e2el_p95_ms=ci([td.e2el_p95_ms for td in trial_data]),
                    e2el_p99_ms=ci([td.e2el_p99_ms for td in trial_data]),
                    e2el_mean_ms=ci([td.e2el_mean_ms for td in trial_data]),
                    # Goodput
                    goodput_pct=ci(
                        [td.goodput_pct for td in trial_data if td.goodput_pct is not None]
                    )
                    if any(td.goodput_pct is not None for td in trial_data)
                    else None,
                )
            )

        return aggregated

    async def run(self, backend: Backend) -> BenchmarkResults:
        """
        Run the full benchmark.

        Args:
            backend: Initialized inference backend

        Returns:
            Complete benchmark results
        """
        # Capture environment information
        environment = capture_environment(
            backend_type=self.config.backend.type,
            model_name=getattr(self.config.backend, "model", None),
            quantization=getattr(self.config.backend, "quantization", None),
            dtype=getattr(self.config.backend, "dtype", None),
        )
        logger.info(f"Environment: {environment.format_summary()}")

        # Load dataset
        dataset = get_dataset(self.config.dataset)
        logger.info(f"Loaded dataset with {len(dataset)} samples")

        # Analyze dataset if requested
        if self.config.dataset.analyze_before_run:
            stats = analyze_dataset(dataset, tokenizer=self.tokenizer)
            print_dataset_analysis(stats)

        # Get strategy
        strategy = self._get_strategy()
        logger.info(f"Using {strategy.__class__.__name__} strategy")

        n_trials = self.config.benchmark.trials

        if n_trials == 1:
            # Single trial - existing behavior
            await self._run_warmup(backend, strategy, dataset)
            concurrency_results = await self._run_single_benchmark(backend, strategy, dataset)

            return BenchmarkResults(
                engine=self.config.backend.type,
                model=getattr(self.config.backend, "model", None) or "unknown",
                timestamp=datetime.now(UTC).isoformat(),
                gpu=self._get_gpu_info(),
                config=self.config.model_dump(),
                results=concurrency_results,
                environment=environment,
                n_trials=1,
            )

        # Multiple trials
        logger.info(f"Running {n_trials} independent trials for statistical rigor")

        trial_results: list[TrialResult] = []

        for trial_idx in range(n_trials):
            logger.info(f"=== Trial {trial_idx + 1}/{n_trials} ===")

            # Run warmup before each trial
            await self._run_warmup(backend, strategy, dataset)

            # Run benchmark
            concurrency_results = await self._run_single_benchmark(backend, strategy, dataset)

            trial_results.append(
                TrialResult(
                    trial_index=trial_idx,
                    concurrency_results=concurrency_results,
                )
            )

        # Aggregate across trials
        logger.info("Aggregating results across trials...")
        aggregated = self._aggregate_trials(
            trial_results,
            self.config.benchmark.confidence_level,
        )

        return BenchmarkResults(
            engine=self.config.backend.type,
            model=getattr(self.config.backend, "model", None) or "unknown",
            timestamp=datetime.now(UTC).isoformat(),
            gpu=self._get_gpu_info(),
            config=self.config.model_dump(),
            results=trial_results[0].concurrency_results,  # First trial for compat
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
