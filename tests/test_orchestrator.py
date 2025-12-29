"""Tests for benchmark orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from splleed import Benchmark, SamplingParams
from splleed.backends.base import Backend, BackendConfigBase, GenerateRequest
from splleed.backends.vllm import VLLMConfig
from splleed.runner.orchestrator import BenchmarkOrchestrator


class MockConfig(BackendConfigBase):
    """Mock config for testing."""

    type: str = "mock"
    model: str = "mock-model"
    endpoint: str | None = None


class MockBackend(Backend[MockConfig]):
    """Mock backend for testing orchestrator."""

    def __init__(
        self,
        tokens: list[str] | None = None,
        delay_per_token: float = 0.001,
    ) -> None:
        self.config = MockConfig()
        self.tokens = tokens or ["Hello", " ", "world", "!"]
        self.delay_per_token = delay_per_token
        self.request_count = 0
        self.started = False
        self.shutdown_called = False

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        """Generate mock tokens."""
        self.request_count += 1
        for token in self.tokens:
            await asyncio.sleep(self.delay_per_token)
            yield token

    async def health(self) -> bool:
        return True

    async def connect(self, endpoint: str) -> None:
        self.started = True

    async def start(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def mock_backend() -> MockBackend:
    """Create a mock backend."""
    backend = MockBackend()
    backend.started = True
    return backend


@pytest.fixture
def prompts() -> list[str]:
    """Create test prompts."""
    return ["Hello", "World", "Test"]


def make_benchmark(
    prompts: list[str],
    mode: str = "throughput",
    concurrency: list[int] | None = None,
    warmup: int = 0,
    trials: int = 1,
    arrival_rate: float | None = None,
    arrival_pattern: str = "poisson",
) -> Benchmark:
    """Helper to create a Benchmark for testing."""
    return Benchmark(
        backend=VLLMConfig(model="test-model"),
        prompts=prompts,
        mode=mode,  # type: ignore[arg-type]
        concurrency=concurrency or [1],
        warmup=warmup,
        trials=trials,
        sampling=SamplingParams(max_tokens=50),
        arrival_rate=arrival_rate,
        arrival_pattern=arrival_pattern,  # type: ignore[arg-type]
    )


class TestBenchmarkOrchestrator:
    """Tests for BenchmarkOrchestrator."""

    @pytest.mark.asyncio
    async def test_run_basic(
        self,
        mock_backend: MockBackend,
        prompts: list[str],
    ):
        """Test basic orchestrator run."""
        benchmark = make_benchmark(prompts, mode="throughput")
        orchestrator = BenchmarkOrchestrator(benchmark, prompts)
        results = await orchestrator.run(mock_backend)

        assert results.engine == "vllm"
        assert results.model == "test-model"
        assert len(results.results) == 1
        assert results.results[0].concurrency == 1

    @pytest.mark.asyncio
    async def test_run_multiple_concurrency(
        self,
        mock_backend: MockBackend,
        prompts: list[str],
    ):
        """Test running at multiple concurrency levels."""
        benchmark = make_benchmark(prompts, mode="throughput", concurrency=[1, 2, 4])
        orchestrator = BenchmarkOrchestrator(benchmark, prompts)
        results = await orchestrator.run(mock_backend)

        assert len(results.results) == 3
        assert results.results[0].concurrency == 1
        assert results.results[1].concurrency == 2
        assert results.results[2].concurrency == 4

    @pytest.mark.asyncio
    async def test_run_with_warmup(
        self,
        mock_backend: MockBackend,
        prompts: list[str],
    ):
        """Test that warmup runs are executed."""
        benchmark = make_benchmark(prompts, mode="throughput", warmup=2)
        orchestrator = BenchmarkOrchestrator(benchmark, prompts)
        await orchestrator.run(mock_backend)

        assert mock_backend.request_count >= 3

    @pytest.mark.asyncio
    async def test_results_have_metrics(
        self,
        mock_backend: MockBackend,
        prompts: list[str],
    ):
        """Test that results contain metrics."""
        benchmark = make_benchmark(prompts)
        orchestrator = BenchmarkOrchestrator(benchmark, prompts)
        results = await orchestrator.run(mock_backend)

        r = results.results[0]
        assert r.num_requests > 0
        assert r.throughput_tokens_per_sec > 0
        assert r.ttft_p50_ms >= 0
        assert r.itl_mean_ms >= 0

    @pytest.mark.asyncio
    async def test_throughput_mode(
        self,
        mock_backend: MockBackend,
        prompts: list[str],
    ):
        """Test throughput benchmark mode."""
        benchmark = make_benchmark(prompts, mode="throughput")
        orchestrator = BenchmarkOrchestrator(benchmark, prompts)
        results = await orchestrator.run(mock_backend)

        assert len(results.results) > 0

    @pytest.mark.asyncio
    async def test_serve_mode(
        self,
        mock_backend: MockBackend,
        prompts: list[str],
    ):
        """Test serve benchmark mode."""
        benchmark = make_benchmark(
            prompts,
            mode="serve",
            arrival_rate=100,
            arrival_pattern="constant",
        )
        orchestrator = BenchmarkOrchestrator(benchmark, prompts)
        results = await orchestrator.run(mock_backend)

        assert len(results.results) > 0

    @pytest.mark.asyncio
    async def test_metadata_included(
        self,
        mock_backend: MockBackend,
        prompts: list[str],
    ):
        """Test that metadata is included in results."""
        benchmark = make_benchmark(prompts)
        orchestrator = BenchmarkOrchestrator(benchmark, prompts)
        results = await orchestrator.run(mock_backend)

        assert results.timestamp is not None
        assert results.config is not None


class TestStrategySelection:
    """Tests for strategy selection."""

    @pytest.mark.asyncio
    async def test_latency_strategy(self, prompts: list[str]):
        """Test that latency mode uses LatencyStrategy."""
        benchmark = make_benchmark(prompts, mode="latency")
        orchestrator = BenchmarkOrchestrator(benchmark, prompts)
        strategy = orchestrator._get_strategy()

        from splleed.runner.strategies import LatencyStrategy

        assert isinstance(strategy, LatencyStrategy)

    @pytest.mark.asyncio
    async def test_throughput_strategy(self, prompts: list[str]):
        """Test that throughput mode uses ThroughputStrategy."""
        benchmark = make_benchmark(prompts, mode="throughput")
        orchestrator = BenchmarkOrchestrator(benchmark, prompts)
        strategy = orchestrator._get_strategy()

        from splleed.runner.strategies import ThroughputStrategy

        assert isinstance(strategy, ThroughputStrategy)

    @pytest.mark.asyncio
    async def test_serve_strategy(self, prompts: list[str]):
        """Test that serve mode uses ServeStrategy."""
        benchmark = make_benchmark(prompts, mode="serve")
        orchestrator = BenchmarkOrchestrator(benchmark, prompts)
        strategy = orchestrator._get_strategy()

        from splleed.runner.strategies import ServeStrategy

        assert isinstance(strategy, ServeStrategy)

    def test_invalid_mode_raises(self, prompts: list[str]):
        """Test that invalid mode raises error."""
        benchmark = make_benchmark(prompts)
        # Bypass Pydantic validation
        object.__setattr__(benchmark, "mode", "invalid")

        orchestrator = BenchmarkOrchestrator(benchmark, prompts)

        with pytest.raises(ValueError, match="Unknown benchmark mode"):
            orchestrator._get_strategy()
