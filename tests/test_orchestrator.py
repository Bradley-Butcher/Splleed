"""Tests for benchmark orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from splleed.backends.base import Backend, BackendConfigBase, GenerateRequest
from splleed.backends.vllm import VLLMConfig
from splleed.config import ArrivalPattern, BenchmarkConfig, SamplingParams
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
def backend_config() -> VLLMConfig:
    """Create a backend config."""
    return VLLMConfig(model="test-model")


@pytest.fixture
def benchmark_config() -> BenchmarkConfig:
    """Create a benchmark config."""
    return BenchmarkConfig(
        mode="latency",
        concurrency=[1],
        warmup=0,
        runs=3,
    )


@pytest.fixture
def sampling() -> SamplingParams:
    """Create sampling params."""
    return SamplingParams(max_tokens=50)


@pytest.fixture
def prompts() -> list[str]:
    """Create test prompts."""
    return ["Hello", "World", "Test"]


class TestBenchmarkOrchestrator:
    """Tests for BenchmarkOrchestrator."""

    @pytest.mark.asyncio
    async def test_run_basic(
        self,
        mock_backend: MockBackend,
        backend_config: VLLMConfig,
        benchmark_config: BenchmarkConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test basic orchestrator run."""
        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )
        results = await orchestrator.run(mock_backend)

        assert results.engine == "vllm"
        assert results.model == "test-model"
        assert len(results.results) == 1
        assert results.results[0].concurrency == 1

    @pytest.mark.asyncio
    async def test_run_multiple_concurrency(
        self,
        mock_backend: MockBackend,
        backend_config: VLLMConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test running at multiple concurrency levels."""
        benchmark_config = BenchmarkConfig(
            mode="latency",
            concurrency=[1, 2, 4],
            warmup=0,
            runs=3,
        )

        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )
        results = await orchestrator.run(mock_backend)

        assert len(results.results) == 3
        assert results.results[0].concurrency == 1
        assert results.results[1].concurrency == 2
        assert results.results[2].concurrency == 4

    @pytest.mark.asyncio
    async def test_run_with_warmup(
        self,
        mock_backend: MockBackend,
        backend_config: VLLMConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test that warmup runs are executed."""
        benchmark_config = BenchmarkConfig(
            mode="latency",
            concurrency=[1],
            warmup=2,
            runs=3,
        )

        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )
        await orchestrator.run(mock_backend)

        assert mock_backend.request_count >= 3

    @pytest.mark.asyncio
    async def test_results_have_metrics(
        self,
        mock_backend: MockBackend,
        backend_config: VLLMConfig,
        benchmark_config: BenchmarkConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test that results contain metrics."""
        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )
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
        backend_config: VLLMConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test throughput benchmark mode."""
        benchmark_config = BenchmarkConfig(
            mode="throughput",
            concurrency=[1],
            warmup=0,
            runs=3,
        )

        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )
        results = await orchestrator.run(mock_backend)

        assert len(results.results) > 0

    @pytest.mark.asyncio
    async def test_serve_mode(
        self,
        mock_backend: MockBackend,
        backend_config: VLLMConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test serve benchmark mode."""
        benchmark_config = BenchmarkConfig(
            mode="serve",
            concurrency=[1],
            warmup=0,
            runs=2,
            arrival=ArrivalPattern(type="constant", rate=100),
        )

        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )
        results = await orchestrator.run(mock_backend)

        assert len(results.results) > 0

    @pytest.mark.asyncio
    async def test_metadata_included(
        self,
        mock_backend: MockBackend,
        backend_config: VLLMConfig,
        benchmark_config: BenchmarkConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test that metadata is included in results."""
        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )
        results = await orchestrator.run(mock_backend)

        assert results.timestamp is not None
        assert results.config is not None
        assert "backend" in results.config


class TestStrategySelection:
    """Tests for strategy selection."""

    @pytest.mark.asyncio
    async def test_latency_strategy(
        self,
        backend_config: VLLMConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test that latency mode uses LatencyStrategy."""
        benchmark_config = BenchmarkConfig(mode="latency", concurrency=[1])

        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )
        strategy = orchestrator._get_strategy()

        from splleed.runner.strategies import LatencyStrategy

        assert isinstance(strategy, LatencyStrategy)

    @pytest.mark.asyncio
    async def test_throughput_strategy(
        self,
        backend_config: VLLMConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test that throughput mode uses ThroughputStrategy."""
        benchmark_config = BenchmarkConfig(mode="throughput", concurrency=[1])

        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )
        strategy = orchestrator._get_strategy()

        from splleed.runner.strategies import ThroughputStrategy

        assert isinstance(strategy, ThroughputStrategy)

    @pytest.mark.asyncio
    async def test_serve_strategy(
        self,
        backend_config: VLLMConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test that serve mode uses ServeStrategy."""
        benchmark_config = BenchmarkConfig(mode="serve", concurrency=[1])

        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )
        strategy = orchestrator._get_strategy()

        from splleed.runner.strategies import ServeStrategy

        assert isinstance(strategy, ServeStrategy)

    def test_invalid_mode_raises(
        self,
        backend_config: VLLMConfig,
        sampling: SamplingParams,
        prompts: list[str],
    ):
        """Test that invalid mode raises error."""
        benchmark_config = BenchmarkConfig(mode="latency", concurrency=[1])
        # Bypass Pydantic validation
        object.__setattr__(benchmark_config, "mode", "invalid")

        orchestrator = BenchmarkOrchestrator(
            backend_config=backend_config,
            prompts=prompts,
            benchmark=benchmark_config,
            sampling=sampling,
        )

        with pytest.raises(ValueError, match="Unknown benchmark mode"):
            orchestrator._get_strategy()
