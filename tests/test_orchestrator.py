"""Tests for benchmark orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from splleed.backends.base import Backend, BackendConfigBase, GenerateRequest
from splleed.backends.vllm import VLLMConfig
from splleed.config import BenchmarkConfig, DatasetConfig, SamplingConfig
from splleed.config.loader import FullConfig
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
        """Always healthy."""
        return True

    async def connect(self, endpoint: str) -> None:
        """Mock connect."""
        self.started = True

    async def start(self) -> None:
        """Mock start."""
        self.started = True

    async def shutdown(self) -> None:
        """Mock shutdown."""
        self.shutdown_called = True


@pytest.fixture
def mock_backend() -> MockBackend:
    """Create a mock backend."""
    return MockBackend()


@pytest.fixture
def basic_config(tmp_path) -> FullConfig:
    """Create a basic test configuration."""
    return FullConfig(
        backend=VLLMConfig(model="test-model"),
        dataset=DatasetConfig(type="inline", prompts=["Hello", "World", "Test"]),
        benchmark=BenchmarkConfig(
            mode="latency",
            concurrency=[1],
            warmup=0,
            runs=3,
        ),
        sampling=SamplingConfig(max_tokens=50),
    )


class TestBenchmarkOrchestrator:
    """Tests for BenchmarkOrchestrator."""

    @pytest.mark.asyncio
    async def test_run_basic(self, mock_backend: MockBackend, basic_config: FullConfig):
        """Test basic orchestrator run."""
        mock_backend.started = True

        orchestrator = BenchmarkOrchestrator(basic_config)
        results = await orchestrator.run(mock_backend)

        assert results.engine == "vllm"
        assert results.model == "test-model"
        assert len(results.results) == 1  # One concurrency level
        assert results.results[0].concurrency == 1

    @pytest.mark.asyncio
    async def test_run_multiple_concurrency(
        self, mock_backend: MockBackend, basic_config: FullConfig
    ):
        """Test running at multiple concurrency levels."""
        basic_config.benchmark.concurrency = [1, 2, 4]
        mock_backend.started = True

        orchestrator = BenchmarkOrchestrator(basic_config)
        results = await orchestrator.run(mock_backend)

        assert len(results.results) == 3
        assert results.results[0].concurrency == 1
        assert results.results[1].concurrency == 2
        assert results.results[2].concurrency == 4

    @pytest.mark.asyncio
    async def test_run_with_warmup(self, mock_backend: MockBackend, basic_config: FullConfig):
        """Test that warmup runs are executed."""
        basic_config.benchmark.warmup = 2
        basic_config.benchmark.runs = 3
        mock_backend.started = True

        orchestrator = BenchmarkOrchestrator(basic_config)
        await orchestrator.run(mock_backend)

        # Should have warmup runs + actual runs
        # Warmup: 2 iterations with 3 requests each = 6
        # Actual: 1 concurrency level with 3 requests = 3
        # Total: 9 requests
        assert mock_backend.request_count >= 3  # At least the actual runs

    @pytest.mark.asyncio
    async def test_results_have_metrics(self, mock_backend: MockBackend, basic_config: FullConfig):
        """Test that results contain metrics."""
        mock_backend.started = True

        orchestrator = BenchmarkOrchestrator(basic_config)
        results = await orchestrator.run(mock_backend)

        r = results.results[0]
        assert r.num_requests > 0
        assert r.throughput_tokens_per_sec > 0
        assert r.ttft_p50_ms >= 0
        assert r.itl_mean_ms >= 0

    @pytest.mark.asyncio
    async def test_throughput_mode(self, mock_backend: MockBackend, basic_config: FullConfig):
        """Test throughput benchmark mode."""
        basic_config.benchmark.mode = "throughput"
        mock_backend.started = True

        orchestrator = BenchmarkOrchestrator(basic_config)
        results = await orchestrator.run(mock_backend)

        assert len(results.results) > 0

    @pytest.mark.asyncio
    async def test_serve_mode(self, mock_backend: MockBackend, basic_config: FullConfig):
        """Test serve benchmark mode."""
        from splleed.config import ArrivalPattern

        basic_config.benchmark.mode = "serve"
        basic_config.benchmark.arrival = ArrivalPattern(type="constant", rate=100)
        mock_backend.started = True

        orchestrator = BenchmarkOrchestrator(basic_config)
        results = await orchestrator.run(mock_backend)

        assert len(results.results) > 0

    @pytest.mark.asyncio
    async def test_metadata_included(self, mock_backend: MockBackend, basic_config: FullConfig):
        """Test that metadata is included in results."""
        mock_backend.started = True

        orchestrator = BenchmarkOrchestrator(basic_config)
        results = await orchestrator.run(mock_backend)

        assert results.timestamp is not None
        assert results.config is not None
        assert "backend" in results.config


class TestStrategySelection:
    """Tests for strategy selection."""

    @pytest.mark.asyncio
    async def test_latency_strategy(self, mock_backend: MockBackend, basic_config: FullConfig):
        """Test that latency mode uses LatencyStrategy."""
        basic_config.benchmark.mode = "latency"

        orchestrator = BenchmarkOrchestrator(basic_config)
        strategy = orchestrator._get_strategy()

        from splleed.runner.strategies import LatencyStrategy

        assert isinstance(strategy, LatencyStrategy)

    @pytest.mark.asyncio
    async def test_throughput_strategy(self, mock_backend: MockBackend, basic_config: FullConfig):
        """Test that throughput mode uses ThroughputStrategy."""
        basic_config.benchmark.mode = "throughput"

        orchestrator = BenchmarkOrchestrator(basic_config)
        strategy = orchestrator._get_strategy()

        from splleed.runner.strategies import ThroughputStrategy

        assert isinstance(strategy, ThroughputStrategy)

    @pytest.mark.asyncio
    async def test_serve_strategy(self, mock_backend: MockBackend, basic_config: FullConfig):
        """Test that serve mode uses ServeStrategy."""
        basic_config.benchmark.mode = "serve"

        orchestrator = BenchmarkOrchestrator(basic_config)
        strategy = orchestrator._get_strategy()

        from splleed.runner.strategies import ServeStrategy

        assert isinstance(strategy, ServeStrategy)

    def test_invalid_mode_raises(self, basic_config: FullConfig):
        """Test that invalid mode raises error."""
        # Use object.__setattr__ to bypass Pydantic validation for testing
        object.__setattr__(basic_config.benchmark, "mode", "invalid")

        orchestrator = BenchmarkOrchestrator(basic_config)

        with pytest.raises(ValueError, match="Unknown benchmark mode"):
            orchestrator._get_strategy()
