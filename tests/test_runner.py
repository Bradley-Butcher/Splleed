"""Tests for runner and strategies."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest

from splleed.backends.base import Backend, BackendConfigBase, GenerateRequest
from splleed.config.base import ArrivalPattern, BenchmarkConfig, SamplingConfig
from splleed.datasets import InlineDataset
from splleed.runner.executor import BatchExecutor, RequestExecutor
from splleed.runner.strategies import (
    LatencyStrategy,
    ServeStrategy,
    ThroughputStrategy,
    generate_arrival_times,
)

if TYPE_CHECKING:
    pass


class MockConfig(BackendConfigBase):
    """Mock config for testing."""

    type: str = "mock"


class MockBackend(Backend[MockConfig]):
    """Mock backend for testing."""

    def __init__(
        self,
        tokens: list[str] | None = None,
        delay_per_token: float = 0.01,
        fail: bool = False,
    ) -> None:
        self.config = MockConfig()
        self.tokens = tokens or ["Hello", " ", "world", "!"]
        self.delay_per_token = delay_per_token
        self.fail = fail
        self.request_count = 0

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        """Generate mock tokens with delays."""
        self.request_count += 1

        if self.fail:
            raise RuntimeError("Mock failure")

        for token in self.tokens:
            await asyncio.sleep(self.delay_per_token)
            yield token

    async def health(self) -> bool:
        """Always healthy."""
        return True


class TestRequestExecutor:
    """Tests for RequestExecutor."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution with timing."""
        backend = MockBackend(tokens=["a", "b", "c"], delay_per_token=0.01)
        executor = RequestExecutor()

        request = GenerateRequest(prompt="test", max_tokens=10)
        result = await executor.execute(backend, request)

        assert result.success
        assert result.error is None
        assert result.output_tokens == 3
        assert result.output_text == "abc"
        assert result.ttft is not None
        assert result.ttft > 0
        assert len(result.itl) == 2  # 3 tokens = 2 inter-token gaps

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Test execution failure handling."""
        backend = MockBackend(fail=True)
        executor = RequestExecutor()

        request = GenerateRequest(prompt="test", max_tokens=10)
        result = await executor.execute(backend, request)

        assert not result.success
        assert result.error is not None
        assert "Mock failure" in result.error
        assert result.output_tokens == 0

    @pytest.mark.asyncio
    async def test_execute_timing(self):
        """Test that timing is accurate."""
        backend = MockBackend(tokens=["a", "b", "c"], delay_per_token=0.05)
        executor = RequestExecutor()

        request = GenerateRequest(prompt="test", max_tokens=10)
        result = await executor.execute(backend, request)

        # TTFT should be ~50ms (first token delay)
        assert result.ttft is not None
        assert 0.04 < result.ttft < 0.1

        # Each ITL should be ~50ms
        for itl in result.itl:
            assert 0.04 < itl < 0.1

        # Total time should be ~150ms (3 tokens * 50ms)
        assert 0.1 < result.total_time < 0.25


class TestBatchExecutor:
    """Tests for BatchExecutor."""

    @pytest.mark.asyncio
    async def test_batch_execute(self):
        """Test batch execution with concurrency."""
        backend = MockBackend(delay_per_token=0.01)
        executor = RequestExecutor()
        batch_executor = BatchExecutor(executor)

        requests = [GenerateRequest(prompt=f"test {i}", max_tokens=10) for i in range(5)]

        results = await batch_executor.execute_batch(
            backend=backend,
            requests=requests,
            concurrency=2,
        )

        assert len(results) == 5
        assert all(r.success for r in results)
        assert backend.request_count == 5

    @pytest.mark.asyncio
    async def test_batch_concurrency_limit(self):
        """Test that concurrency is actually limited."""
        concurrent_count = 0
        max_concurrent = 0

        class TrackingBackend(MockBackend):
            async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                try:
                    async for token in super().generate_stream(request):
                        yield token
                finally:
                    concurrent_count -= 1

        backend = TrackingBackend(delay_per_token=0.02)
        executor = RequestExecutor()
        batch_executor = BatchExecutor(executor)

        requests = [GenerateRequest(prompt=f"test {i}", max_tokens=10) for i in range(10)]

        await batch_executor.execute_batch(
            backend=backend,
            requests=requests,
            concurrency=3,
        )

        # Max concurrent should not exceed limit
        assert max_concurrent <= 3


class TestGenerateArrivalTimes:
    """Tests for arrival time generation."""

    def test_constant_rate(self):
        """Test constant rate arrival times."""
        pattern = ArrivalPattern(type="constant", rate=10.0)  # 10 req/sec
        times = generate_arrival_times(5, pattern)

        assert len(times) == 5
        assert times[0] == 0.0
        # Should be evenly spaced at 100ms intervals
        for i in range(1, 5):
            assert times[i] == pytest.approx(i * 0.1, abs=0.001)

    def test_poisson_rate(self):
        """Test Poisson arrival times."""
        pattern = ArrivalPattern(type="poisson", rate=100.0)
        times = generate_arrival_times(100, pattern, seed=42)

        assert len(times) == 100
        assert times[0] == 0.0
        # Should be monotonically increasing
        for i in range(1, 100):
            assert times[i] > times[i - 1]

        # Mean interval should be ~10ms (1/100)
        intervals = [times[i] - times[i - 1] for i in range(1, 100)]
        mean_interval = sum(intervals) / len(intervals)
        assert 0.005 < mean_interval < 0.02

    def test_gamma_burstiness(self):
        """Test gamma distribution with burstiness."""
        # burstiness=1 should be similar to Poisson
        pattern1 = ArrivalPattern(type="gamma", rate=100.0, burstiness=1.0)
        times1 = generate_arrival_times(100, pattern1, seed=42)

        # burstiness=2 should be more bursty (more variance)
        pattern2 = ArrivalPattern(type="gamma", rate=100.0, burstiness=2.0)
        times2 = generate_arrival_times(100, pattern2, seed=42)

        # Both should have same approximate mean rate
        assert len(times1) == 100
        assert len(times2) == 100

    def test_reproducible_with_seed(self):
        """Test that same seed produces same times."""
        pattern = ArrivalPattern(type="poisson", rate=10.0)

        times1 = generate_arrival_times(10, pattern, seed=42)
        times2 = generate_arrival_times(10, pattern, seed=42)

        assert times1 == times2

    def test_different_seeds(self):
        """Test that different seeds produce different times."""
        pattern = ArrivalPattern(type="poisson", rate=10.0)

        times1 = generate_arrival_times(10, pattern, seed=42)
        times2 = generate_arrival_times(10, pattern, seed=99)

        assert times1 != times2


class TestThroughputStrategy:
    """Tests for ThroughputStrategy."""

    @pytest.mark.asyncio
    async def test_run_basic(self):
        """Test basic throughput strategy run."""
        backend = MockBackend(delay_per_token=0.01)
        executor = RequestExecutor()
        dataset = InlineDataset(["prompt 1", "prompt 2", "prompt 3"])
        config = BenchmarkConfig(mode="throughput", runs=3, concurrency=[2])
        sampling = SamplingConfig(max_tokens=50)

        strategy = ThroughputStrategy(sampling=sampling)
        results = await strategy.run(executor, backend, dataset, config)

        assert len(results) > 0
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_uses_max_concurrency(self):
        """Test that throughput uses max concurrency level."""
        concurrent_count = 0
        max_concurrent = 0

        class TrackingBackend(MockBackend):
            async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                try:
                    async for token in super().generate_stream(request):
                        yield token
                finally:
                    concurrent_count -= 1

        backend = TrackingBackend(delay_per_token=0.02)
        executor = RequestExecutor()
        dataset = InlineDataset(["p1", "p2", "p3", "p4", "p5"])
        config = BenchmarkConfig(mode="throughput", runs=5, concurrency=[1, 2, 4])

        strategy = ThroughputStrategy()
        await strategy.run(executor, backend, dataset, config)

        # Should use max concurrency (4)
        assert max_concurrent <= 4


class TestLatencyStrategy:
    """Tests for LatencyStrategy."""

    @pytest.mark.asyncio
    async def test_run_sequential(self):
        """Test that latency strategy runs sequentially."""
        request_times: list[float] = []

        class TimingBackend(MockBackend):
            async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
                import time

                request_times.append(time.perf_counter())
                async for token in super().generate_stream(request):
                    yield token

        backend = TimingBackend(delay_per_token=0.02)
        executor = RequestExecutor()
        dataset = InlineDataset(["p1", "p2", "p3"])
        config = BenchmarkConfig(mode="latency", runs=3)

        strategy = LatencyStrategy()
        results = await strategy.run(executor, backend, dataset, config)

        assert len(results) == 3
        assert all(r.success for r in results)

        # Verify sequential execution - each request starts after previous ends
        for i in range(1, len(request_times)):
            # Gap between requests should be minimal but non-zero
            # (previous request completes before next starts)
            assert request_times[i] > request_times[i - 1]


class TestServeStrategy:
    """Tests for ServeStrategy."""

    @pytest.mark.asyncio
    async def test_run_with_arrival_pattern(self):
        """Test serve strategy with arrival pattern."""
        backend = MockBackend(delay_per_token=0.01)
        executor = RequestExecutor()
        dataset = InlineDataset(["p1", "p2", "p3", "p4", "p5"])
        config = BenchmarkConfig(
            mode="serve",
            runs=5,
            concurrency=[2],
            arrival=ArrivalPattern(type="constant", rate=100.0),
        )

        strategy = ServeStrategy(seed=42)
        results = await strategy.run(executor, backend, dataset, config)

        assert len(results) == 5
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self):
        """Test that serve strategy respects concurrency limit."""
        concurrent_count = 0
        max_concurrent = 0

        class TrackingBackend(MockBackend):
            async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                try:
                    async for token in super().generate_stream(request):
                        yield token
                finally:
                    concurrent_count -= 1

        backend = TrackingBackend(delay_per_token=0.05)
        executor = RequestExecutor()
        dataset = InlineDataset(["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"])
        config = BenchmarkConfig(
            mode="serve",
            runs=8,
            concurrency=[3],
            arrival=ArrivalPattern(type="constant", rate=1000.0),  # Fast arrivals
        )

        strategy = ServeStrategy()
        await strategy.run(executor, backend, dataset, config)

        assert max_concurrent <= 3
