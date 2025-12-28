"""Integration tests for end-to-end benchmark flow."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from splleed import Benchmark, BenchmarkResults, SamplingParams, VLLMConfig
from splleed.backends.base import Backend, GenerateRequest
from splleed.reporters import write_csv, write_json


class MockBackend(Backend["MockBackend"]):
    """Mock backend for integration testing."""

    def __init__(self) -> None:
        self._started = False
        self._request_count = 0

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        """Generate tokens with realistic timing."""
        self._request_count += 1
        tokens = ["The", " answer", " is", " 42", "."]
        for token in tokens:
            await asyncio.sleep(0.001)
            yield token

    async def health(self) -> bool:
        return self._started

    async def connect(self, endpoint: str) -> None:
        self._started = True

    async def start(self) -> None:
        self._started = True

    async def shutdown(self) -> None:
        self._started = False

    @property
    def request_count(self) -> int:
        return self._request_count


@pytest.fixture
def mock_backend() -> MockBackend:
    """Create a started mock backend."""
    backend = MockBackend()
    backend._started = True
    return backend


class TestBenchmarkAPI:
    """Test the public Benchmark API."""

    @pytest.mark.asyncio
    async def test_basic_benchmark(self, mock_backend: MockBackend):
        """Test basic benchmark flow."""
        b = Benchmark(
            backend=VLLMConfig(model="test-model"),
            prompts=["What is 2+2?", "Hello world"],
            concurrency=[1],
            warmup=0,
            runs=2,
        )

        results = await b.run(_backend=mock_backend)

        assert isinstance(results, BenchmarkResults)
        assert results.model == "test-model"
        assert len(results.results) == 1
        assert results.results[0].concurrency == 1
        assert results.results[0].num_requests == 2

    @pytest.mark.asyncio
    async def test_multiple_concurrency_levels(self, mock_backend: MockBackend):
        """Test benchmark with multiple concurrency levels."""
        b = Benchmark(
            backend=VLLMConfig(model="test-model"),
            prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
            concurrency=[1, 2],
            warmup=0,
            runs=3,
        )

        results = await b.run(_backend=mock_backend)

        assert len(results.results) == 2
        assert results.results[0].concurrency == 1
        assert results.results[1].concurrency == 2

    @pytest.mark.asyncio
    async def test_latency_mode(self, mock_backend: MockBackend):
        """Test latency benchmark mode."""
        b = Benchmark(
            backend=VLLMConfig(model="test-model"),
            prompts=["Test prompt"],
            mode="latency",
            concurrency=[1],
            warmup=0,
            runs=3,
        )

        results = await b.run(_backend=mock_backend)

        r = results.results[0]
        assert r.ttft_p50_ms >= 0
        assert r.ttft_p95_ms >= 0
        assert r.itl_mean_ms >= 0

    @pytest.mark.asyncio
    async def test_throughput_mode(self, mock_backend: MockBackend):
        """Test throughput benchmark mode."""
        b = Benchmark(
            backend=VLLMConfig(model="test-model"),
            prompts=["Test prompt 1", "Test prompt 2"],
            mode="throughput",
            concurrency=[2],
            warmup=0,
            runs=2,
        )

        results = await b.run(_backend=mock_backend)

        r = results.results[0]
        assert r.throughput_tokens_per_sec >= 0
        assert r.throughput_requests_per_sec >= 0

    @pytest.mark.asyncio
    async def test_serve_mode(self, mock_backend: MockBackend):
        """Test serve benchmark mode with arrival pattern."""
        b = Benchmark(
            backend=VLLMConfig(model="test-model"),
            prompts=["Test 1", "Test 2"],
            mode="serve",
            concurrency=[1],
            arrival_rate=100.0,
            arrival_pattern="constant",
            warmup=0,
            runs=2,
        )

        results = await b.run(_backend=mock_backend)

        assert len(results.results) >= 1
        assert results.results[0].num_requests > 0

    @pytest.mark.asyncio
    async def test_warmup_executed(self, mock_backend: MockBackend):
        """Test that warmup runs are executed."""
        b = Benchmark(
            backend=VLLMConfig(model="test-model"),
            prompts=["Test"],
            concurrency=[1],
            warmup=2,
            runs=1,
        )

        initial_count = mock_backend.request_count
        await b.run(_backend=mock_backend)

        # Warmup + actual runs should result in multiple requests
        assert mock_backend.request_count > initial_count

    @pytest.mark.asyncio
    async def test_sampling_params(self, mock_backend: MockBackend):
        """Test custom sampling parameters are used."""
        b = Benchmark(
            backend=VLLMConfig(model="test-model"),
            prompts=["Test"],
            concurrency=[1],
            warmup=0,
            runs=1,
            sampling=SamplingParams(max_tokens=50, temperature=0.7),
        )

        results = await b.run(_backend=mock_backend)

        assert results.config["sampling"]["max_tokens"] == 50
        assert results.config["sampling"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_results_contain_config(self, mock_backend: MockBackend):
        """Test that results include configuration."""
        b = Benchmark(
            backend=VLLMConfig(model="my-model"),
            prompts=["Test"],
            concurrency=[1],
            warmup=0,
            runs=1,
        )

        results = await b.run(_backend=mock_backend)

        assert results.config is not None
        assert results.config["backend"]["type"] == "vllm"
        assert results.config["backend"]["model"] == "my-model"


class TestBenchmarkValidation:
    """Test Benchmark validation."""

    def test_requires_prompts_or_dataset(self):
        """Test that either prompts or dataset is required."""
        with pytest.raises(ValueError, match="Either 'prompts' or 'dataset'"):
            Benchmark(backend=VLLMConfig(model="test"))

    def test_cannot_have_both_prompts_and_dataset(self):
        """Test that both prompts and dataset is invalid."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            Benchmark(
                backend=VLLMConfig(model="test"),
                prompts=["test"],
                dataset="some-dataset",
            )

    def test_concurrency_must_be_nonempty(self):
        """Test that concurrency list cannot be empty."""
        with pytest.raises(ValueError):
            Benchmark(
                backend=VLLMConfig(model="test"),
                prompts=["test"],
                concurrency=[],
            )


class TestFileOutput:
    """Test file output via reporter functions."""

    @pytest.mark.asyncio
    async def test_json_output(self, mock_backend: MockBackend, tmp_path: Path):
        """Test write_json creates valid JSON."""
        b = Benchmark(
            backend=VLLMConfig(model="test-model"),
            prompts=["Test prompt"],
            concurrency=[1],
            warmup=0,
            runs=1,
        )

        results = await b.run(_backend=mock_backend)

        output_path = tmp_path / "results.json"
        write_json(results, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["model"] == "test-model"
        assert "results" in data

    @pytest.mark.asyncio
    async def test_csv_output(self, mock_backend: MockBackend, tmp_path: Path):
        """Test write_csv creates valid CSV."""
        b = Benchmark(
            backend=VLLMConfig(model="test-model"),
            prompts=["Test prompt"],
            concurrency=[1],
            warmup=0,
            runs=1,
        )

        results = await b.run(_backend=mock_backend)

        output_path = tmp_path / "results.csv"
        write_csv(results, output_path)

        assert output_path.exists()

        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) >= 2  # Header + data

        header = lines[0].lower()
        assert "concurrency" in header

    @pytest.mark.asyncio
    async def test_output_file_parameter(self, mock_backend: MockBackend, tmp_path: Path):
        """Test that output_file parameter saves results."""
        output_path = tmp_path / "auto_results.json"

        b = Benchmark(
            backend=VLLMConfig(model="test-model"),
            prompts=["Test"],
            concurrency=[1],
            warmup=0,
            runs=1,
            output_file=output_path,
        )

        await b.run(_backend=mock_backend)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["model"] == "test-model"
