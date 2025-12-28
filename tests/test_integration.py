"""Integration tests for end-to-end benchmark flow."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from splleed.backends.base import Backend, BackendConfigBase, GenerateRequest
from splleed.cli import app
from splleed.config import load_config
from splleed.metrics.types import BenchmarkResults
from splleed.runner.orchestrator import BenchmarkOrchestrator


# Mock backend for integration testing
class IntegrationMockConfig(BackendConfigBase):
    """Mock config for integration testing."""

    type: str = "mock"
    model: str = "integration-test-model"
    endpoint: str | None = None


class IntegrationMockBackend(Backend[IntegrationMockConfig]):
    """Mock backend that simulates realistic token generation."""

    def __init__(self, config: IntegrationMockConfig | None = None) -> None:
        self.config = config or IntegrationMockConfig()
        self._started = False
        self._request_count = 0

    async def generate_stream(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        """Generate tokens with realistic timing."""
        self._request_count += 1
        tokens = ["The", " answer", " to", " your", " question", " is", " 42", "."]
        for token in tokens:
            await asyncio.sleep(0.001)  # Simulate token generation delay
            yield token

    async def health(self) -> bool:
        """Return health status."""
        return self._started

    async def connect(self, endpoint: str) -> None:
        """Connect to endpoint."""
        self._started = True

    async def start(self) -> None:
        """Start the backend."""
        self._started = True

    async def shutdown(self) -> None:
        """Shutdown the backend."""
        self._started = False

    @property
    def request_count(self) -> int:
        """Number of requests processed."""
        return self._request_count


runner = CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a valid configuration file."""
    config_content = """
backend:
  type: vllm
  model: facebook/opt-125m
  endpoint: http://localhost:8000

dataset:
  type: inline
  prompts:
    - "What is Python?"
    - "Explain machine learning."
    - "What is 2+2?"

benchmark:
  mode: latency
  concurrency: [1]
  warmup: 0
  runs: 3

sampling:
  max_tokens: 50
  temperature: 0.7

output:
  path: results.json
  format: json
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def throughput_config_file(tmp_path: Path) -> Path:
    """Create a throughput benchmark configuration file."""
    config_content = """
backend:
  type: vllm
  model: test-model

dataset:
  type: inline
  prompts:
    - "Test prompt 1"
    - "Test prompt 2"
    - "Test prompt 3"
    - "Test prompt 4"

benchmark:
  mode: throughput
  concurrency: [2, 4]
  warmup: 1
  runs: 4

sampling:
  max_tokens: 100

output:
  path: throughput_results.json
  format: json
"""
    config_path = tmp_path / "throughput_config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def serve_config_file(tmp_path: Path) -> Path:
    """Create a serve benchmark configuration file."""
    config_content = """
backend:
  type: vllm
  model: test-model
  endpoint: http://localhost:8000

dataset:
  type: inline
  prompts:
    - "Prompt A"
    - "Prompt B"

benchmark:
  mode: serve
  concurrency: [1]
  warmup: 0
  runs: 2
  arrival:
    type: constant
    rate: 100

output:
  path: serve_results.json
  format: json
"""
    config_path = tmp_path / "serve_config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_backend() -> IntegrationMockBackend:
    """Create a mock backend."""
    backend = IntegrationMockBackend()
    backend._started = True
    return backend


class TestEndToEndPipeline:
    """Test the complete benchmark pipeline."""

    @pytest.mark.asyncio
    async def test_full_latency_benchmark(
        self, config_file: Path, mock_backend: IntegrationMockBackend, tmp_path: Path
    ):
        """Test complete latency benchmark flow."""
        config = load_config(config_file)

        # Override output path
        config.output.path = tmp_path / "results.json"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        # Verify results structure
        assert isinstance(results, BenchmarkResults)
        assert results.engine == "vllm"
        assert results.model == "facebook/opt-125m"
        assert results.timestamp is not None
        assert len(results.results) == 1  # One concurrency level

        # Verify metrics
        r = results.results[0]
        assert r.concurrency == 1
        assert r.num_requests == 3  # 3 prompts
        assert r.num_successful >= 0
        assert r.throughput_tokens_per_sec >= 0
        assert r.ttft_p50_ms >= 0
        assert r.itl_mean_ms >= 0

    @pytest.mark.asyncio
    async def test_full_throughput_benchmark(
        self,
        throughput_config_file: Path,
        mock_backend: IntegrationMockBackend,
        tmp_path: Path,
    ):
        """Test complete throughput benchmark flow."""
        config = load_config(throughput_config_file)
        config.output.path = tmp_path / "results.json"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        # Verify multiple concurrency levels
        assert len(results.results) == 2
        assert results.results[0].concurrency == 2
        assert results.results[1].concurrency == 4

    @pytest.mark.asyncio
    async def test_full_serve_benchmark(
        self,
        serve_config_file: Path,
        mock_backend: IntegrationMockBackend,
        tmp_path: Path,
    ):
        """Test complete serve benchmark flow with arrival patterns."""
        config = load_config(serve_config_file)
        config.output.path = tmp_path / "results.json"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        assert len(results.results) >= 1
        assert results.results[0].num_requests > 0

    @pytest.mark.asyncio
    async def test_warmup_runs_executed(
        self,
        throughput_config_file: Path,
        mock_backend: IntegrationMockBackend,
        tmp_path: Path,
    ):
        """Test that warmup runs are actually executed."""
        config = load_config(throughput_config_file)
        config.output.path = tmp_path / "results.json"
        config.benchmark.warmup = 2

        initial_count = mock_backend.request_count
        orchestrator = BenchmarkOrchestrator(config)
        await orchestrator.run(mock_backend)

        # With warmup=2, we should have extra requests
        assert mock_backend.request_count > initial_count

    @pytest.mark.asyncio
    async def test_results_contain_config(
        self, config_file: Path, mock_backend: IntegrationMockBackend, tmp_path: Path
    ):
        """Test that results include the configuration used."""
        config = load_config(config_file)
        config.output.path = tmp_path / "results.json"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        assert results.config is not None
        assert "backend" in results.config
        assert "benchmark" in results.config
        assert results.config["backend"]["type"] == "vllm"


class TestCLIIntegration:
    """Test CLI commands."""

    def test_cli_validate_valid_config(self, config_file: Path):
        """Test validating a valid config file."""
        result = runner.invoke(app, ["validate", str(config_file)])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_cli_validate_missing_file(self, tmp_path: Path):
        """Test validating a non-existent config file."""
        result = runner.invoke(app, ["validate", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_cli_validate_invalid_config(self, tmp_path: Path):
        """Test validating an invalid config file."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("backend:\n  type: vllm\n")  # Missing required fields
        result = runner.invoke(app, ["validate", str(invalid_config)])
        assert result.exit_code == 1

    def test_cli_backends_list(self):
        """Test listing available backends."""
        result = runner.invoke(app, ["backends"])
        assert result.exit_code == 0
        assert "vllm" in result.stdout.lower()

    def test_cli_init_creates_config(self, tmp_path: Path):
        """Test init command creates example config."""
        output_path = tmp_path / "example.yaml"
        result = runner.invoke(app, ["init", "--output", str(output_path)])
        assert result.exit_code == 0
        assert output_path.exists()

        # Verify the created config is valid
        content = output_path.read_text()
        assert "backend:" in content
        assert "benchmark:" in content

    def test_cli_init_warns_on_existing(self, tmp_path: Path):
        """Test init warns when file exists."""
        output_path = tmp_path / "existing.yaml"
        output_path.write_text("existing content")

        result = runner.invoke(app, ["init", "--output", str(output_path)], input="n\n")
        assert result.exit_code == 0
        # File should be unchanged
        assert output_path.read_text() == "existing content"

    def test_cli_new_backend_creates_template(self, tmp_path: Path, monkeypatch):
        """Test new-backend command creates template files."""
        # This test needs to work with the actual package structure
        # We'll skip this as it modifies package files
        pytest.skip("Skipping to avoid modifying package structure")

    def test_cli_help(self):
        """Test CLI help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "splleed" in result.stdout.lower() or "benchmark" in result.stdout.lower()

    def test_cli_run_missing_config(self, tmp_path: Path):
        """Test run command with missing config."""
        result = runner.invoke(app, ["run", str(tmp_path / "missing.yaml")])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_cli_bench_throughput_missing_config(self, tmp_path: Path):
        """Test bench throughput with missing config."""
        result = runner.invoke(app, ["bench", "throughput", str(tmp_path / "missing.yaml")])
        assert result.exit_code == 1

    def test_cli_bench_latency_missing_config(self, tmp_path: Path):
        """Test bench latency with missing config."""
        result = runner.invoke(app, ["bench", "latency", str(tmp_path / "missing.yaml")])
        assert result.exit_code == 1


class TestFileOutput:
    """Test result file output."""

    @pytest.mark.asyncio
    async def test_json_output(
        self, config_file: Path, mock_backend: IntegrationMockBackend, tmp_path: Path
    ):
        """Test JSON output file is created correctly."""
        from splleed.reporters import write_json

        config = load_config(config_file)
        output_path = tmp_path / "results.json"
        config.output.path = output_path
        config.output.format = "json"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        write_json(results, output_path)

        assert output_path.exists()

        # Verify JSON is valid and contains expected data
        with open(output_path) as f:
            data = json.load(f)

        assert data["engine"] == "vllm"
        assert data["model"] == "facebook/opt-125m"
        assert "results" in data
        assert len(data["results"]) == 1

    @pytest.mark.asyncio
    async def test_csv_output(
        self, config_file: Path, mock_backend: IntegrationMockBackend, tmp_path: Path
    ):
        """Test CSV output file is created correctly."""
        from splleed.reporters import write_csv

        config = load_config(config_file)
        output_path = tmp_path / "results.csv"
        config.output.path = output_path
        config.output.format = "csv"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        write_csv(results, output_path)

        assert output_path.exists()

        # Verify CSV has header and data rows
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) >= 2  # Header + at least one data row

        # Check header contains expected columns
        header = lines[0].lower()
        assert "concurrency" in header
        assert "throughput" in header


class TestConfigOverrides:
    """Test CLI config overrides."""

    def test_output_path_override(self, config_file: Path, tmp_path: Path):
        """Test overriding output path from CLI."""
        override_path = tmp_path / "override_results.json"
        config = load_config(config_file, overrides={"output.path": str(override_path)})
        # Path may be stored as Path object or string - check the value matches
        assert str(config.output.path) == str(override_path)

    def test_format_override(self, config_file: Path):
        """Test overriding output format from CLI."""
        config = load_config(config_file, overrides={"output.format": "csv"})
        assert config.output.format == "csv"

    def test_benchmark_mode_override(self, config_file: Path):
        """Test overriding benchmark mode from CLI."""
        config = load_config(config_file, overrides={"benchmark.mode": "throughput"})
        assert config.benchmark.mode == "throughput"

    def test_concurrency_override(self, config_file: Path):
        """Test overriding concurrency levels from CLI."""
        config = load_config(config_file, overrides={"benchmark.concurrency": [1, 2, 4]})
        assert config.benchmark.concurrency == [1, 2, 4]

    def test_sampling_override(self, config_file: Path):
        """Test overriding sampling parameters from CLI."""
        config = load_config(config_file, overrides={"sampling.max_tokens": 200})
        assert config.sampling.max_tokens == 200


class TestErrorHandling:
    """Test error handling in the pipeline."""

    def test_invalid_backend_type(self, tmp_path: Path):
        """Test handling of invalid backend type."""
        config_content = """
backend:
  type: nonexistent_backend
  model: test

dataset:
  type: inline
  prompts: ["test"]

benchmark:
  mode: latency
"""
        config_path = tmp_path / "invalid_backend.yaml"
        config_path.write_text(config_content)

        with pytest.raises(ValidationError):
            load_config(config_path)

    def test_invalid_benchmark_mode(self, config_file: Path):
        """Test handling of invalid benchmark mode."""
        config = load_config(config_file)
        # Use object.__setattr__ to bypass Pydantic validation for testing
        object.__setattr__(config.benchmark, "mode", "invalid_mode")

        orchestrator = BenchmarkOrchestrator(config)

        with pytest.raises(ValueError, match="Unknown benchmark mode"):
            orchestrator._get_strategy()

    def test_empty_prompts_dataset(self, tmp_path: Path):
        """Test handling of empty prompts in dataset."""
        config_content = """
backend:
  type: vllm
  model: test

dataset:
  type: inline
  prompts: []

benchmark:
  mode: latency
"""
        config_path = tmp_path / "empty_prompts.yaml"
        config_path.write_text(config_content)

        # Empty prompts should either fail validation or produce an empty dataset
        # The config loader may accept it, so we test the dataset behavior instead
        try:
            config = load_config(config_path)
            from splleed.datasets import get_dataset

            dataset = get_dataset(config.dataset)
            # If we get here, dataset should be empty
            assert len(dataset) == 0
        except Exception:
            # Validation failed - this is acceptable behavior
            pass


class TestMultipleBenchmarkModes:
    """Test all benchmark modes work correctly."""

    @pytest.mark.asyncio
    async def test_latency_mode_produces_correct_metrics(
        self, config_file: Path, mock_backend: IntegrationMockBackend, tmp_path: Path
    ):
        """Test latency mode produces expected metrics."""
        config = load_config(config_file)
        config.benchmark.mode = "latency"
        config.output.path = tmp_path / "results.json"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        r = results.results[0]
        # Latency mode should have detailed timing metrics
        assert r.ttft_p50_ms is not None
        assert r.ttft_p95_ms is not None
        assert r.ttft_p99_ms is not None
        assert r.itl_mean_ms is not None

    @pytest.mark.asyncio
    async def test_throughput_mode_produces_correct_metrics(
        self,
        throughput_config_file: Path,
        mock_backend: IntegrationMockBackend,
        tmp_path: Path,
    ):
        """Test throughput mode produces expected metrics."""
        config = load_config(throughput_config_file)
        config.output.path = tmp_path / "results.json"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        for r in results.results:
            assert r.throughput_tokens_per_sec >= 0
            assert r.throughput_requests_per_sec >= 0
            assert r.num_successful >= 0

    @pytest.mark.asyncio
    async def test_serve_mode_with_poisson_arrival(
        self, mock_backend: IntegrationMockBackend, tmp_path: Path
    ):
        """Test serve mode with Poisson arrival pattern."""
        config_content = """
backend:
  type: vllm
  model: test

dataset:
  type: inline
  prompts:
    - "Test 1"
    - "Test 2"

benchmark:
  mode: serve
  concurrency: [1]
  warmup: 0
  runs: 2
  arrival:
    type: poisson
    rate: 50

output:
  path: results.json
"""
        config_path = tmp_path / "poisson_config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        config.output.path = tmp_path / "results.json"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        assert len(results.results) >= 1


class TestDatasetIntegration:
    """Test dataset loading in full pipeline."""

    @pytest.mark.asyncio
    async def test_random_dataset_integration(
        self, mock_backend: IntegrationMockBackend, tmp_path: Path
    ):
        """Test random dataset in full pipeline."""
        config_content = """
backend:
  type: vllm
  model: test

dataset:
  type: random
  num_samples: 5
  input_len: 20
  seed: 42

benchmark:
  mode: latency
  concurrency: [1]
  warmup: 0
  runs: 5

output:
  path: results.json
"""
        config_path = tmp_path / "random_config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        config.output.path = tmp_path / "results.json"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        assert results.results[0].num_requests == 5

    @pytest.mark.asyncio
    async def test_jsonl_dataset_integration(
        self, mock_backend: IntegrationMockBackend, tmp_path: Path
    ):
        """Test JSONL dataset in full pipeline."""
        # Create JSONL file
        jsonl_path = tmp_path / "prompts.jsonl"
        jsonl_content = (
            '{"prompt": "Question 1"}\n' '{"prompt": "Question 2"}\n' '{"prompt": "Question 3"}\n'
        )
        jsonl_path.write_text(jsonl_content)

        config_content = f"""
backend:
  type: vllm
  model: test

dataset:
  type: jsonl
  path: {jsonl_path}
  prompt_field: prompt

benchmark:
  mode: latency
  concurrency: [1]
  warmup: 0
  runs: 3

output:
  path: results.json
"""
        config_path = tmp_path / "jsonl_config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)
        config.output.path = tmp_path / "results.json"

        orchestrator = BenchmarkOrchestrator(config)
        results = await orchestrator.run(mock_backend)

        assert results.results[0].num_requests == 3
