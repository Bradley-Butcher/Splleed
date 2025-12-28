"""Tests for result reporters."""

import json
from pathlib import Path

import pytest

from splleed.metrics.types import BenchmarkResults, ConcurrencyResult
from splleed.reporters import format_summary, to_csv, to_json, write_csv, write_json


@pytest.fixture
def sample_results() -> BenchmarkResults:
    """Create sample benchmark results for testing."""
    return BenchmarkResults(
        engine="vllm",
        model="test-model",
        timestamp="2025-01-15T10:30:00Z",
        gpu="NVIDIA A100",
        config={"test": "config"},
        results=[
            ConcurrencyResult(
                concurrency=1,
                num_requests=100,
                num_successful=95,
                num_failed=5,
                total_time_sec=10.0,
                throughput_requests_per_sec=10.0,
                throughput_tokens_per_sec=500.0,
                ttft_p50_ms=50.0,
                ttft_p95_ms=80.0,
                ttft_p99_ms=100.0,
                ttft_mean_ms=55.0,
                itl_p50_ms=10.0,
                itl_p95_ms=15.0,
                itl_p99_ms=20.0,
                itl_mean_ms=11.0,
                tpot_mean_ms=12.0,
                e2el_p50_ms=500.0,
                e2el_p95_ms=800.0,
                e2el_p99_ms=1000.0,
                e2el_mean_ms=550.0,
                goodput_pct=90.0,
            ),
            ConcurrencyResult(
                concurrency=4,
                num_requests=100,
                num_successful=90,
                num_failed=10,
                total_time_sec=5.0,
                throughput_requests_per_sec=20.0,
                throughput_tokens_per_sec=1000.0,
                ttft_p50_ms=100.0,
                ttft_p95_ms=150.0,
                ttft_p99_ms=200.0,
                ttft_mean_ms=110.0,
                itl_p50_ms=15.0,
                itl_p95_ms=20.0,
                itl_p99_ms=25.0,
                itl_mean_ms=16.0,
                tpot_mean_ms=17.0,
                e2el_p50_ms=600.0,
                e2el_p95_ms=900.0,
                e2el_p99_ms=1200.0,
                e2el_mean_ms=650.0,
                goodput_pct=75.0,
            ),
        ],
    )


class TestJSONReporter:
    """Tests for JSON reporter."""

    def test_to_json_basic(self, sample_results: BenchmarkResults):
        """Test basic JSON conversion."""
        output = to_json(sample_results)

        # Should be valid JSON
        data = json.loads(output)

        assert data["engine"] == "vllm"
        assert data["model"] == "test-model"
        assert data["gpu"] == "NVIDIA A100"
        assert len(data["results"]) == 2

    def test_to_json_results_structure(self, sample_results: BenchmarkResults):
        """Test that result structure is correct."""
        data = json.loads(to_json(sample_results))

        result = data["results"][0]
        assert result["concurrency"] == 1
        assert result["num_requests"] == 100
        assert result["throughput_tokens_per_sec"] == 500.0
        assert result["ttft_p50_ms"] == 50.0
        assert result["goodput_pct"] == 90.0

    def test_write_json(self, sample_results: BenchmarkResults, tmp_path: Path):
        """Test writing JSON to file."""
        output_path = tmp_path / "results.json"
        write_json(sample_results, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["engine"] == "vllm"

    def test_write_json_creates_directories(self, sample_results: BenchmarkResults, tmp_path: Path):
        """Test that write_json creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "results.json"
        write_json(sample_results, output_path)

        assert output_path.exists()


class TestCSVReporter:
    """Tests for CSV reporter."""

    def test_to_csv_basic(self, sample_results: BenchmarkResults):
        """Test basic CSV conversion."""
        output = to_csv(sample_results)

        lines = output.strip().split("\n")
        assert len(lines) == 3  # Header + 2 data rows

        # Check header
        header = lines[0]
        assert "concurrency" in header
        assert "throughput_tokens_per_sec" in header
        assert "ttft_p50_ms" in header

    def test_to_csv_data_rows(self, sample_results: BenchmarkResults):
        """Test CSV data rows."""
        output = to_csv(sample_results)
        lines = output.strip().split("\n")

        # First data row
        assert "1," in lines[1]  # concurrency=1
        assert "500.00" in lines[1]  # throughput

        # Second data row
        assert "4," in lines[2]  # concurrency=4
        assert "1000.00" in lines[2]  # throughput

    def test_write_csv(self, sample_results: BenchmarkResults, tmp_path: Path):
        """Test writing CSV to file."""
        output_path = tmp_path / "results.csv"
        write_csv(sample_results, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "concurrency" in content

    def test_to_csv_handles_none_goodput(self, sample_results: BenchmarkResults):
        """Test that None goodput is handled."""
        sample_results.results[0].goodput_pct = None
        output = to_csv(sample_results)

        # Should not error
        assert "concurrency" in output


class TestConsoleReporter:
    """Tests for console reporter."""

    def test_format_summary(self, sample_results: BenchmarkResults):
        """Test summary formatting."""
        summary = format_summary(sample_results)

        assert "Engine: vllm" in summary
        assert "Model: test-model" in summary
        assert "Concurrency 1:" in summary
        assert "Concurrency 4:" in summary
        assert "500.0 tokens/s" in summary

    def test_format_summary_multiple_concurrency(self, sample_results: BenchmarkResults):
        """Test that all concurrency levels are included."""
        summary = format_summary(sample_results)

        assert "Concurrency 1:" in summary
        assert "Concurrency 4:" in summary


class TestReporterEdgeCases:
    """Tests for edge cases in reporters."""

    def test_empty_results(self):
        """Test with no concurrency results."""
        results = BenchmarkResults(
            engine="test",
            model="test",
            timestamp="2025-01-15T10:30:00Z",
            gpu=None,
            config={},
            results=[],
        )

        # Should not error
        json_out = to_json(results)
        assert json.loads(json_out)["results"] == []

        csv_out = to_csv(results)
        assert "concurrency" in csv_out  # Header only

        summary = format_summary(results)
        assert "Engine: test" in summary

    def test_no_gpu(self, sample_results: BenchmarkResults):
        """Test with no GPU info."""
        sample_results.gpu = None
        data = json.loads(to_json(sample_results))
        assert data["gpu"] is None

    def test_raw_results_excluded_from_json(self, sample_results: BenchmarkResults):
        """Test that None raw_results are excluded from JSON."""
        sample_results.results[0].raw_results = None
        data = json.loads(to_json(sample_results))

        # raw_results key should not exist when None
        assert "raw_results" not in data["results"][0]
