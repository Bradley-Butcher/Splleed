"""Tests for dataset analysis utilities."""

from splleed.datasets import DatasetStats, InlineDataset, analyze_dataset
from splleed.datasets.analyzer import DistributionStats, format_dataset_summary
from splleed.tokenizers import FallbackTokenizer


class TestDistributionStats:
    """Tests for DistributionStats dataclass."""

    def test_create(self):
        """Test creating distribution stats."""
        stats = DistributionStats(
            min=10,
            max=100,
            mean=50,
            median=45,
            std=20,
            p50=45,
            p95=90,
            p99=98,
            count=100,
        )
        assert stats.mean == 50
        assert stats.count == 100


class TestAnalyzeDataset:
    """Tests for analyze_dataset function."""

    def test_analyze_with_tokenizer(self):
        """Test dataset analysis with tokenizer."""
        dataset = InlineDataset(
            prompts=["short", "medium length", "this is a longer prompt"],
            expected_output_len=100,
        )
        tokenizer = FallbackTokenizer()
        stats = analyze_dataset(dataset, tokenizer=tokenizer)

        assert isinstance(stats, DatasetStats)
        assert stats.total_samples == 3
        assert stats.prompt_token_stats is not None
        assert stats.prompt_token_stats.count == 3

    def test_no_token_stats_without_tokenizer(self):
        """Test that token stats are None without tokenizer."""
        dataset = InlineDataset(prompts=["hello", "world"])
        stats = analyze_dataset(dataset, tokenizer=None)
        assert stats.prompt_token_stats is None

    def test_output_len_stats(self):
        """Test output length statistics when available."""
        dataset = InlineDataset(
            prompts=["test1", "test2", "test3"],
            expected_output_len=100,
        )
        stats = analyze_dataset(dataset)
        assert stats.output_len_stats is not None
        assert stats.output_len_stats.mean == 100

    def test_max_samples_limit(self):
        """Test that max_samples limits analysis."""
        prompts = [f"prompt {i}" for i in range(100)]
        dataset = InlineDataset(prompts=prompts)
        tokenizer = FallbackTokenizer()

        stats = analyze_dataset(dataset, tokenizer=tokenizer, max_samples=10)
        assert stats.total_samples == 100
        assert stats.prompt_token_stats is not None
        assert stats.prompt_token_stats.count == 10


class TestFormatDatasetSummary:
    """Tests for format_dataset_summary function."""

    def test_format_without_tokens(self):
        """Test formatting without token stats."""
        stats = DatasetStats(
            total_samples=100,
            prompt_token_stats=None,
            output_len_stats=None,
        )
        summary = format_dataset_summary(stats)
        assert "100 samples" in summary

    def test_format_with_tokens(self):
        """Test formatting with token stats."""
        stats = DatasetStats(
            total_samples=100,
            prompt_token_stats=DistributionStats(
                min=10, max=250, mean=125, median=100, std=50, p50=100, p95=200, p99=240, count=100
            ),
            output_len_stats=None,
        )
        summary = format_dataset_summary(stats)
        assert "100 samples" in summary
        assert "tokens" in summary
