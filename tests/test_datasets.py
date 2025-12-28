"""Tests for dataset loaders."""

import json
from pathlib import Path

import pytest

from splleed.config.base import DatasetConfig
from splleed.datasets import (
    InlineDataset,
    JSONLDataset,
    RandomDataset,
    SampleRequest,
    get_dataset,
)


class TestSampleRequest:
    """Tests for SampleRequest dataclass."""

    def test_create_minimal(self):
        """Test creating with just prompt."""
        req = SampleRequest(prompt="Hello world")
        assert req.prompt == "Hello world"
        assert req.expected_output_len is None
        assert req.prompt_tokens is None

    def test_create_full(self):
        """Test creating with all fields."""
        req = SampleRequest(
            prompt="Hello world",
            expected_output_len=100,
            prompt_tokens=2,
        )
        assert req.prompt == "Hello world"
        assert req.expected_output_len == 100
        assert req.prompt_tokens == 2


class TestJSONLDataset:
    """Tests for JSONL dataset loader."""

    @pytest.fixture
    def sample_jsonl(self, tmp_path: Path) -> Path:
        """Create a sample JSONL file."""
        path = tmp_path / "prompts.jsonl"
        data = [
            {"prompt": "What is Python?", "expected_output_len": 100},
            {"prompt": "Explain machine learning.", "expected_output_len": 200},
            {"prompt": "What is AI?", "expected_output_len": 50},
        ]
        with path.open("w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return path

    def test_load_basic(self, sample_jsonl: Path):
        """Test basic loading."""
        dataset = JSONLDataset(sample_jsonl, shuffle=False)
        assert len(dataset) == 3

        samples = dataset.sample(3)
        assert len(samples) == 3
        assert samples[0].prompt == "What is Python?"
        assert samples[0].expected_output_len == 100

    def test_load_with_limit(self, sample_jsonl: Path):
        """Test loading with num_samples limit."""
        dataset = JSONLDataset(sample_jsonl, num_samples=2, shuffle=False)
        assert len(dataset) == 2

    def test_load_with_shuffle(self, sample_jsonl: Path):
        """Test loading with shuffle and seed."""
        dataset1 = JSONLDataset(sample_jsonl, shuffle=True, seed=42)
        dataset2 = JSONLDataset(sample_jsonl, shuffle=True, seed=42)
        dataset3 = JSONLDataset(sample_jsonl, shuffle=True, seed=99)

        # Same seed should produce same order
        samples1 = dataset1.sample(3)
        samples2 = dataset2.sample(3)
        assert [s.prompt for s in samples1] == [s.prompt for s in samples2]

        # Different seed should produce different order (usually)
        # We just verify it runs - actual order may vary
        assert len(dataset3.sample(3)) == 3

    def test_load_with_length_filter(self, sample_jsonl: Path):
        """Test loading with input length filter."""
        # Filter for short prompts (less than 12 chars)
        # "What is AI?" is 11 chars
        # "What is Python?" is 15 chars
        # "Explain machine learning." is 25 chars
        dataset = JSONLDataset(
            sample_jsonl,
            input_len_range=(0, 12),
            shuffle=False,
        )
        # Only "What is AI?" (11 chars) should match
        assert len(dataset) == 1
        assert dataset.sample(1)[0].prompt == "What is AI?"

    def test_oversampling(self, sample_jsonl: Path):
        """Test that sampling more than available repeats."""
        dataset = JSONLDataset(sample_jsonl, shuffle=False)
        samples = dataset.sample(10)
        assert len(samples) == 10
        # Should repeat the dataset
        assert samples[3].prompt == samples[0].prompt

    def test_file_not_found(self, tmp_path: Path):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            JSONLDataset(tmp_path / "nonexistent.jsonl")

    def test_invalid_json(self, tmp_path: Path):
        """Test error on invalid JSON."""
        path = tmp_path / "invalid.jsonl"
        path.write_text("not valid json\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            JSONLDataset(path)

    def test_missing_prompt_field(self, tmp_path: Path):
        """Test error on missing prompt field."""
        path = tmp_path / "missing.jsonl"
        path.write_text('{"text": "no prompt field"}\n')
        with pytest.raises(ValueError, match="Missing 'prompt' field"):
            JSONLDataset(path)

    def test_empty_file(self, tmp_path: Path):
        """Test error on empty file."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="No valid samples"):
            JSONLDataset(path)

    def test_blank_lines_ignored(self, tmp_path: Path):
        """Test that blank lines are ignored."""
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"prompt": "hello"}\n\n\n{"prompt": "world"}\n')
        dataset = JSONLDataset(path, shuffle=False)
        assert len(dataset) == 2


class TestRandomDataset:
    """Tests for random dataset generator."""

    def test_generate_basic(self):
        """Test basic generation."""
        dataset = RandomDataset(num_samples=10, input_len=50, output_len=100)
        assert len(dataset) == 10

        samples = dataset.sample(10)
        assert len(samples) == 10
        for s in samples:
            assert len(s.prompt) == 50
            assert s.expected_output_len == 100

    def test_reproducible_with_seed(self):
        """Test that same seed produces same prompts."""
        dataset1 = RandomDataset(num_samples=5, seed=42)
        dataset2 = RandomDataset(num_samples=5, seed=42)

        samples1 = dataset1.sample(5)
        samples2 = dataset2.sample(5)

        assert [s.prompt for s in samples1] == [s.prompt for s in samples2]

    def test_different_seeds_different_prompts(self):
        """Test that different seeds produce different prompts."""
        dataset1 = RandomDataset(num_samples=5, seed=42)
        dataset2 = RandomDataset(num_samples=5, seed=99)

        samples1 = dataset1.sample(5)
        samples2 = dataset2.sample(5)

        assert [s.prompt for s in samples1] != [s.prompt for s in samples2]

    def test_oversampling(self):
        """Test requesting more samples than generated."""
        dataset = RandomDataset(num_samples=5, seed=42)
        samples = dataset.sample(10)
        assert len(samples) == 10

    def test_lazy_generation(self):
        """Test that samples are generated lazily."""
        dataset = RandomDataset(num_samples=1000)
        # Should not generate until accessed
        assert dataset._samples is None
        _ = len(dataset)
        assert dataset._samples is not None


class TestInlineDataset:
    """Tests for inline dataset."""

    def test_basic(self):
        """Test basic inline dataset."""
        prompts = ["Hello", "World", "Test"]
        dataset = InlineDataset(prompts)

        assert len(dataset) == 3
        samples = dataset.sample(3)
        assert [s.prompt for s in samples] == prompts

    def test_with_output_len(self):
        """Test custom expected output length."""
        dataset = InlineDataset(["Hello"], expected_output_len=200)
        samples = dataset.sample(1)
        assert samples[0].expected_output_len == 200

    def test_shuffle(self):
        """Test shuffling."""
        prompts = ["A", "B", "C", "D", "E"]
        dataset1 = InlineDataset(prompts, shuffle=True, seed=42)
        dataset2 = InlineDataset(prompts, shuffle=True, seed=42)

        samples1 = dataset1.sample(5)
        samples2 = dataset2.sample(5)

        assert [s.prompt for s in samples1] == [s.prompt for s in samples2]

    def test_oversampling(self):
        """Test oversampling repeats."""
        dataset = InlineDataset(["A", "B"])
        samples = dataset.sample(5)
        assert len(samples) == 5
        assert samples[2].prompt == "A"  # Wrapped around

    def test_empty_prompts_error(self):
        """Test error on empty prompts list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InlineDataset([])

    def test_does_not_modify_input(self):
        """Test that input list is not modified."""
        prompts = ["A", "B", "C"]
        original = prompts.copy()
        InlineDataset(prompts, shuffle=True)
        assert prompts == original


class TestGetDataset:
    """Tests for get_dataset factory function."""

    @pytest.fixture
    def sample_jsonl(self, tmp_path: Path) -> Path:
        """Create a sample JSONL file."""
        path = tmp_path / "prompts.jsonl"
        data = [{"prompt": "Hello"}, {"prompt": "World"}]
        with path.open("w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return path

    def test_get_jsonl_dataset(self, sample_jsonl: Path):
        """Test creating JSONL dataset from config."""
        config = DatasetConfig(type="jsonl", path=sample_jsonl, num_samples=100)
        dataset = get_dataset(config)
        assert isinstance(dataset, JSONLDataset)
        assert len(dataset) == 2

    def test_get_random_dataset(self):
        """Test creating random dataset from config."""
        config = DatasetConfig(type="random", num_samples=50, output_len=100)
        dataset = get_dataset(config)
        assert isinstance(dataset, RandomDataset)
        assert len(dataset) == 50

    def test_get_inline_dataset(self):
        """Test creating inline dataset from config."""
        config = DatasetConfig(
            type="inline",
            prompts=["Hello", "World"],
            output_len=50,
        )
        dataset = get_dataset(config)
        assert isinstance(dataset, InlineDataset)
        assert len(dataset) == 2

    def test_jsonl_requires_path(self):
        """Test that JSONL requires path."""
        config = DatasetConfig(type="jsonl")
        with pytest.raises(ValueError, match="'path' is required"):
            get_dataset(config)

    def test_inline_requires_prompts(self):
        """Test that inline requires prompts."""
        config = DatasetConfig(type="inline")
        with pytest.raises(ValueError, match="'prompts' is required"):
            get_dataset(config)
