"""Tests for synthetic prompt generation."""

from __future__ import annotations

from splleed import synthetic_prompts


def mock_tokenizer(s: str) -> list[int]:
    """Simple tokenizer: each word is 1 token."""
    return list(range(len(s.split())))


def mock_tokenizer_varied(s: str) -> list[int]:
    """Tokenizer where 'jumps' and 'over' are 2 tokens."""
    tokens = []
    for word in s.split():
        if word in ["jumps", "over"]:
            tokens.extend([1, 2])
        else:
            tokens.append(1)
    return tokens


class TestSyntheticPrompts:
    """Tests for synthetic_prompts function."""

    def test_exact_token_count(self):
        """Test that prompts have exact token count."""
        prompts = synthetic_prompts(10, 50, mock_tokenizer, seed=42)

        assert len(prompts) == 10
        for p in prompts:
            assert len(mock_tokenizer(p)) == 50

    def test_with_multi_token_words(self):
        """Test with tokenizer that has multi-token words."""
        prompts = synthetic_prompts(10, 100, mock_tokenizer_varied, seed=42)

        assert len(prompts) == 10
        for p in prompts:
            assert len(mock_tokenizer_varied(p)) == 100

    def test_reproducible_with_seed(self):
        """Test that same seed produces same prompts."""
        prompts1 = synthetic_prompts(5, 20, mock_tokenizer, seed=42)
        prompts2 = synthetic_prompts(5, 20, mock_tokenizer, seed=42)

        assert prompts1 == prompts2

    def test_different_seeds(self):
        """Test that different seeds produce different prompts."""
        prompts1 = synthetic_prompts(5, 20, mock_tokenizer, seed=42)
        prompts2 = synthetic_prompts(5, 20, mock_tokenizer, seed=99)

        assert prompts1 != prompts2

    def test_small_target(self):
        """Test with small token target."""
        prompts = synthetic_prompts(5, 3, mock_tokenizer, seed=42)

        assert len(prompts) == 5
        for p in prompts:
            assert len(mock_tokenizer(p)) == 3

    def test_large_target(self):
        """Test with large token target."""
        prompts = synthetic_prompts(3, 512, mock_tokenizer, seed=42)

        assert len(prompts) == 3
        for p in prompts:
            assert len(mock_tokenizer(p)) == 512
