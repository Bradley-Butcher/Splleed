"""Inline prompt dataset."""

from __future__ import annotations

import random

from .base import Dataset, SampleRequest


class InlineDataset(Dataset):
    """
    Dataset from inline prompt list.

    Useful for quick testing or when prompts are
    provided directly in configuration.
    """

    def __init__(
        self,
        prompts: list[str],
        expected_output_len: int = 128,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Initialize inline dataset.

        Args:
            prompts: List of prompt strings
            expected_output_len: Expected output length for all prompts
            shuffle: Whether to shuffle prompts
            seed: Random seed for shuffling
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")

        self.prompts = prompts.copy()
        self.expected_output_len = expected_output_len

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.prompts)

        self._samples = [
            SampleRequest(prompt=p, expected_output_len=expected_output_len) for p in self.prompts
        ]

    def sample(self, n: int) -> list[SampleRequest]:
        """Sample n prompts from the dataset."""
        if n > len(self._samples):
            # Allow oversampling by repeating
            repeats = (n // len(self._samples)) + 1
            pool = self._samples * repeats
            return pool[:n]
        return self._samples[:n]

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._samples)
