"""Random prompt dataset generator."""

from __future__ import annotations

import random
import string

from .base import Dataset, SampleRequest


class RandomDataset(Dataset):
    """
    Generate random prompts for benchmarking.

    Useful for testing without real data or for
    generating prompts of specific lengths.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        input_len: int = 100,
        output_len: int = 128,
        seed: int | None = None,
        vocab: str | None = None,
    ) -> None:
        """
        Initialize random dataset.

        Args:
            num_samples: Number of samples to generate
            input_len: Character length of each prompt
            output_len: Expected output length
            seed: Random seed for reproducibility
            vocab: Characters to use (default: lowercase + space)
        """
        self.num_samples = num_samples
        self.input_len = input_len
        self.output_len = output_len
        self.seed = seed
        self.vocab = vocab or (string.ascii_lowercase + " ")

        self._rng = random.Random(seed)
        self._samples: list[SampleRequest] | None = None

    def _generate_prompt(self) -> str:
        """Generate a single random prompt of exactly input_len characters."""
        # Generate random characters directly for exact length
        chars = self._rng.choices(self.vocab, k=self.input_len)
        return "".join(chars)

    def _ensure_generated(self) -> None:
        """Lazily generate samples."""
        if self._samples is None:
            self._samples = [
                SampleRequest(
                    prompt=self._generate_prompt(),
                    expected_output_len=self.output_len,
                )
                for _ in range(self.num_samples)
            ]

    def sample(self, n: int) -> list[SampleRequest]:
        """Sample n prompts from the dataset."""
        self._ensure_generated()
        assert self._samples is not None

        if n > len(self._samples):
            # Regenerate more samples if needed
            additional = n - len(self._samples)
            for _ in range(additional):
                self._samples.append(
                    SampleRequest(
                        prompt=self._generate_prompt(),
                        expected_output_len=self.output_len,
                    )
                )

        return self._samples[:n]

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        self._ensure_generated()
        assert self._samples is not None
        return len(self._samples)
