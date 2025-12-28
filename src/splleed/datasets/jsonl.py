"""JSONL dataset loader."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .base import Dataset, SampleRequest


class JSONLDataset(Dataset):
    """
    Load prompts from a JSONL file.

    Expected format (one JSON object per line):
    {"prompt": "...", "expected_output_len": 100}

    The expected_output_len field is optional.
    """

    def __init__(
        self,
        path: Path | str,
        num_samples: int | None = None,
        input_len_range: tuple[int, int] | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Initialize JSONL dataset.

        Args:
            path: Path to JSONL file
            num_samples: Maximum number of samples to load (None = all)
            input_len_range: Filter by prompt character length (min, max)
            shuffle: Whether to shuffle samples
            seed: Random seed for shuffling
        """
        self.path = Path(path)
        self.num_samples = num_samples
        self.input_len_range = input_len_range
        self.shuffle = shuffle
        self.seed = seed

        self._samples: list[SampleRequest] = []
        self._load()

    def _load(self) -> None:
        """Load and parse the JSONL file."""
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        samples: list[SampleRequest] = []

        with self.path.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

                if "prompt" not in data:
                    raise ValueError(f"Missing 'prompt' field on line {line_num}")

                prompt = data["prompt"]

                # Filter by length if specified
                if self.input_len_range:
                    min_len, max_len = self.input_len_range
                    if not (min_len <= len(prompt) <= max_len):
                        continue

                samples.append(
                    SampleRequest(
                        prompt=prompt,
                        expected_output_len=data.get("expected_output_len"),
                        prompt_tokens=data.get("prompt_tokens"),
                    )
                )

        if not samples:
            raise ValueError(f"No valid samples found in {self.path}")

        # Shuffle if requested
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(samples)

        # Limit samples if requested
        if self.num_samples is not None:
            samples = samples[: self.num_samples]

        self._samples = samples

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
