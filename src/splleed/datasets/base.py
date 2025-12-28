"""Base classes for dataset loading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SampleRequest:
    """A single sample from a dataset."""

    prompt: str
    expected_output_len: int | None = None
    prompt_tokens: int | None = None


class Dataset(ABC):
    """Abstract base class for benchmark datasets."""

    @abstractmethod
    def sample(self, n: int) -> list[SampleRequest]:
        """
        Sample n prompts from the dataset.

        Args:
            n: Number of samples to return

        Returns:
            List of SampleRequest objects
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        ...
