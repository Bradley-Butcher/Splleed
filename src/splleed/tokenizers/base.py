"""Abstract tokenizer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.

        Args:
            text: Input text to tokenize

        Returns:
            Number of tokens
        """
        ...

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return tokenizer name/model identifier."""
        ...
