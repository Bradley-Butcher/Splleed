"""Fallback tokenizer when transformers is not available."""

from __future__ import annotations

import logging

from .base import Tokenizer

logger = logging.getLogger(__name__)


class FallbackTokenizer(Tokenizer):
    """
    Fallback tokenizer using character-based estimation.

    Uses approximately 4 characters per token, which is typical for
    GPT-style BPE tokenizers on English text. This is a rough estimate
    and should only be used when proper tokenization is not available.
    """

    CHARS_PER_TOKEN = 4

    def __init__(self, model: str = "fallback") -> None:
        """
        Initialize fallback tokenizer.

        Args:
            model: Model name (used for identification only)
        """
        self._model = model
        logger.warning(
            "Using fallback tokenizer (character estimation). "
            "For accurate token counts, install transformers: pip install transformers"
        )

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count from character length.

        Uses ~4 characters per token heuristic.
        """
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def encode(self, text: str) -> list[int]:
        """Not supported by fallback tokenizer."""
        raise NotImplementedError(
            "Fallback tokenizer cannot encode text to token IDs. "
            "Install transformers for full tokenizer support."
        )

    @property
    def name(self) -> str:
        """Return tokenizer name with fallback indicator."""
        return f"{self._model} (fallback)"
