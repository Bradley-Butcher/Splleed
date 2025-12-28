"""HuggingFace transformers tokenizer wrapper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import Tokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class TransformersTokenizer(Tokenizer):
    """Tokenizer using HuggingFace transformers AutoTokenizer."""

    def __init__(
        self,
        model: str,
        trust_remote_code: bool = False,
    ) -> None:
        """
        Initialize transformers tokenizer.

        Args:
            model: HuggingFace model name or path
            trust_remote_code: Whether to trust remote code for custom tokenizers

        Raises:
            ImportError: If transformers is not installed
        """
        self._model = model
        self._trust_remote_code = trust_remote_code
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._load()

    def _load(self) -> None:
        """Load the tokenizer from HuggingFace."""
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "transformers is required for token counting. "
                "Install with: pip install transformers"
            ) from e

        logger.info(f"Loading tokenizer for model: {self._model}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model,
            trust_remote_code=self._trust_remote_code,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return len(self._tokenizer.encode(text))

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._tokenizer.encode(text)

    @property
    def name(self) -> str:
        """Return tokenizer model name."""
        return self._model
