"""Tokenizer factory and utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .fallback import FallbackTokenizer

if TYPE_CHECKING:
    from splleed.config.base import TokenizerConfig

    from .base import Tokenizer

logger = logging.getLogger(__name__)

_TRANSFORMERS_AVAILABLE = False
try:
    import transformers  # noqa: F401  # type: ignore[import-not-found]

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def is_transformers_available() -> bool:
    """Check if transformers library is available."""
    return _TRANSFORMERS_AVAILABLE


def get_tokenizer(
    config: TokenizerConfig,
    default_model: str | None = None,
) -> Tokenizer | None:
    """
    Get a tokenizer based on configuration.

    Args:
        config: Tokenizer configuration
        default_model: Model name to use if not specified in config

    Returns:
        Tokenizer instance, or None if tokenizer is disabled
    """
    if not config.enabled:
        return None

    model = config.model or default_model
    if not model:
        logger.warning("No tokenizer model specified and no default available")
        return FallbackTokenizer()

    if _TRANSFORMERS_AVAILABLE:
        from .transformers_impl import TransformersTokenizer

        try:
            return TransformersTokenizer(
                model=model,
                trust_remote_code=config.trust_remote_code,
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model}: {e}")
            logger.warning("Falling back to character-based estimation")
            return FallbackTokenizer(model)
    else:
        logger.warning("transformers not installed, using fallback tokenizer")
        return FallbackTokenizer(model)


def count_tokens(text: str, tokenizer: Tokenizer | None) -> int | None:
    """
    Count tokens in text using tokenizer.

    Args:
        text: Text to count tokens in
        tokenizer: Tokenizer instance (or None)

    Returns:
        Token count, or None if tokenizer is not available
    """
    if tokenizer is None:
        return None
    return tokenizer.count_tokens(text)
