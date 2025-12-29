"""Synthetic prompt generation for controlled benchmarks."""

from __future__ import annotations

import random
from collections.abc import Callable


def synthetic_prompts(
    count: int,
    target_tokens: int,
    tokenizer: Callable[[str], list[int]],
    seed: int | None = None,
) -> list[str]:
    """
    Generate prompts of exact token length for controlled benchmarks.

    Args:
        count: Number of prompts to generate.
        target_tokens: Exact token count for each prompt.
        tokenizer: Function that takes a string and returns token IDs.
        seed: Random seed for reproducibility.

    Returns:
        List of prompts, each tokenizing to exactly target_tokens.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        >>> prompts = synthetic_prompts(100, 512, tok.encode, seed=42)
    """
    rng = random.Random(seed)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "a"]

    # Get token counts for each word (with leading space, as they'll appear in joined text)
    word_tokens = {w: len(tokenizer(f" {w}")) for w in words}

    # Find a 1-token filler (try "a" first, then "the", then any)
    filler = None
    for candidate in ["a", "the"]:
        if word_tokens.get(candidate) == 1:
            filler = candidate
            break
    if filler is None:
        filler = next((w for w, c in word_tokens.items() if c == 1), None)
    if filler is None:
        raise ValueError("No 1-token word found in word list")

    max_word_tokens = max(word_tokens.values())

    prompts = []
    for _ in range(count):
        prompt_words = []
        current = 0

        # Add random words while there's room for the largest word
        while current + max_word_tokens <= target_tokens:
            w = rng.choice(words)
            prompt_words.append(w)
            current += word_tokens[w]

        # Fill remainder exactly with 1-token filler
        prompt_words.extend([filler] * (target_tokens - current))
        prompts.append(" ".join(prompt_words))

    return prompts
