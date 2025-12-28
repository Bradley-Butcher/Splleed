"""Data loading utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset  # type: ignore[import-not-found]


def load_hf_dataset(
    dataset_name: str,
    processor: Callable[[Dataset], Dataset] | None = None,
    prompt_column: str = "prompt",
    num_samples: int | None = None,
) -> list[str]:
    """
    Load prompts from a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "tatsu-lab/alpaca")
        processor: Optional function to process the dataset. Must return
                   a dataset with a 'prompt' column.
        prompt_column: Column name to extract prompts from (default: "prompt")
        num_samples: Optional limit on number of samples to return

    Returns:
        List of prompt strings

    Raises:
        ImportError: If the 'datasets' package is not installed
        ValueError: If the required column is not found
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace dataset support. "
            "Install it with: pip install splleed[hf]"
        ) from None

    ds = load_dataset(dataset_name, split="train")

    if processor is not None:
        ds = processor(ds)
        if "prompt" not in ds.column_names:
            raise ValueError(
                f"Processor must return dataset with 'prompt' column, "
                f"got columns: {ds.column_names}"
            )
        prompt_column = "prompt"
    elif prompt_column not in ds.column_names:
        raise ValueError(
            f"Column '{prompt_column}' not found in dataset. "
            f"Available columns: {ds.column_names}"
        )

    if num_samples is not None and num_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(num_samples))

    return list(ds[prompt_column])
