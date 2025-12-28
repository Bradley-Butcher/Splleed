"""Dataset factory for creating datasets from configuration."""

from __future__ import annotations

from splleed.config.base import DatasetConfig

from .base import Dataset
from .inline import InlineDataset
from .jsonl import JSONLDataset
from .random import RandomDataset


def get_dataset(config: DatasetConfig) -> Dataset:
    """
    Create a dataset from configuration.

    Args:
        config: Dataset configuration

    Returns:
        Initialized dataset instance
    """
    match config.type:
        case "jsonl":
            if config.path is None:
                raise ValueError("'path' is required for jsonl dataset")
            return JSONLDataset(
                path=config.path,
                num_samples=config.num_samples,
                input_len_range=config.input_len_range,
            )

        case "random":
            return RandomDataset(
                num_samples=config.num_samples,
                input_len=config.input_len_range[0] if config.input_len_range else 100,
                output_len=config.output_len,
            )

        case "inline":
            if not config.prompts:
                raise ValueError("'prompts' is required for inline dataset")
            return InlineDataset(
                prompts=config.prompts,
                expected_output_len=config.output_len,
            )

        case _:
            raise ValueError(f"Unknown dataset type: {config.type}")
