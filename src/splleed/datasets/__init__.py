"""Dataset loaders for benchmark prompts."""

from typing import Literal

from splleed.config.base import DatasetConfig

from .base import Dataset, SampleRequest
from .inline import InlineDataset
from .jsonl import JSONLDataset
from .random import RandomDataset

DATASETS: dict[str, type[Dataset]] = {
    "jsonl": JSONLDataset,
    "random": RandomDataset,
    "inline": InlineDataset,
}


def get_dataset(config: DatasetConfig) -> Dataset:
    """
    Create a dataset from configuration.

    Args:
        config: Dataset configuration

    Returns:
        Initialized dataset instance
    """
    dataset_type: Literal["sharegpt", "random", "jsonl", "inline"] = config.type

    if dataset_type == "jsonl":
        if config.path is None:
            raise ValueError("'path' is required for jsonl dataset")
        return JSONLDataset(
            path=config.path,
            num_samples=config.num_samples,
            input_len_range=config.input_len_range,
        )

    elif dataset_type == "random":
        return RandomDataset(
            num_samples=config.num_samples,
            input_len=config.input_len_range[0] if config.input_len_range else 100,
            output_len=config.output_len,
        )

    elif dataset_type == "inline":
        if not config.prompts:
            raise ValueError("'prompts' is required for inline dataset")
        return InlineDataset(
            prompts=config.prompts,
            expected_output_len=config.output_len,
        )

    elif dataset_type == "sharegpt":
        # ShareGPT is just JSONL with specific format
        if config.path is None:
            raise ValueError("'path' is required for sharegpt dataset")
        return JSONLDataset(
            path=config.path,
            num_samples=config.num_samples,
            input_len_range=config.input_len_range,
        )

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


__all__ = [
    "Dataset",
    "DATASETS",
    "get_dataset",
    "InlineDataset",
    "JSONLDataset",
    "RandomDataset",
    "SampleRequest",
]
