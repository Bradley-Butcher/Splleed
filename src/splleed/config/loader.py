"""Configuration loading and merging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from splleed.backends import BackendConfig

from .base import (
    SplleedConfig,
)


class FullConfig(SplleedConfig):
    """Full configuration including backend (resolved from union)."""

    backend: BackendConfig


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open() as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    return data


def merge_cli_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Merge CLI overrides into configuration.

    Supports dotted paths for nested keys (e.g., "backend.model").

    Args:
        config: Base configuration dictionary
        overrides: CLI override dictionary (dotted keys)

    Returns:
        Merged configuration
    """
    result = config.copy()

    for key, value in overrides.items():
        if value is None:
            continue

        parts = key.split(".")
        current = result

        # Navigate to nested location
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value
        final_key = parts[-1]

        # Handle list values (e.g., concurrency="1,2,4")
        if isinstance(value, str) and "," in value:
            try:
                value = [int(x.strip()) for x in value.split(",")]
            except ValueError:
                value = [x.strip() for x in value.split(",")]

        current[final_key] = value

    return result


def load_config(
    path: Path,
    overrides: dict[str, Any] | None = None,
) -> FullConfig:
    """
    Load and validate configuration from YAML with optional overrides.

    Args:
        path: Path to YAML configuration file
        overrides: Optional CLI overrides (dotted keys)

    Returns:
        Validated FullConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If configuration is invalid
    """
    data = load_yaml(path)

    if overrides:
        data = merge_cli_overrides(data, overrides)

    return FullConfig(**data)


def validate_config(data: dict[str, Any]) -> list[str]:
    """
    Validate configuration and return list of errors.

    Args:
        data: Configuration dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    try:
        FullConfig(**data)
    except ValidationError as e:
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"{loc}: {msg}")

    return errors


def generate_example_config() -> str:
    """
    Generate an example configuration file.

    Returns:
        YAML string with example configuration
    """
    example = {
        "backend": {
            "type": "vllm",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "tensor_parallel": 1,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.9,
        },
        "dataset": {
            "type": "jsonl",
            "path": "prompts/sample.jsonl",
            "num_samples": 100,
        },
        "benchmark": {
            "mode": "serve",
            "concurrency": [1, 2, 4, 8],
            "warmup": 2,
            "runs": 100,
            "arrival": {
                "type": "poisson",
                "rate": 10.0,
            },
        },
        "sampling": {
            "max_tokens": 256,
            "temperature": 0.0,
        },
        "output": {
            "path": "results.json",
            "format": "json",
        },
    }

    return yaml.dump(example, default_flow_style=False, sort_keys=False)
