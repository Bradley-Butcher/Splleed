"""JSON result reporter."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from splleed.metrics.types import BenchmarkResults


def _serialize_value(obj: Any) -> Any:
    """Serialize values for JSON output."""
    # Check for custom to_dict() method first (e.g., EnvironmentInfo)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    elif hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize_value(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [_serialize_value(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def to_json(results: BenchmarkResults, indent: int = 2) -> str:
    """
    Convert benchmark results to JSON string.

    Args:
        results: Benchmark results
        indent: JSON indentation level

    Returns:
        JSON string
    """
    data = _serialize_value(results)

    # Remove raw_results if None to keep output clean
    for result in data.get("results", []):
        if result.get("raw_results") is None:
            del result["raw_results"]

    return json.dumps(data, indent=indent)


def write_json(results: BenchmarkResults, path: Path) -> None:
    """
    Write benchmark results to JSON file.

    Args:
        results: Benchmark results
        path: Output file path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(results))
