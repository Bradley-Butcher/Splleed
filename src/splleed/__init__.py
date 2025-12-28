"""Splleed - LLM inference benchmarking harness with pluggable backends."""

from splleed.api import Benchmark
from splleed.backends import TGIConfig, VLLMConfig
from splleed.config.base import SamplingParams
from splleed.metrics.types import BenchmarkResults

__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "BenchmarkResults",
    "SamplingParams",
    "TGIConfig",
    "VLLMConfig",
]
