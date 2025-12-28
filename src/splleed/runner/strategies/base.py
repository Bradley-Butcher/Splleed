"""Base class for benchmark strategies."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from splleed.backends.base import Backend
    from splleed.config.base import BenchmarkConfig
    from splleed.datasets.base import Dataset
    from splleed.metrics.types import RequestResult
    from splleed.runner.executor import RequestExecutor


class BenchmarkStrategy(ABC):
    """
    Abstract base class for benchmark execution strategies.

    Different strategies handle request scheduling differently:
    - ThroughputStrategy: Send all requests at once
    - LatencyStrategy: Send one request at a time
    - ServeStrategy: Simulate realistic arrival patterns
    - StartupStrategy: Measure server startup time
    """

    @abstractmethod
    async def run(
        self,
        executor: "RequestExecutor",
        backend: "Backend",
        dataset: "Dataset",
        config: "BenchmarkConfig",
    ) -> list["RequestResult"]:
        """
        Execute the benchmark strategy.

        Args:
            executor: Request executor for timing individual requests
            backend: Inference backend to benchmark
            dataset: Dataset of prompts to use
            config: Benchmark configuration

        Returns:
            List of request results with timing information
        """
        ...
