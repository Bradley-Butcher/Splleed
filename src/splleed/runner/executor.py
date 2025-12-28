"""Request executor with timing logic."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from splleed.metrics.types import RequestResult, Token

if TYPE_CHECKING:
    from splleed.backends.base import Backend, GenerateRequest

logger = logging.getLogger(__name__)


class RequestExecutor:
    """
    Executes generation requests and collects timing data.

    Collects raw token timestamps. Metrics (TTFT, ITL) are computed
    from timestamps by RequestResult properties.
    """

    async def execute(self, backend: Backend, request: GenerateRequest) -> RequestResult:
        """
        Execute a generation request and collect token timestamps.

        Args:
            backend: The inference backend to use
            request: Generation request parameters

        Returns:
            RequestResult with tokens and timing information
        """
        start_time = time.perf_counter()
        tokens: list[Token] = []
        error: str | None = None

        try:
            async for text in backend.generate_stream(request):
                tokens.append(Token(text=text, timestamp=time.perf_counter()))
        except Exception as e:
            logger.exception(f"Request failed: {e}")
            error = str(e)

        end_time = time.perf_counter()

        return RequestResult(
            success=error is None,
            start_time=start_time,
            end_time=end_time,
            tokens=tokens,
            error=error,
        )


async def execute_concurrent(
    executor: RequestExecutor,
    backend: Backend,
    requests: list[GenerateRequest],
    concurrency: int,
) -> list[RequestResult]:
    """
    Execute requests concurrently with semaphore limiting.

    Args:
        executor: Request executor for timing
        backend: The inference backend to use
        requests: List of generation requests
        concurrency: Maximum concurrent requests

    Returns:
        List of results (same order as requests)
    """
    import asyncio

    semaphore = asyncio.Semaphore(concurrency)

    async def run(req: GenerateRequest) -> RequestResult:
        async with semaphore:
            return await executor.execute(backend, req)

    return list(await asyncio.gather(*[run(r) for r in requests]))
