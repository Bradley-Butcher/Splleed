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
    Executes single generation requests and handles all timing.

    This is the central place where TTFT and ITL are measured.
    Backends just yield tokens; the executor timestamps them.
    """

    async def execute(self, backend: Backend, request: GenerateRequest) -> RequestResult:
        """
        Execute a generation request and collect timing metrics.

        Args:
            backend: The inference backend to use
            request: Generation request parameters

        Returns:
            RequestResult with tokens and timing information
        """
        start_time = time.perf_counter()
        tokens: list[Token] = []
        error: str | None = None
        ttft: float | None = None
        itl: list[float] = []

        try:
            last_token_time: float | None = None

            async for text in backend.generate_stream(request):
                now = time.perf_counter()

                # Record TTFT on first token
                if ttft is None:
                    ttft = now - start_time

                # Record ITL for subsequent tokens
                if last_token_time is not None:
                    itl.append(now - last_token_time)

                tokens.append(Token(text=text, timestamp=now))
                last_token_time = now

        except Exception as e:
            logger.exception(f"Request failed: {e}")
            error = str(e)

        end_time = time.perf_counter()

        return RequestResult(
            success=error is None,
            start_time=start_time,
            end_time=end_time,
            tokens=tokens,
            ttft=ttft,
            itl=itl,
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
