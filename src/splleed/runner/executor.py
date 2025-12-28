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


class BatchExecutor:
    """
    Executes multiple requests with concurrency control.

    Used by benchmark strategies to run requests in parallel.
    """

    def __init__(self, executor: RequestExecutor) -> None:
        self.executor = executor

    async def execute_batch(
        self,
        backend: Backend,
        requests: list[GenerateRequest],
        concurrency: int,
    ) -> list[RequestResult]:
        """
        Execute multiple requests with limited concurrency.

        Args:
            backend: The inference backend to use
            requests: List of generation requests
            concurrency: Maximum concurrent requests

        Returns:
            List of results in same order as requests
        """
        import asyncio

        semaphore = asyncio.Semaphore(concurrency)
        results: list[RequestResult | None] = [None] * len(requests)

        async def run_with_semaphore(idx: int, req: GenerateRequest) -> None:
            async with semaphore:
                results[idx] = await self.executor.execute(backend, req)

        tasks = [run_with_semaphore(i, req) for i, req in enumerate(requests)]
        await asyncio.gather(*tasks)

        return [r for r in results if r is not None]
