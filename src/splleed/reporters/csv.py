"""CSV result reporter."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from splleed.metrics.types import BenchmarkResults


def to_csv(results: BenchmarkResults) -> str:
    """
    Convert benchmark results to CSV string.

    One row per concurrency level with key metrics.

    Args:
        results: Benchmark results

    Returns:
        CSV string
    """
    output = io.StringIO()

    fieldnames = [
        "concurrency",
        "num_requests",
        "num_successful",
        "num_failed",
        "throughput_tokens_per_sec",
        "throughput_requests_per_sec",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "ttft_mean_ms",
        "itl_p50_ms",
        "itl_p95_ms",
        "itl_p99_ms",
        "itl_mean_ms",
        "tpot_mean_ms",
        "e2el_p50_ms",
        "e2el_p95_ms",
        "e2el_p99_ms",
        "e2el_mean_ms",
        "goodput_pct",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for result in results.results:
        row = {
            "concurrency": result.concurrency,
            "num_requests": result.num_requests,
            "num_successful": result.num_successful,
            "num_failed": result.num_failed,
            "throughput_tokens_per_sec": f"{result.throughput_tokens_per_sec:.2f}",
            "throughput_requests_per_sec": f"{result.throughput_requests_per_sec:.2f}",
            "ttft_p50_ms": f"{result.ttft_p50_ms:.2f}",
            "ttft_p95_ms": f"{result.ttft_p95_ms:.2f}",
            "ttft_p99_ms": f"{result.ttft_p99_ms:.2f}",
            "ttft_mean_ms": f"{result.ttft_mean_ms:.2f}",
            "itl_p50_ms": f"{result.itl_p50_ms:.2f}",
            "itl_p95_ms": f"{result.itl_p95_ms:.2f}",
            "itl_p99_ms": f"{result.itl_p99_ms:.2f}",
            "itl_mean_ms": f"{result.itl_mean_ms:.2f}",
            "tpot_mean_ms": f"{result.tpot_mean_ms:.2f}",
            "e2el_p50_ms": f"{result.e2el_p50_ms:.2f}",
            "e2el_p95_ms": f"{result.e2el_p95_ms:.2f}",
            "e2el_p99_ms": f"{result.e2el_p99_ms:.2f}",
            "e2el_mean_ms": f"{result.e2el_mean_ms:.2f}",
            "goodput_pct": f"{result.goodput_pct:.2f}" if result.goodput_pct is not None else "",
        }
        writer.writerow(row)

    return output.getvalue()


def write_csv(results: BenchmarkResults, path: Path) -> None:
    """
    Write benchmark results to CSV file.

    Args:
        results: Benchmark results
        path: Output file path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_csv(results))
