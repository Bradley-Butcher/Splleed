"""Console result reporter using Rich tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from splleed.metrics.types import BenchmarkResults


def print_results(results: BenchmarkResults, console: Console | None = None) -> None:
    """
    Print benchmark results to console as a rich table.

    Args:
        results: Benchmark results
        console: Rich console (creates new one if None)
    """
    if console is None:
        console = Console()

    # Print metadata
    console.print()
    console.print("[bold]Benchmark Results[/bold]")
    console.print(f"  Engine: {results.engine}")
    console.print(f"  Model: {results.model}")
    console.print(f"  Timestamp: {results.timestamp}")
    if results.gpu:
        console.print(f"  GPU: {results.gpu}")
    console.print()

    # Create summary table
    table = Table(title="Summary by Concurrency Level")

    table.add_column("Concurrency", justify="right", style="cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Tokens/s", justify="right", style="yellow")
    table.add_column("TTFT p50", justify="right")
    table.add_column("TTFT p99", justify="right")
    table.add_column("ITL p50", justify="right")
    table.add_column("ITL p99", justify="right")
    table.add_column("E2E p50", justify="right")
    table.add_column("Goodput", justify="right")

    for r in results.results:
        goodput = f"{r.goodput_pct:.1f}%" if r.goodput_pct is not None else "-"
        table.add_row(
            str(r.concurrency),
            str(r.num_requests),
            f"{r.num_successful}/{r.num_requests}",
            f"{r.throughput_tokens_per_sec:.1f}",
            f"{r.ttft_p50_ms:.1f}ms",
            f"{r.ttft_p99_ms:.1f}ms",
            f"{r.itl_p50_ms:.1f}ms",
            f"{r.itl_p99_ms:.1f}ms",
            f"{r.e2el_p50_ms:.1f}ms",
            goodput,
        )

    console.print(table)
    console.print()


def format_summary(results: BenchmarkResults) -> str:
    """
    Format a brief text summary of results.

    Args:
        results: Benchmark results

    Returns:
        Summary string
    """
    lines = [
        f"Engine: {results.engine}",
        f"Model: {results.model}",
        "",
    ]

    for r in results.results:
        lines.append(f"Concurrency {r.concurrency}:")
        lines.append(f"  Throughput: {r.throughput_tokens_per_sec:.1f} tokens/s")
        lines.append(f"  TTFT: p50={r.ttft_p50_ms:.1f}ms, p99={r.ttft_p99_ms:.1f}ms")
        lines.append(f"  ITL: p50={r.itl_p50_ms:.1f}ms, p99={r.itl_p99_ms:.1f}ms")
        lines.append("")

    return "\n".join(lines)
