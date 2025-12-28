"""Console result reporter using Rich tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from splleed.metrics.types import BenchmarkResults, ConcurrencyResultWithCI
    from splleed.stats import ConfidenceInterval


def _format_ci(ci: ConfidenceInterval, suffix: str = "") -> str:
    """Format a confidence interval as 'mean +/- margin suffix'."""
    if ci.n_samples <= 1:
        return f"{ci.mean:.1f}{suffix}"
    return f"{ci.mean:.1f} +/- {ci.margin:.1f}{suffix}"


def _format_ci_pct(ci: ConfidenceInterval) -> str:
    """Format a confidence interval as percentage."""
    if ci.n_samples <= 1:
        return f"{ci.mean:.1f}%"
    return f"{ci.mean:.1f} +/- {ci.margin:.1f}%"


def print_results(results: BenchmarkResults, console: Console | None = None) -> None:
    """
    Print benchmark results to console as a rich table.

    Automatically detects if results have confidence intervals (multi-trial)
    and formats accordingly.

    Args:
        results: Benchmark results
        console: Rich console (creates new one if None)
    """
    if console is None:
        console = Console()

    # Print metadata
    console.print()
    console.print("[bold cyan]Benchmark Results[/bold cyan]")
    console.print(f"  Engine: {results.engine}")
    console.print(f"  Model: {results.model}")
    console.print(f"  Timestamp: {results.timestamp}")
    if results.gpu:
        console.print(f"  GPU: {results.gpu}")

    # Print trial info if multi-trial
    if results.n_trials > 1:
        bench_config = results.config.get("benchmark", {})
        confidence_pct = int(bench_config.get("confidence_level", 0.95) * 100)
        console.print(f"  Trials: {results.n_trials} ({confidence_pct}% CI)")

    console.print()

    # Print environment if available
    if results.environment:
        _print_environment(results.environment, console)

    # Check if we have aggregated results with CIs
    if results.aggregated_results and results.n_trials > 1:
        _print_ci_table(results.aggregated_results, console)
    else:
        _print_single_trial_table(results, console)


def _print_environment(env, console: Console) -> None:
    """Print environment information."""
    console.print("[bold]Environment[/bold]")

    if env.gpus:
        gpu_str = ", ".join(f"{gpu.name} ({gpu.vram_mb // 1024}GB)" for gpu in env.gpus)
        cuda_str = env.gpus[0].cuda_version if env.gpus else None
        if cuda_str:
            console.print(f"  GPU: {gpu_str} (CUDA {cuda_str})")
        else:
            console.print(f"  GPU: {gpu_str}")

    if env.engine:
        version_str = f" {env.engine.version}" if env.engine.version else ""
        console.print(f"  Engine: {env.engine.name}{version_str}")

    if env.system:
        console.print(f"  System: Python {env.system.python_version}")

    console.print()


def _print_single_trial_table(results: BenchmarkResults, console: Console) -> None:
    """Print results table for single trial (no CIs)."""
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


def _print_ci_table(
    aggregated: list[ConcurrencyResultWithCI],
    console: Console,
) -> None:
    """Print results table with confidence intervals (multi-trial)."""
    table = Table(title="Summary by Concurrency Level (with CI)")

    table.add_column("Concurrency", justify="right", style="cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Tokens/s", justify="right", style="yellow")
    table.add_column("TTFT p50", justify="right")
    table.add_column("TTFT p99", justify="right")
    table.add_column("ITL p50", justify="right")
    table.add_column("E2E p50", justify="right")
    table.add_column("Goodput", justify="right")

    for r in aggregated:
        goodput = _format_ci_pct(r.goodput_pct) if r.goodput_pct is not None else "-"
        table.add_row(
            str(r.concurrency),
            str(r.num_requests),
            f"{r.num_successful}/{r.num_requests}",
            _format_ci(r.throughput_tokens_per_sec),
            _format_ci(r.ttft_p50_ms, "ms"),
            _format_ci(r.ttft_p99_ms, "ms"),
            _format_ci(r.itl_p50_ms, "ms"),
            _format_ci(r.e2el_p50_ms, "ms"),
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
    ]

    if results.n_trials > 1:
        lines.append(f"Trials: {results.n_trials}")

    lines.append("")

    # Use aggregated results if available
    if results.aggregated_results and results.n_trials > 1:
        for r in results.aggregated_results:
            lines.append(f"Concurrency {r.concurrency}:")
            lines.append(f"  Throughput: {_format_ci(r.throughput_tokens_per_sec)} tokens/s")
            ttft_p50 = _format_ci(r.ttft_p50_ms)
            ttft_p99 = _format_ci(r.ttft_p99_ms)
            lines.append(f"  TTFT: p50={ttft_p50}ms, p99={ttft_p99}ms")
            itl_p50 = _format_ci(r.itl_p50_ms)
            itl_p99 = _format_ci(r.itl_p99_ms)
            lines.append(f"  ITL: p50={itl_p50}ms, p99={itl_p99}ms")
            lines.append("")
    else:
        for r in results.results:
            lines.append(f"Concurrency {r.concurrency}:")
            lines.append(f"  Throughput: {r.throughput_tokens_per_sec:.1f} tokens/s")
            lines.append(f"  TTFT: p50={r.ttft_p50_ms:.1f}ms, p99={r.ttft_p99_ms:.1f}ms")
            lines.append(f"  ITL: p50={r.itl_p50_ms:.1f}ms, p99={r.itl_p99_ms:.1f}ms")
            lines.append("")

    return "\n".join(lines)
