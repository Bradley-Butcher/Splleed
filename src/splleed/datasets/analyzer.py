"""Dataset distribution analysis utilities."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console
from rich.table import Table

from .base import Dataset

if TYPE_CHECKING:
    from splleed.tokenizers import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class DistributionStats:
    """Statistics for a distribution of values."""

    min: float
    max: float
    mean: float
    median: float
    std: float
    p50: float
    p95: float
    p99: float
    count: int


@dataclass
class DatasetStats:
    """Complete statistics about a dataset."""

    total_samples: int
    prompt_token_stats: DistributionStats | None
    output_len_stats: DistributionStats | None


def _compute_stats(values: Sequence[int | float]) -> DistributionStats:
    """Compute distribution statistics for a list of values."""
    if not values:
        return DistributionStats(
            min=0, max=0, mean=0, median=0, std=0, p50=0, p95=0, p99=0, count=0
        )

    arr = np.array(values)
    return DistributionStats(
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        count=len(values),
    )


def analyze_dataset(
    dataset: Dataset,
    tokenizer: Tokenizer | None = None,
    max_samples: int = 10000,
) -> DatasetStats:
    """
    Analyze dataset distribution.

    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer for token counts (required for meaningful analysis)
        max_samples: Maximum samples to analyze (for large datasets)

    Returns:
        DatasetStats with distribution information
    """
    n_samples = min(len(dataset), max_samples)
    samples = dataset.sample(n_samples)

    token_lens: list[int] = []
    output_lens: list[int] = []

    for sample in samples:
        if sample.prompt_tokens is not None:
            token_lens.append(sample.prompt_tokens)
        elif tokenizer is not None:
            token_lens.append(tokenizer.count_tokens(sample.prompt))

        if sample.expected_output_len is not None:
            output_lens.append(sample.expected_output_len)

    return DatasetStats(
        total_samples=len(dataset),
        prompt_token_stats=_compute_stats(token_lens) if token_lens else None,
        output_len_stats=_compute_stats(output_lens) if output_lens else None,
    )


def print_dataset_analysis(
    stats: DatasetStats,
    console: Console | None = None,
) -> None:
    """
    Print dataset analysis to console with Rich formatting.

    Args:
        stats: Dataset statistics to display
        console: Rich console (creates new one if not provided)
    """
    if console is None:
        console = Console()

    console.print()
    console.print("[bold cyan]Dataset Analysis[/bold cyan]")
    console.print(f"  Total samples: {stats.total_samples:,}")
    console.print()

    if stats.prompt_token_stats:
        table = Table(title="Prompt Token Distribution")
        table.add_column("Metric", style="cyan")
        table.add_column("Tokens", justify="right")

        ts = stats.prompt_token_stats
        for metric, val in [
            ("min", ts.min),
            ("max", ts.max),
            ("mean", ts.mean),
            ("median", ts.median),
            ("p95", ts.p95),
            ("p99", ts.p99),
        ]:
            table.add_row(metric, f"{val:,.0f}")

        console.print(table)

    if stats.output_len_stats:
        console.print()
        out_table = Table(title="Expected Output Length Distribution")
        out_table.add_column("Metric", style="cyan")
        out_table.add_column("Tokens", justify="right")

        out_stats = stats.output_len_stats
        for metric, val in [
            ("min", out_stats.min),
            ("max", out_stats.max),
            ("mean", out_stats.mean),
            ("p50", out_stats.p50),
            ("p95", out_stats.p95),
        ]:
            out_table.add_row(metric, f"{val:,.0f}")

        console.print(out_table)

    console.print()


def format_dataset_summary(stats: DatasetStats) -> str:
    """
    Format dataset statistics as a brief one-line summary.

    Args:
        stats: Dataset statistics

    Returns:
        Summary string like "500 samples, 100-1000 tokens (mean: 462)"
    """
    parts = [f"{stats.total_samples:,} samples"]

    if stats.prompt_token_stats:
        ts = stats.prompt_token_stats
        parts.append(f"{ts.min:.0f}-{ts.max:.0f} tokens (mean: {ts.mean:.0f})")

    return ", ".join(parts)
