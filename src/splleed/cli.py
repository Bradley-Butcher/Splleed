"""Splleed CLI - LLM inference benchmarking harness."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(
    name="splleed",
    help="LLM inference benchmarking harness with pluggable backends.",
    no_args_is_help=True,
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def run_async(coro):
    """Run async function in sync context."""
    return asyncio.run(coro)


@app.command()
def run(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to configuration YAML file"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Override output path"),
    ] = None,
    format: Annotated[
        str | None,
        typer.Option("--format", "-f", help="Output format (json, csv, markdown)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
    cloud: Annotated[
        bool,
        typer.Option("--cloud", help="Launch on cloud via SkyPilot"),
    ] = False,
    gpu: Annotated[
        str | None,
        typer.Option("--gpu", help="GPU type for cloud (e.g., A100:1, H100:1)"),
    ] = None,
) -> None:
    """Run a benchmark from a configuration file."""
    setup_logging(verbose)

    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_path}")
        raise typer.Exit(1)

    if cloud:
        console.print("[yellow]Cloud mode not yet implemented[/yellow]")
        raise typer.Exit(1)

    console.print(f"[bold]Loading config from:[/bold] {config_path}")

    try:
        from splleed.config import load_config

        # Build overrides from CLI options
        overrides = {}
        if output:
            overrides["output.path"] = str(output)
        if format:
            overrides["output.format"] = format

        config = load_config(config_path, overrides if overrides else None)

    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1) from None

    console.print(f"  Backend: [cyan]{config.backend.type}[/cyan]")
    console.print(f"  Model: [cyan]{getattr(config.backend, 'model', 'N/A')}[/cyan]")
    console.print(f"  Mode: [cyan]{config.benchmark.mode}[/cyan]")
    console.print()

    try:
        from splleed.reporters import print_results, write_csv, write_json
        from splleed.runner import run_benchmark

        results = run_async(run_benchmark(config))

        # Print results to console
        print_results(results, console)

        # Write to file
        output_path = config.output.path
        output_format = config.output.format

        if output_format == "json":
            write_json(results, output_path)
        elif output_format == "csv":
            write_csv(results, output_path)
        else:
            write_json(results, output_path)

        console.print(f"[green]Results written to:[/green] {output_path}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted[/yellow]")
        raise typer.Exit(130) from None

    except Exception as e:
        console.print(f"[red]Benchmark failed:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from None


@app.command()
def validate(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to configuration YAML file"),
    ],
) -> None:
    """Validate a configuration file without running."""
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_path}")
        raise typer.Exit(1)

    try:
        from splleed.config import load_yaml, validate_config

        data = load_yaml(config_path)
        errors = validate_config(data)

        if errors:
            console.print("[red]Config validation failed:[/red]")
            for error in errors:
                console.print(f"  - {error}")
            raise typer.Exit(1)

        console.print(f"[green]Config is valid:[/green] {config_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@app.command()
def backends() -> None:
    """List available backends."""
    from splleed.backends import BACKENDS

    table = Table(title="Available Backends")
    table.add_column("Name", style="cyan")
    table.add_column("Config Class", style="green")
    table.add_column("Backend Class", style="yellow")

    for name, (config_cls, backend_cls) in BACKENDS.items():
        table.add_row(name, config_cls.__name__, backend_cls.__name__)

    console.print(table)


@app.command("new-backend")
def new_backend(
    name: Annotated[
        str,
        typer.Argument(help="Name for the new backend (e.g., 'ollama', 'tgi')"),
    ],
) -> None:
    """Scaffold a new backend from template."""
    import shutil

    template_dir = Path(__file__).parent / "backends" / "_template"
    target_dir = Path(__file__).parent / "backends" / name

    if target_dir.exists():
        console.print(f"[red]Error:[/red] Backend directory already exists: {target_dir}")
        raise typer.Exit(1)

    # Copy template
    shutil.copytree(template_dir, target_dir)

    # Replace template placeholders in files
    for file_path in target_dir.glob("*.py"):
        content = file_path.read_text()
        content = content.replace("template", name)
        content = content.replace("Template", name.title().replace("_", ""))
        file_path.write_text(content)

    console.print(f"[green]Created new backend at:[/green] {target_dir}")
    console.print("\nNext steps:")
    console.print(f"  1. Edit {target_dir}/config.py - add your config options")
    console.print(f"  2. Edit {target_dir}/backend.py - implement the methods")
    console.print("  3. Register in backends/__init__.py")
    console.print(f"  4. Run tests: pytest tests/backends/test_{name}.py")


@app.command("init")
def init_config(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for config file"),
    ] = Path("splleed.yaml"),
) -> None:
    """Generate an example configuration file."""
    from splleed.config import generate_example_config

    if output.exists():
        console.print(f"[yellow]Warning:[/yellow] {output} already exists")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit(0)

    content = generate_example_config()
    output.write_text(content)
    console.print(f"[green]Created example config:[/green] {output}")


# Benchmark subcommand group
bench_app = typer.Typer(help="Run specific benchmark modes")
app.add_typer(bench_app, name="bench")


def _run_bench_mode(config_path: Path, mode: str, **overrides) -> None:
    """Common logic for bench subcommands."""
    setup_logging()

    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_path}")
        raise typer.Exit(1)

    try:
        from splleed.config import load_config
        from splleed.reporters import print_results, write_json
        from splleed.runner import run_benchmark

        # Override mode
        cli_overrides = {"benchmark.mode": mode}
        cli_overrides.update(overrides)

        config = load_config(config_path, cli_overrides)

        console.print(f"[bold]Running {mode} benchmark[/bold]")

        results = run_async(run_benchmark(config))
        print_results(results, console)

        output_path = config.output.path
        write_json(results, output_path)
        console.print(f"[green]Results written to:[/green] {output_path}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@bench_app.command()
def throughput(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to configuration YAML file"),
    ],
) -> None:
    """Run throughput benchmark (batch all requests)."""
    _run_bench_mode(config_path, "throughput")


@bench_app.command()
def latency(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to configuration YAML file"),
    ],
) -> None:
    """Run latency benchmark (single request)."""
    _run_bench_mode(config_path, "latency")


@bench_app.command()
def serve(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to configuration YAML file"),
    ],
    rate: Annotated[
        float | None,
        typer.Option("--rate", "-r", help="Request rate (requests/sec)"),
    ] = None,
) -> None:
    """Run serving benchmark with arrival patterns."""
    overrides = {}
    if rate is not None:
        overrides["benchmark.arrival.rate"] = rate
    _run_bench_mode(config_path, "serve", **overrides)


@bench_app.command()
def startup(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to configuration YAML file"),
    ],
) -> None:
    """Measure server startup time."""
    _run_bench_mode(config_path, "startup")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
