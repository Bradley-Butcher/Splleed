"""Benchmark with HuggingFace dataset.

Requires the 'hf' optional dependency:
    pip install splleed[hf]
    # or with uv:
    uv pip install splleed[hf]
"""

import asyncio

from splleed import Benchmark, VLLMConfig
from splleed.reporters import print_results


async def main():
    # Option 1: Load dataset yourself (recommended for complex preprocessing)
    from datasets import load_dataset  # pyright: ignore

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(100))
    prompts = list(ds["instruction"])

    b = Benchmark(
        backend=VLLMConfig(model="Qwen/Qwen2.5-0.5B-Instruct"),
        prompts=prompts,
        concurrency=[1, 2, 4, 8],
        trials=3,
        confidence_level=0.95,
    )

    results = await b.run()
    print_results(results)


async def main_with_processor():
    """Alternative: use dataset= and processor= parameters."""

    def my_processor(ds):
        """Process dataset: filter short prompts and rename column."""
        ds = ds.filter(lambda r: len(r["instruction"]) > 50)
        ds = ds.rename_column("instruction", "prompt")
        return ds

    b = Benchmark(
        backend=VLLMConfig(model="Qwen/Qwen2.5-0.5B-Instruct"),
        dataset="tatsu-lab/alpaca",
        num_samples=100,
        concurrency=[1, 2, 4],
    )

    results = await b.run(processor=my_processor)
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
