"""Benchmark with HuggingFace dataset.

Requires: pip install splleed[hf]
"""

import asyncio

from splleed import Benchmark, SamplingParams, VLLMConfig


async def main():
    from datasets import load_dataset

    # Load and preprocess dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(100))
    prompts = list(ds["instruction"])

    results = await Benchmark(
        backend=VLLMConfig(model="Qwen/Qwen2.5-3B-Instruct"),
        prompts=prompts,
        concurrency=[1, 2, 4, 8],
        trials=3,
        sampling=SamplingParams(max_tokens=128),
    ).run()

    results.print()
    results.save("hf_results.json")


if __name__ == "__main__":
    asyncio.run(main())
