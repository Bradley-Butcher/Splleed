"""Benchmark with ShareGPT-style workload.

ShareGPT is the industry standard dataset for LLM serving benchmarks.
It contains real conversations with realistic input/output length distributions.

Requires: pip install splleed[hf]

Usage:
    python examples/sharegpt_benchmark.py
"""

import asyncio
import random

from splleed import Benchmark, SamplingParams, VLLMConfig


def get_prompts(num_prompts: int = 500, min_length: int = 10, seed: int = 42) -> list[str]:
    """Load prompts from ShareGPT dataset.

    Extracts the first human turn from each conversation.

    Args:
        num_prompts: Number of prompts to sample
        min_length: Minimum prompt length in characters
        seed: Random seed for sampling
    """
    from datasets import load_dataset

    print("Loading ShareGPT dataset...")
    ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")

    prompts: list[str] = []
    for row in ds.to_list():
        conversations: list[dict] = row.get("conversations", [])
        for turn in conversations:
            if turn.get("from") == "human":
                text: str = turn.get("value", "")
                if len(text) > min_length:
                    prompts.append(text)
                break

    rng = random.Random(seed)
    prompts = rng.sample(prompts, min(num_prompts, len(prompts)))

    print(f"Loaded {len(prompts)} prompts")

    return prompts


async def main():
    prompts = get_prompts(num_prompts=500, seed=42)

    results = await Benchmark(
        backend=VLLMConfig(model="Qwen/Qwen2.5-3B-Instruct"),
        prompts=prompts,
        mode="serve",
        arrival_rate=5.0,
        arrival_pattern="poisson",
        concurrency=[16],
        warmup=2,
        trials=3,
        sampling=SamplingParams(max_tokens=256, temperature=0.0),
    ).run()

    results.print()
    results.save("sharegpt_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
