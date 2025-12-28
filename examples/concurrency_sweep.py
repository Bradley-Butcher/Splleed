"""Sweep concurrency levels to find optimal throughput.

This example runs benchmarks at increasing concurrency levels
to find the point where throughput saturates and latency degrades.

Usage:
    python examples/concurrency_sweep.py
"""

import asyncio

from splleed import Benchmark, SamplingParams, VLLMConfig


async def main():
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to sort a list.",
        "What are the benefits of renewable energy?",
        "Describe the water cycle.",
        "How does machine learning work?",
    ] * 20  # 100 prompts

    # Sweep concurrency from 1 to 64
    results = await Benchmark(
        backend=VLLMConfig(model="Qwen/Qwen2.5-3B-Instruct"),
        prompts=prompts,
        mode="throughput",
        concurrency=[1, 2, 4, 8, 16, 32, 64],
        warmup=2,
        trials=3,
        sampling=SamplingParams(max_tokens=128),
    ).run()

    results.print()

    # Print scaling analysis
    if results.aggregated_results:
        print("\nScaling Analysis:")
        print("-" * 50)
        base_throughput = results.aggregated_results[0].throughput_tokens_per_sec.mean
        for r in results.aggregated_results:
            scaling = r.throughput_tokens_per_sec.mean / base_throughput
            print(
                f"  Concurrency {r.concurrency:2d}: "
                f"{r.throughput_tokens_per_sec.mean:6.1f} tok/s "
                f"({scaling:.2f}x vs baseline)"
            )


if __name__ == "__main__":
    asyncio.run(main())
