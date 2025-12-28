"""Benchmark with managed vLLM server (splleed starts the server)."""

import asyncio

from splleed import Benchmark, SamplingParams, VLLMConfig
from splleed.reporters import print_results


async def main():
    # No endpoint = splleed will start and manage the vLLM server
    b = Benchmark(
        backend=VLLMConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            gpu_memory_utilization=0.8,
        ),
        prompts=[
            "What is the capital of France?",
            "Explain photosynthesis in one sentence.",
            "Write a haiku about coding.",
            "What is 15 * 23?",
            "Name three programming languages.",
        ],
        mode="latency",
        concurrency=[1, 2],
        warmup=1,
        runs=5,
        sampling=SamplingParams(max_tokens=64, temperature=0.0),
    )

    results = await b.run()
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
