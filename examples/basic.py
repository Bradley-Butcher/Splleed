"""Basic benchmark with inline prompts."""

import asyncio

from splleed import Benchmark, SamplingParams, VLLMConfig
from splleed.reporters import print_results


async def main():
    b = Benchmark(
        backend=VLLMConfig(model="Qwen/Qwen2.5-0.5B-Instruct"),
        prompts=[
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about programming.",
        ],
        concurrency=[1, 2, 4],
        mode="latency",
        warmup=2,
        runs=10,
        sampling=SamplingParams(max_tokens=100, temperature=0.0),
    )

    results = await b.run()
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
