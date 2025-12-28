"""Basic benchmark with inline prompts."""

import asyncio

from splleed import Benchmark, SamplingParams, VLLMConfig


async def main():
    results = await Benchmark(
        backend=VLLMConfig(model="Qwen/Qwen2.5-0.5B-Instruct"),
        prompts=[
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about programming.",
        ],
        concurrency=[1, 2, 4],
        warmup=2,
        trials=3,
        sampling=SamplingParams(max_tokens=100),
    ).run()

    results.print()


if __name__ == "__main__":
    asyncio.run(main())
