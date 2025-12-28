"""Benchmark in serve mode with realistic arrival patterns.

Serve mode simulates real-world traffic by sending requests according
to an arrival pattern (Poisson, constant, or gamma distribution).
"""

import asyncio

from splleed import Benchmark, SamplingParams, VLLMConfig
from splleed.reporters import print_results


async def main():
    b = Benchmark(
        backend=VLLMConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            endpoint="http://localhost:8000",
        ),
        prompts=[
            "What is the weather like today?",
            "How do I make pasta?",
            "Explain blockchain technology.",
            "What is the capital of Japan?",
            "Write a joke about programming.",
            "How does a CPU work?",
            "What is machine learning?",
            "Describe the water cycle.",
        ],
        mode="serve",
        concurrency=[4],  # Max concurrent requests
        arrival_rate=10.0,  # 10 requests per second average
        arrival_pattern="poisson",  # Realistic traffic pattern
        runs=8,
        sampling=SamplingParams(max_tokens=100),
    )

    results = await b.run()
    print_results(results)


async def constant_rate():
    """Alternative: constant arrival rate for predictable load testing."""
    b = Benchmark(
        backend=VLLMConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            endpoint="http://localhost:8000",
        ),
        prompts=["Test prompt"] * 20,
        mode="serve",
        concurrency=[8],
        arrival_rate=50.0,  # Exactly 50 requests per second
        arrival_pattern="constant",
        runs=20,
    )

    results = await b.run()
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
