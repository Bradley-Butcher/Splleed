"""Advanced: backend-specific options, custom sampling, and output."""

import asyncio
from pathlib import Path

from splleed import Benchmark, SamplingParams, VLLMConfig
from splleed.reporters import print_results


async def main():
    # Backend with vLLM-specific options
    backend = VLLMConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel=2,
        gpu_memory_utilization=0.85,
        quantization="awq",
    )

    # Custom sampling parameters
    sampling = SamplingParams(
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

    b = Benchmark(
        backend=backend,
        prompts=[
            "Explain the theory of relativity.",
            "Write a short story about a robot learning to paint.",
            "What are the main differences between Python and Rust?",
            "Describe the process of photosynthesis.",
        ],
        concurrency=[1, 4, 8, 16],
        mode="throughput",
        trials=5,
        warmup=3,
        runs=20,
        sampling=sampling,
        output_file=Path("benchmark_results.json"),
    )

    results = await b.run()
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
