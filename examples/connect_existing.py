"""Connect to an already-running vLLM or TGI server."""

import asyncio

from splleed import Benchmark, SamplingParams, VLLMConfig
from splleed.reporters import print_results


async def main():
    # Connect to existing vLLM server instead of starting one
    # The server should already be running, e.g.:
    #   vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
    b = Benchmark(
        backend=VLLMConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            endpoint="http://localhost:8000",
        ),
        prompts=[
            "What is 2+2?",
            "Write a poem about the ocean.",
            "Explain machine learning briefly.",
        ],
        concurrency=[1, 2],
        sampling=SamplingParams(max_tokens=50),
    )

    results = await b.run()
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
