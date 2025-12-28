"""Replicate vLLM's benchmark_serving.py using splleed.

This example shows how to run a benchmark equivalent to:
    vllm bench serve --model Qwen/Qwen2.5-3B-Instruct \
        --dataset-name random --num-prompts 100 \
        --request-rate 10 --max-concurrency 32

Usage:
    python examples/vllm_benchmark.py
"""

import asyncio
import random

from splleed import Benchmark, SamplingParams, VLLMConfig


def generate_random_prompts(
    num_prompts: int,
    input_len: int = 512,
    seed: int = 42,
) -> list[str]:
    """Generate random prompts of approximately input_len tokens.

    Mimics vLLM's random dataset. Uses simple word repetition
    since exact token count depends on tokenizer.
    """
    rng = random.Random(seed)
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "hello",
        "world",
        "python",
        "code",
        "data",
        "model",
        "train",
        "test",
        "run",
        "fast",
        "slow",
        "big",
        "small",
        "new",
        "old",
    ]

    prompts = []
    for _ in range(num_prompts):
        # Approximate: 1 word ≈ 1.3 tokens on average
        num_words = int(input_len / 1.3)
        prompt = " ".join(rng.choices(words, k=num_words))
        prompts.append(prompt)

    return prompts


async def main():
    # Configuration matching vLLM benchmark defaults
    model = "Qwen/Qwen2.5-3B-Instruct"
    num_prompts = 100
    input_len = 512
    output_len = 128
    request_rate = 10.0  # requests per second
    max_concurrency = 32

    print(f"Model: {model}")
    print(f"Num prompts: {num_prompts}")
    print(f"Input length: ~{input_len} tokens")
    print(f"Output length: {output_len} tokens")
    print(f"Request rate: {request_rate} req/s")
    print(f"Max concurrency: {max_concurrency}")
    print()

    # Generate random prompts (like vLLM's --dataset-name random)
    prompts = generate_random_prompts(
        num_prompts=num_prompts,
        input_len=input_len,
        seed=42,
    )

    # Run benchmark with Poisson arrivals (like vLLM's default)
    results = await Benchmark(
        backend=VLLMConfig(model=model),
        prompts=prompts,
        mode="serve",
        arrival_rate=request_rate,
        arrival_pattern="poisson",
        concurrency=[max_concurrency],
        warmup=3,
        trials=1,
        sampling=SamplingParams(
            max_tokens=output_len,
            temperature=0.0,
        ),
    ).run()

    results.print()
    results.save("vllm_benchmark_results.json")
    print("\nResults saved to vllm_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
