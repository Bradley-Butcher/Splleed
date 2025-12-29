"""Controlled benchmarks for kernel/engine developers.

This example shows how to run controlled experiments to isolate
prefill vs decode performance - useful when optimizing vLLM/TGI kernels.

Usage:
    python examples/kernel_dev.py
"""

import asyncio

from transformers import AutoTokenizer

from splleed import Benchmark, SamplingParams, VLLMConfig, synthetic_prompts


async def main():
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model)

    # =========================================================================
    # Prefill benchmark: Long input, minimal output
    # =========================================================================
    # Use this to test attention kernel optimizations (FlashAttention, etc.)
    # TTFT is the key metric here.

    print("=" * 60)
    print("PREFILL BENCHMARK")
    print("Long input (1024 tokens), minimal output (1 token)")
    print("Key metric: TTFT (time to first token)")
    print("=" * 60)

    prefill_prompts = synthetic_prompts(
        count=50,
        target_tokens=1024,
        tokenizer=tokenizer.encode,
        seed=42,
    )

    prefill_results = await Benchmark(
        backend=VLLMConfig(model=model),
        prompts=prefill_prompts,
        concurrency=[1],
        warmup=5,
        trials=3,
        sampling=SamplingParams(max_tokens=1, min_tokens=1),
    ).run()

    prefill_results.print()

    # =========================================================================
    # Decode benchmark: Short input, forced long output
    # =========================================================================
    # Use this to test KV cache / PagedAttention optimizations.
    # ITL (inter-token latency) and throughput are key metrics.

    print("\n" + "=" * 60)
    print("DECODE BENCHMARK")
    print("Short input (128 tokens), forced output (256 tokens)")
    print("Key metric: ITL (inter-token latency), throughput")
    print("=" * 60)

    decode_prompts = synthetic_prompts(
        count=50,
        target_tokens=128,
        tokenizer=tokenizer.encode,
        seed=42,
    )

    decode_results = await Benchmark(
        backend=VLLMConfig(model=model),
        prompts=decode_prompts,
        concurrency=[1, 2, 4, 8],
        warmup=5,
        trials=3,
        sampling=SamplingParams(max_tokens=256, min_tokens=256),
    ).run()

    decode_results.print()

    # =========================================================================
    # Scaling benchmark: Find saturation point
    # =========================================================================
    # Use this to test scheduler/batching optimizations.
    # Watch throughput plateau to find where you saturate.

    print("\n" + "=" * 60)
    print("SCALING BENCHMARK")
    print("Fixed workload, sweep concurrency to find saturation")
    print("Key metric: throughput scaling, saturation point")
    print("=" * 60)

    scaling_prompts = synthetic_prompts(
        count=100,
        target_tokens=512,
        tokenizer=tokenizer.encode,
        seed=42,
    )

    scaling_results = await Benchmark(
        backend=VLLMConfig(model=model),
        prompts=scaling_prompts,
        concurrency=[1, 2, 4, 8, 16, 32],
        warmup=5,
        trials=3,
        sampling=SamplingParams(max_tokens=128, min_tokens=128),
    ).run()

    scaling_results.print()
    scaling_results.save("kernel_dev_results.json")


if __name__ == "__main__":
    asyncio.run(main())
