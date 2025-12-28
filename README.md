# splleed

LLM inference benchmarking with a Python-first API.

## Features

- **Python API**: Write benchmarks as scripts, not config files
- **Pluggable backends**: vLLM, TGI (more coming)
- **Comprehensive metrics**: TTFT, ITL, TPOT, throughput, E2E latency
- **Statistical rigor**: Multiple trials with confidence intervals
- **Flexible operation**: Connect to existing servers or let splleed manage them

## Installation

```bash
pip install splleed
```

For HuggingFace dataset support:
```bash
pip install splleed[hf]
```

Inference engines (vLLM, TGI) are **not** bundled - install them separately.

## Quick Start

```python
import asyncio
from splleed import Benchmark, VLLMConfig, SamplingParams

async def main():
    results = await Benchmark(
        backend=VLLMConfig(model="Qwen/Qwen2.5-0.5B-Instruct"),
        prompts=[
            "What is the capital of France?",
            "Explain quantum computing briefly.",
        ],
        concurrency=[1, 2, 4],
        trials=3,
        sampling=SamplingParams(max_tokens=100),
    ).run()

    results.print()
    results.save("results.json")

if __name__ == "__main__":
    asyncio.run(main())
```

## Connect vs Managed Mode

**Managed mode** - splleed starts and stops the server:
```python
backend = VLLMConfig(model="Qwen/Qwen2.5-0.5B-Instruct")
```

**Connect mode** - use an existing server:
```python
backend = VLLMConfig(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    endpoint="http://localhost:8000",
)
```

## Using HuggingFace Datasets

```python
from datasets import load_dataset
from splleed import Benchmark, VLLMConfig

async def main():
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(100))
    prompts = list(ds["instruction"])

    results = await Benchmark(
        backend=VLLMConfig(model="Qwen/Qwen2.5-3B-Instruct"),
        prompts=prompts,
        concurrency=[1, 2, 4, 8],
        trials=3,
    ).run()

    results.print()
```

## Backend Configuration

### vLLM

```python
from splleed import VLLMConfig

backend = VLLMConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel=2,
    gpu_memory_utilization=0.9,
    quantization="awq",  # optional
    dtype="auto",
)
```

### TGI

```python
from splleed import TGIConfig

backend = TGIConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantize="bitsandbytes-nf4",  # optional
)
```

## Benchmark Options

```python
Benchmark(
    backend=...,
    prompts=["..."],

    # Benchmark settings
    mode="latency",          # "latency", "throughput", or "serve"
    concurrency=[1, 4, 8],   # concurrency levels to test
    warmup=2,                # warmup iterations
    runs=10,                 # requests per concurrency level
    trials=3,                # independent trials for CI
    confidence_level=0.95,   # confidence interval level

    # Sampling parameters
    sampling=SamplingParams(
        max_tokens=100,
        temperature=0.0,
        top_p=1.0,
    ),
)
```

## Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token |
| ITL | Inter-token latency |
| TPOT | Time per output token (mean ITL) |
| E2E | End-to-end request latency |
| Throughput | Tokens/sec |
| Goodput | % of requests meeting SLO |

All latency metrics include p50, p95, p99, and mean. With multiple trials, results include 95% confidence intervals.

## Output Formats

```python
results.print()              # Rich table to console
results.save("out.json")     # JSON format
results.save("out.csv")      # CSV format

json_str = results.to_json()
csv_str = results.to_csv()
```

## License

MIT
