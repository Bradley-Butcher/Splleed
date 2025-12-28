# splleed

LLM inference benchmarking harness with pluggable backends.

## Features

- **Pluggable backends**: vLLM, TGI (more coming)
- **Comprehensive metrics**: TTFT, ITL, TPOT, throughput, E2E latency
- **Multiple modes**: throughput, latency, serve simulation
- **Flexible operation**: Connect to existing servers or let splleed manage them

## Installation

```bash
# Clone the repo
git clone https://github.com/Bradley-Butcher/Splleed.git
cd Splleed

# With uv (recommended)
uv sync
uv run splleed --help

# Or with pip
pip install -e .
splleed --help
```

Inference engines (vLLM, TGI) are **not** bundled - install them separately as needed.

## Quick Start

```bash
# Run a benchmark
splleed run examples/vllm.yaml

# Other commands
splleed validate config.yaml   # Check config syntax
splleed backends               # List available backends
splleed init -o config.yaml    # Generate example config
```

## Configuration

### Connect Mode
Connect to an already-running server:

```yaml
backend:
  type: vllm
  endpoint: http://localhost:8000
```

### Managed Mode
Let splleed start and stop the server:

```yaml
backend:
  type: vllm
  model: Qwen/Qwen2.5-0.5B-Instruct
  port: 8000
```

### Full Example

```yaml
backend:
  type: vllm
  model: meta-llama/Llama-3.1-8B-Instruct
  port: 8000
  gpu_memory_utilization: 0.9

dataset:
  type: inline
  prompts:
    - "What is the capital of France?"
    - "Explain quantum computing."

benchmark:
  mode: latency        # throughput, latency, or serve
  concurrency: [1, 4, 8]
  warmup: 2
  runs: 10

sampling:
  max_tokens: 100
  temperature: 0.0

output:
  format: json
```

See `examples/` for more configurations.

## Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token |
| ITL | Inter-token latency |
| TPOT | Time per output token |
| E2E | End-to-end latency |
| Throughput | Tokens/sec |

All latency metrics include p50, p95, p99, and mean.

## Backend Setup

For managed mode, splleed finds the engine executable via:

1. Config: `executable: /path/to/vllm`
2. Env var: `SPLLEED_VLLM_PATH` or `SPLLEED_TGI_PATH`
3. System PATH

## Adding Backends

```bash
splleed new-backend my_engine
```

See `src/splleed/backends/_template/` for the template.

## License

MIT
