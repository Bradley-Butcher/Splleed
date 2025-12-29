# Improvements to Make Splleed Useful for vLLM/TGI Optimization Teams

This document outlines concrete steps to evolve Splleed from a client-side benchmarking tool into something valuable for inference engine optimization work.

---

## 1. Server-Side Metrics Collection (High Impact)

### The Problem
Currently Splleed only measures what a client sees over HTTP. Optimization engineers need to see what's happening inside the engine.

### The Solution
Both vLLM and TGI expose Prometheus metrics endpoints that contain exactly this data.

**vLLM exposes at `/metrics`:**
```
# Batch size
vllm:num_requests_running{...}
vllm:num_requests_waiting{...}

# KV Cache
vllm:gpu_cache_usage_perc{...}
vllm:cpu_cache_usage_perc{...}

# Timing breakdown
vllm:time_to_first_token_seconds{...}
vllm:time_per_output_token_seconds{...}
vllm:e2e_request_latency_seconds{...}

# Prefill/decode (in newer versions)
vllm:avg_prompt_throughput_toks_per_s{...}
vllm:avg_generation_throughput_toks_per_s{...}
```

**TGI exposes similar metrics:**
```
tgi_batch_current_size
tgi_queue_size
tgi_request_duration_sum
tgi_request_generated_tokens_sum
```

### Implementation

```python
# New module: src/splleed/metrics/server_metrics.py

class ServerMetricsCollector:
    """Collect Prometheus metrics from inference servers."""

    def __init__(self, endpoint: str, interval: float = 0.5):
        self.metrics_url = f"{endpoint}/metrics"
        self.interval = interval
        self.samples: list[MetricsSample] = []

    async def start_collection(self):
        """Background task to poll metrics during benchmark."""
        while self._running:
            sample = await self._scrape_metrics()
            self.samples.append(sample)
            await asyncio.sleep(self.interval)

    def get_timeseries(self) -> ServerMetricsTimeseries:
        """Return collected metrics as time series."""
        return ServerMetricsTimeseries(
            timestamps=[s.timestamp for s in self.samples],
            batch_size=[s.batch_size for s in self.samples],
            kv_cache_usage=[s.kv_cache_pct for s in self.samples],
            queue_depth=[s.queue_depth for s in self.samples],
            # ... etc
        )
```

**New metrics exposed to users:**

| Metric | Source | Why It Matters |
|--------|--------|----------------|
| `batch_size_mean` | Server metrics | Batching efficiency |
| `batch_size_p99` | Server metrics | Worst-case batching |
| `kv_cache_usage_mean` | Server metrics | Memory pressure |
| `kv_cache_usage_max` | Server metrics | Did we hit limits? |
| `queue_depth_mean` | Server metrics | Scheduling pressure |
| `prefill_throughput` | Server metrics | Prompt processing speed |
| `decode_throughput` | Server metrics | Token generation speed |

---

## 2. GPU Metrics via NVML (High Impact)

### The Problem
Client-side timing can't show if the GPU was saturated or idle during inference.

### The Solution
Use `pynvml` to collect GPU metrics during benchmarks.

```python
# New module: src/splleed/metrics/gpu_metrics.py

import pynvml

class GPUMetricsCollector:
    """Collect GPU utilization metrics during benchmarks."""

    def __init__(self, device_ids: list[int] | None = None, interval: float = 0.1):
        pynvml.nvmlInit()
        self.devices = device_ids or list(range(pynvml.nvmlDeviceGetCount()))
        self.interval = interval
        self.samples: list[GPUSample] = []

    async def collect_sample(self):
        sample = GPUSample(timestamp=time.perf_counter())
        for idx in self.devices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            sample.devices[idx] = DeviceMetrics(
                sm_utilization=util.gpu,
                memory_utilization=util.memory,
                memory_used_gb=mem.used / 1e9,
                memory_total_gb=mem.total / 1e9,
            )
        self.samples.append(sample)
```

**New metrics:**

| Metric | Why It Matters |
|--------|----------------|
| `gpu_sm_util_mean` | Were compute units busy? |
| `gpu_sm_util_min` | Pipeline stalls? |
| `gpu_memory_util_mean` | Memory bandwidth bound? |
| `gpu_memory_used_max_gb` | Peak memory usage |

---

## 3. Prefill vs Decode Phase Separation (High Impact)

### The Problem
TTFT conflates network latency with actual prefill time. ITL conflates decode with scheduling delays.

### The Solution
Use server-side timestamps when available, or infer from token patterns.

**Option A: Parse server-side timing from response headers**

vLLM can be configured to include timing headers:
```python
# vLLM response includes:
# X-Prefill-Time-Ms: 45.2
# X-First-Token-Time-Ms: 47.1
```

**Option B: Separate prefill throughput from decode throughput**

```python
@dataclass
class PhasedMetrics:
    # Prefill phase
    prefill_tokens: int
    prefill_time_ms: float
    prefill_throughput_tps: float  # tokens/sec for prompt processing

    # Decode phase
    decode_tokens: int
    decode_time_ms: float
    decode_throughput_tps: float  # tokens/sec for generation

    # Derived
    prefill_pct: float  # What % of time was prefill?
```

This distinction is critical for optimization work:
- High prefill time → optimize attention kernels, prompt caching
- High decode time → optimize KV cache access, batching

---

## 4. A/B Comparison Mode (Medium Impact)

### The Problem
Optimization engineers constantly run before/after comparisons. Currently they must manually diff two JSON files.

### The Solution
First-class support for comparative benchmarking.

```python
from splleed import Benchmark, compare

# Run baseline
baseline = await Benchmark(
    backend=VLLMConfig(model="meta-llama/Llama-3.1-8B"),
    prompts=prompts,
    name="baseline",  # NEW: name for comparison
).run()

# Run with optimization
optimized = await Benchmark(
    backend=VLLMConfig(
        model="meta-llama/Llama-3.1-8B",
        enable_prefix_caching=True,  # The optimization
    ),
    prompts=prompts,  # Same prompts!
    name="prefix_caching",
).run()

# Compare
comparison = compare(baseline, optimized)
comparison.print()
comparison.save("comparison.html")  # Visual diff report
```

**Comparison output:**
```
┌─────────────────────────────────────────────────────────────┐
│ Comparison: baseline vs prefix_caching                       │
├──────────────────┬──────────┬───────────────┬───────────────┤
│ Metric           │ Baseline │ Prefix Cache  │ Change        │
├──────────────────┼──────────┼───────────────┼───────────────┤
│ TTFT p50 (ms)    │ 45.2     │ 12.3          │ -72.8% ✓      │
│ TTFT p99 (ms)    │ 89.1     │ 28.4          │ -68.1% ✓      │
│ ITL p50 (ms)     │ 8.2      │ 8.1           │ -1.2%         │
│ Throughput (t/s) │ 1,245    │ 1,312         │ +5.4% ✓       │
│ KV Cache Usage   │ 45%      │ 62%           │ +17pp         │
└──────────────────┴──────────┴───────────────┴───────────────┘
                              Statistically significant (p<0.05): ✓
```

---

## 5. Profiler Integration (Medium Impact)

### The Problem
Engineers want to correlate Splleed benchmarks with GPU profiles but must manually align timestamps.

### The Solution
Launch profilers alongside benchmarks with synchronized timestamps.

```python
results = await Benchmark(
    backend=VLLMConfig(model="..."),
    prompts=prompts,
    profile=ProfileConfig(
        tool="nsight",  # or "torch_profiler"
        output_dir="./profiles",
    ),
).run()

# Results now include:
# - results.profile_path = "./profiles/benchmark_20240115_143022.nsys-rep"
# - results.profile_annotations = [
#     {"name": "request_0", "start_ns": ..., "end_ns": ...},
#     ...
# ]
```

For `torch.profiler` integration:
```python
profile=ProfileConfig(
    tool="torch_profiler",
    activities=["cpu", "cuda"],
    with_stack=True,
)

# Produces Chrome trace JSON aligned with request timing
```

---

## 6. Request-Level Trace Export (Medium Impact)

### The Problem
Aggregate statistics hide interesting patterns. Engineers need to see individual request behavior.

### The Solution
Export detailed per-request traces in standard formats.

```python
results = await Benchmark(..., include_traces=True).run()

# Export to Chrome Trace format (viewable in chrome://tracing)
results.export_traces("traces.json", format="chrome")

# Export to OpenTelemetry format
results.export_traces("traces.otlp", format="otlp")
```

**Chrome trace visualization shows:**
- Each request as a span
- Token arrivals as events within the span
- Batch boundaries from server metrics
- GPU utilization overlay

---

## 7. Speculative Decoding Metrics (Low-Medium Impact)

### The Problem
Speculative decoding is increasingly important, but no visibility into acceptance rates.

### The Solution
Parse vLLM's speculative decoding metrics when enabled.

```python
@dataclass
class SpeculativeDecodingMetrics:
    draft_tokens_proposed: int
    draft_tokens_accepted: int
    acceptance_rate: float
    speculation_efficiency: float  # Speedup from speculation

    # Per-position acceptance (are later tokens rejected more?)
    position_acceptance_rates: list[float]
```

vLLM exposes:
```
vllm:spec_decode_draft_acceptance_rate{...}
vllm:spec_decode_efficiency{...}
```

---

## 8. Reproducibility Improvements (Low Impact but Important)

### Current State
Splleed captures some environment info but misses key details.

### Improvements

```python
@dataclass
class ExtendedEnvironmentInfo:
    # Current
    gpu_name: str
    gpu_count: int

    # Add: Software versions
    vllm_version: str  # "0.4.1"
    torch_version: str
    cuda_version: str
    driver_version: str

    # Add: vLLM/TGI configuration
    engine_config: dict  # Parsed from server /v1/models or logs

    # Add: System state
    gpu_power_limit_w: int
    gpu_clocks_mhz: tuple[int, int]  # (graphics, memory)
    numa_topology: dict

    # Add: Git state (for engine development)
    engine_git_sha: str | None
    engine_git_dirty: bool
```

---

## 9. Workload Characterization (Low Impact)

### The Problem
Engineers need to understand if benchmark prompts are representative.

### The Solution
Analyze and report workload characteristics.

```python
@dataclass
class WorkloadStats:
    prompt_lengths: PercentileStats  # p50, p95, p99
    output_lengths: PercentileStats
    prompt_length_distribution: str  # "uniform", "bimodal", "long-tail"

    # Token-level analysis
    unique_token_ratio: float  # Affects KV cache
    common_prefix_length: int  # Affects prefix caching

    # Sequence characteristics
    has_system_prompts: bool
    multi_turn_conversation: bool
```

Report this alongside results so engineers know what the benchmark represents.

---

## Implementation Priority

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Server-side metrics | High | Medium | **P0** |
| GPU metrics via NVML | High | Low | **P0** |
| Prefill/decode separation | High | Medium | **P1** |
| A/B comparison mode | Medium | Medium | **P1** |
| Request-level traces | Medium | Medium | **P2** |
| Profiler integration | Medium | High | **P2** |
| Speculative decoding | Low-Med | Low | **P2** |
| Reproducibility | Low | Low | **P3** |
| Workload characterization | Low | Low | **P3** |

---

## Example: What This Enables

After implementing P0 and P1 features, an optimization engineer could run:

```python
comparison = compare(
    await Benchmark(
        backend=VLLMConfig(model="llama-3.1-8b"),
        prompts=production_sample,
        collect_server_metrics=True,
        collect_gpu_metrics=True,
    ).run(),

    await Benchmark(
        backend=VLLMConfig(
            model="llama-3.1-8b",
            extra_args={"enable_chunked_prefill": True}
        ),
        prompts=production_sample,
        collect_server_metrics=True,
        collect_gpu_metrics=True,
    ).run(),
)

comparison.print()
```

And see:
```
Chunked Prefill Impact Analysis
═══════════════════════════════

Client-Side Metrics:
  TTFT p50:     45ms → 52ms (+15%) ✗   [Prefill now overlaps with decode]
  TTFT p99:     89ms → 67ms (-25%) ✓   [Less variance in prefill]
  ITL p50:      8.2ms → 7.1ms (-13%) ✓ [Better decode batching]
  Throughput:   1,245 → 1,456 t/s (+17%) ✓

Server-Side Metrics:
  Batch size mean:      4.2 → 6.8 (+62%) ✓
  KV cache usage:       45% → 52% (+7pp)
  Prefill throughput:   12,400 → 11,200 t/s (-10%)
  Decode throughput:    1,180 → 1,380 t/s (+17%) ✓

GPU Metrics:
  SM utilization:       72% → 84% (+12pp) ✓
  Memory bandwidth:     68% → 71% (+3pp)

Conclusion: Chunked prefill trades slight prefill regression for
significantly better batching and decode performance.
```

**This is actionable data for an optimization engineer.**

---

## Summary

The path from "user benchmarking tool" to "optimization engineering tool" requires:

1. **Seeing inside the engine** (server metrics, GPU metrics)
2. **Understanding phases** (prefill vs decode)
3. **Comparing systematically** (A/B mode with statistical significance)
4. **Correlating with profilers** (timestamp alignment, trace export)

These changes would make Splleed genuinely useful for vLLM/TGI optimization work, complementing rather than replacing their internal tools.
