# Improvements to Make Splleed More Useful for Optimization Teams

This document outlines realistic improvements based on how optimization teams actually work and what would genuinely differentiate Splleed from existing tools.

---

## Context: What We Learned

1. **vLLM's own benchmark does client-side timing only**—same as Splleed
2. **Client-side benchmarks are validation tools**—they answer "did it work?" not "why is it slow?"
3. **Prometheus metrics are for monitoring, not benchmarking**—aggregate data doesn't explain per-request behavior
4. **The competition is `vllm bench serve`**—not Nsight or profilers

Splleed doesn't need to become a profiler. It needs to be a **better validation tool** than what already exists.

---

## High-Value Improvements

### 1. A/B Comparison Mode (Highest Priority)

**The Problem:** Optimization engineers constantly run before/after comparisons. Currently they run two benchmarks and manually compare JSON files.

**The Solution:**

```python
from splleed import Benchmark, compare

baseline = await Benchmark(
    backend=VLLMConfig(model="llama-3.1-8b"),
    prompts=prompts,
    trials=5,
).run()

optimized = await Benchmark(
    backend=VLLMConfig(
        model="llama-3.1-8b",
        enable_prefix_caching=True,
    ),
    prompts=prompts,  # Same prompts
    trials=5,
).run()

diff = compare(baseline, optimized)
diff.print()
```

**Output:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Comparison: baseline vs prefix_caching (5 trials each)          │
├──────────────────┬───────────────┬───────────────┬──────────────┤
│ Metric           │ Baseline      │ Optimized     │ Change       │
├──────────────────┼───────────────┼───────────────┼──────────────┤
│ TTFT p50 (ms)    │ 45.2 ± 2.1    │ 12.3 ± 0.8    │ -72.8% ***   │
│ TTFT p99 (ms)    │ 89.1 ± 4.3    │ 28.4 ± 2.1    │ -68.1% ***   │
│ ITL p50 (ms)     │ 8.2 ± 0.3     │ 8.1 ± 0.2     │ -1.2%        │
│ Throughput (t/s) │ 1,245 ± 42    │ 1,312 ± 38    │ +5.4% *      │
└──────────────────┴───────────────┴───────────────┴──────────────┘
Statistical significance: *** p<0.001, ** p<0.01, * p<0.05
```

**Why This Matters:**
- This is the core workflow for optimization validation
- Statistical significance testing is something `vllm bench` doesn't do
- Makes Splleed the obvious choice for rigorous A/B testing

---

### 2. Regression Detection for CI (High Priority)

**The Problem:** Teams want to catch performance regressions in CI before merging.

**The Solution:**

```python
from splleed import Benchmark, RegressionCheck

# Load baseline from previous run
baseline = RegressionCheck.load_baseline("benchmarks/baseline.json")

# Run current benchmark
current = await Benchmark(
    backend=VLLMConfig(model="llama-3.1-8b"),
    prompts=load_prompts("benchmarks/prompts.json"),
    trials=3,
).run()

# Check for regressions
check = RegressionCheck(
    baseline=baseline,
    current=current,
    thresholds={
        "ttft_p99_ms": 0.10,      # Fail if >10% regression
        "throughput_tokens_per_sec": -0.05,  # Fail if >5% slower
    },
)

if check.has_regressions():
    print(check.report())
    sys.exit(1)

# Update baseline if this is a release
if os.environ.get("UPDATE_BASELINE"):
    current.save("benchmarks/baseline.json")
```

**CI Integration (GitHub Actions):**

```yaml
- name: Performance regression check
  run: python benchmarks/check_regression.py
  env:
    UPDATE_BASELINE: ${{ github.ref == 'refs/heads/main' }}
```

**Why This Matters:**
- Catches regressions before they ship
- Python API makes this natural (vs CLI scripting)
- Threshold-based checks are more useful than "just show me numbers"

---

### 3. GPU Utilization Overlay (Medium Priority)

**The Problem:** After a benchmark, engineers want to know: "Was the GPU even busy, or was something else the bottleneck?"

**The Solution:**

```python
results = await Benchmark(
    backend=VLLMConfig(model="llama-3.1-8b"),
    prompts=prompts,
    gpu_metrics=True,  # Enable NVML polling
).run()

results.print()
```

**Output:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Benchmark Results                                                │
├──────────────────┬──────────────────────────────────────────────┤
│ Concurrency      │ 1          4          8          16          │
├──────────────────┼──────────────────────────────────────────────┤
│ TTFT p50 (ms)    │ 23.4       25.1       28.9       45.2        │
│ Throughput (t/s) │ 312        1,102      1,845      2,012       │
├──────────────────┼──────────────────────────────────────────────┤
│ GPU Util Mean    │ 45%        78%        92%        94%         │
│ GPU Util Min     │ 12%        65%        88%        91%         │
│ GPU Memory       │ 18.2 GB    18.4 GB    18.8 GB    19.1 GB     │
└──────────────────┴──────────────────────────────────────────────┘
```

**Why This Matters:**
- Shows if you're compute-bound or something else
- Low GPU util + high latency = scheduling/batching problem
- High GPU util + high latency = need faster kernels
- Simple to implement with `pynvml`

**Implementation:**

```python
import pynvml

class GPUMetricsCollector:
    def __init__(self, interval: float = 0.1):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples = []

    async def collect_sample(self):
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.samples.append(GPUSample(
            sm_util=util.gpu,
            mem_util=util.memory,
            mem_used_gb=mem.used / 1e9,
        ))
```

---

### 4. Workload Fingerprinting (Medium Priority)

**The Problem:** Benchmark results are meaningless without knowing what workload was used.

**The Solution:**

```python
results = await Benchmark(...).run()
results.print()
```

**Output includes:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Workload Characteristics                                         │
├──────────────────────────────────────────────────────────────────┤
│ Prompts: 100                                                     │
│ Prompt length: p50=142, p95=312, p99=528 tokens                 │
│ Output length: p50=89, p95=156, p99=200 tokens (max_tokens=200) │
│ Common prefix: 12 tokens (system prompt detected)                │
│ Dataset: custom (hash: 8f3a2b1c)                                 │
└──────────────────────────────────────────────────────────────────┘
```

**Why This Matters:**
- Reproducibility: others can understand what was tested
- Comparability: apples-to-apples comparison requires similar workloads
- Debugging: "Oh, this benchmark has long prompts—that explains the TTFT"

---

### 5. Profile-Aligned Timestamps (Lower Priority)

**The Problem:** Engineers run Nsight alongside benchmarks but can't correlate request timing with GPU traces.

**The Solution:**

```python
results = await Benchmark(
    ...,
    emit_markers=True,  # Emit NVTX markers for Nsight
).run()

# Or export for manual correlation
results.export_timeline("timeline.json")
```

**Timeline format:**

```json
{
  "requests": [
    {"id": 0, "start_ns": 1234567890, "first_token_ns": 1234590000, "end_ns": 1234890000},
    {"id": 1, "start_ns": 1234567900, "first_token_ns": 1234591000, "end_ns": 1234895000}
  ]
}
```

**Why This Matters:**
- Engineers can overlay this on Nsight traces
- "Request 42 had high TTFT—let me see what the GPU was doing at that timestamp"
- Bridges the gap between validation and diagnosis

---

## What NOT to Build

Based on our analysis, these would be low-value:

| Feature | Why Skip It |
|---------|-------------|
| Prometheus scraping | Aggregate metrics don't explain per-request behavior |
| Server-side instrumentation | Requires engine changes, not Splleed changes |
| Full profiler integration | Nsight/torch.profiler already exist and are better |
| Prefill/decode breakdown | Not exposed by engines in a scrapable way |

---

## Implementation Priority

| Feature | Value | Effort | Priority |
|---------|-------|--------|----------|
| A/B comparison mode | Very High | Medium | **P0** |
| Regression detection | High | Low | **P0** |
| GPU utilization overlay | Medium | Low | **P1** |
| Workload fingerprinting | Medium | Low | **P1** |
| Profile timestamps | Low-Medium | Low | **P2** |

---

## Positioning Strategy

Don't compete with:
- `vllm bench serve` for quick internal tests
- Nsight/torch.profiler for diagnosis

Compete on:
- **Statistical rigor** for publishable results
- **A/B comparison** with significance testing
- **CI integration** for regression detection
- **Neutrality** for competitive benchmarks
- **Multi-backend** for vLLM vs TGI comparisons

---

## Example: The Full Workflow

```python
from splleed import Benchmark, compare, RegressionCheck

# 1. Run A/B comparison during development
baseline = await Benchmark(backend=VLLMConfig(model="llama"), prompts=prompts, trials=5).run()
optimized = await Benchmark(backend=VLLMConfig(model="llama", chunked_prefill=True), prompts=prompts, trials=5).run()

diff = compare(baseline, optimized)
diff.print()  # Shows statistical significance

# 2. If optimization works, update CI baseline
optimized.save("benchmarks/baseline.json")

# 3. CI runs regression checks on every PR
check = RegressionCheck.from_baseline("benchmarks/baseline.json", current_results)
assert not check.has_regressions(), check.report()

# 4. Published results use Splleed for credibility
# "Benchmarks performed using Splleed v1.2.0 with 5 trials and 95% CIs"
```

This positions Splleed as the **rigorous, neutral benchmark tool** that complements (rather than replaces) internal tooling.
