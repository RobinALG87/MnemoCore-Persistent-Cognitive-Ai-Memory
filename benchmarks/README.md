# MnemoCore Benchmark Suite

Comprehensive benchmarking framework for MnemoCore memory system.

## Overview

The benchmark suite measures:

1. **Latency** - Per-tier operation latency (store, recall, synthesize)
2. **Throughput** - Operations per second under concurrent load
3. **Memory Footprint** - RAM and disk usage at scale
4. **Regression Detection** - Alert when performance degrades >10%
5. **System Comparison** - Framework for comparing vs MemGPT, Zep, LangMem

## Installation

```bash
# Install with benchmark dependencies
pip install -e ".[benchmark]"

# Or install pytest-benchmark separately
pip install pytest-benchmark psutil
```

## Quick Start

### Run All Benchmarks

```bash
# Run full benchmark suite
python -m benchmarks.run_benchmarks

# Or using the CLI script
python benchmarks/run_benchmarks.py

# Quick CI suite (reduced dataset)
python -m benchmarks.run_benchmarks --quick
```

### Run Individual Benchmarks

```bash
# Latency benchmark
python -m benchmarks.run_benchmarks latency

# Throughput benchmark
python -m benchmarks.run_benchmarks throughput --workers 20

# Memory footprint benchmark
python -m benchmarks.run_benchmarks footprint --max-scale 50000
```

### Using Pytest

```bash
# Run all pytest benchmarks
pytest benchmarks/pytest_benchmarks.py -v

# Run only latency benchmarks
pytest benchmarks/pytest_benchmarks.py -k latency -v

# Run with pytest-benchmark output
pytest benchmarks/pytest_benchmarks.py --benchmark-only

# Generate histogram
pytest benchmarks/pytest_benchmarks.py --benchmark-only --benchmark-histogram
```

## Benchmark Results

Results are saved to `benchmark_results/` directory as JSON files:

```
benchmark_results/
├── latency_tier_latency_20240220_120000.json
├── throughput_concurrent_throughput_20240220_120100.json
├── footprint_memory_footprint_20240220_120200.json
├── suite_full_suite_20240220_120300.json
└── baselines/
    ├── latency_tier_latency.json
    ├── throughput_concurrent_throughput.json
    └── footprint_memory_footprint.json
```

## Regression Detection

### Initialize Baselines

```bash
# Create baselines from current results
python -m benchmarks.run_benchmarks init-baselines
```

### Detect Regressions

```bash
# Check a specific result file against baselines
python -m benchmarks.run_benchmarks detect benchmark_results/latency_*.json
```

Regression thresholds (default):
- P99 latency: +10% = warning
- Throughput: -10% = warning
- Memory footprint: +20% = warning

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[benchmark]"

      - name: Run benchmarks
        run: |
          python -m benchmarks.run_benchmarks --quick

      - name: Check for regressions
        run: |
          python -m benchmarks.run_benchmarks init-baselines || true
          python -m benchmarks.run_benchmarks detect

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results/
```

## Performance SLOs

Current Service Level Objectives:

| Metric | Target |
|--------|--------|
| Store P99 latency | < 50ms |
| Query P99 latency | < 50ms |
| Throughput | > 100 ops/sec |
| Memory per 1K | < 50MB RAM |

## Architecture

```
benchmarks/
├── __init__.py           # Package exports
├── base.py               # Base classes and utilities
├── latency.py            # Latency benchmarks per tier
├── throughput.py         # Throughput benchmarks
├── memory_footprint.py   # Memory footprint benchmarks
├── regression.py         # Regression detection
├── comparison.py         # System comparison framework
├── runner.py             # Main orchestrator
├── run_benchmarks.py     # CLI script
├── pytest_benchmarks.py  # Pytest integration
└── README.md             # This file
```

## Programmatic Usage

```python
import asyncio
from benchmarks.runner import BenchmarkRunner, BenchmarkConfig

async def run_custom_benchmark():
    config = BenchmarkConfig(
        dimension=8192,
        n_samples=5000,
        output_dir="my_benchmarks",
    )

    runner = BenchmarkRunner(config)

    # Run individual benchmarks
    latency_result = await runner.run_latency()
    throughput_result = await runner.run_throughput()

    print(f"Latency P99: {latency_result.p99_ms:.2f}ms")
    print(f"Throughput: {throughput_result.throughput_ops:.2f} ops/sec")

asyncio.run(run_custom_benchmark())
```

## Extending the Suite

To add a new benchmark:

1. Create a new class extending `BenchmarkBase` in `base.py`
2. Implement the `run()` method returning a `BenchmarkResult`
3. Add the benchmark to `runner.py`

```python
from benchmarks.base import BenchmarkBase, BenchmarkResult

class MyBenchmark(BenchmarkBase):
    async def run(self) -> BenchmarkResult:
        await self.setup()

        try:
            # Your benchmark logic here
            latencies = []
            for _ in range(self.config.n_samples):
                start = time.time()
                await self.engine.store("test")
                latencies.append((time.time() - start) * 1000)

            # Compute stats
            stats = compute_percentiles(latencies)

            return BenchmarkResult(
                name="my_benchmark",
                category="custom",
                timestamp=time.time(),
                duration_ms=sum(latencies),
                samples=len(latencies),
                mean_ms=stats["mean"],
                median_ms=stats["median"],
                p50_ms=stats["p50"],
                p95_ms=stats["p95"],
                p99_ms=stats["p99"],
                min_ms=stats["min"],
                max_ms=stats["max"],
            )
        finally:
            await self.teardown()
```

## System Comparison (Future)

The comparison framework (`comparison.py`) provides adapters for:

- **MemGPT** - Placeholder, needs client implementation
- **Zep** - Placeholder, needs client implementation
- **LangMem** - Placeholder, needs client implementation

To add a comparison:

```python
from benchmarks.comparison import ComparisonBenchmark, MySystemAdapter

config = BenchmarkConfig(dimension=4096)
comparison = ComparisonBenchmark(config)

comparison.add_competitor(MySystemAdapter(config))
result = await comparison.run()
```

## License

MIT License - See main LICENSE file.
