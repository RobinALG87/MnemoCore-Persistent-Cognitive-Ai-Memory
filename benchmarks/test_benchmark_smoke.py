"""
Smoke test for benchmark suite.

Quick test to verify all components work.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks import (
    BenchmarkConfig,
    BenchmarkRunner,
    LatencyBenchmark,
    ThroughputBenchmark,
    MemoryFootprintBenchmark,
    RegressionDetector,
)


async def test_smoke():
    """Quick smoke test of benchmark components."""
    print("=" * 72)
    print("BENCHMARK SUITE SMOKE TEST")
    print("=" * 72)

    # Test 1: Config creation
    print("\n[1/5] Testing BenchmarkConfig...")
    config = BenchmarkConfig(
        dimension=1024,
        n_samples=100,
        output_dir="test_benchmark_results",
    )
    print(f"  Created config: dimension={config.dimension}, samples={config.n_samples}")

    # Test 2: Latency benchmark instantiation
    print("\n[2/5] Testing LatencyBenchmark...")
    latency_bench = LatencyBenchmark(config)
    print(f"  Created LatencyBenchmark")

    # Test 3: Throughput benchmark instantiation
    print("\n[3/5] Testing ThroughputBenchmark...")
    throughput_bench = ThroughputBenchmark(config)
    print(f"  Created ThroughputBenchmark")

    # Test 4: Memory footprint benchmark instantiation
    print("\n[4/5] Testing MemoryFootprintBenchmark...")
    footprint_bench = MemoryFootprintBenchmark(config)
    print(f"  Created MemoryFootprintBenchmark")

    # Test 5: Regression detector instantiation
    print("\n[5/5] Testing RegressionDetector...")
    detector = RegressionDetector(baseline_dir="test_baselines")
    print(f"  Created RegressionDetector")

    print("\n" + "=" * 72)
    print("SMOKE TEST PASSED")
    print("=" * 72)
    print("\nAll benchmark components can be imported and instantiated.")
    print("To run actual benchmarks, use:")
    print("  python -m benchmarks.run_benchmarks --quick")


if __name__ == "__main__":
    asyncio.run(test_smoke())
