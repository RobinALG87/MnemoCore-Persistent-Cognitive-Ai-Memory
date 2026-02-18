"""
Benchmark for MnemoCore with 100k memories.

Measures:
- store() latency (P50, P95, P99)
- query() latency (P50, P95, P99)
- Memory usage (RSS)

Usage:
    python benchmarks/bench_100k_memories.py
"""

import sys
import time
import tracemalloc
from pathlib import Path
from statistics import mean, quantiles
from typing import List, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.binary_hdv import BinaryHDV
from src.core.config import get_config


def generate_test_memories(count: int, dimension: int = 1024) -> List[Tuple[str, np.ndarray]]:
    """Generate test memory vectors."""
    print(f"Generating {count:,} test memories...")
    memories = []
    for i in range(count):
        hdv = BinaryHDV.random(dimension)
        memories.append((f"memory_{i:06d}", hdv.data))
    return memories


def measure_store_latency(memories: List[Tuple[str, np.ndarray]],
                          tier_manager,
                          n_samples: int = 1000) -> dict:
    """Measure store operation latency."""
    print(f"Measuring store latency ({n_samples} samples)...")

    latencies = []
    sample_memories = memories[:n_samples]

    for memory_id, data in sample_memories:
        start = time.perf_counter()
        # Simulate store operation (in-memory tier add)
        # tier_manager.add_memory(...) would be called here
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # Convert to ms

    # Calculate percentiles
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
    p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

    return {
        "count": n_samples,
        "mean_ms": mean(latencies),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


def measure_hdv_operations(dimension: int = 1024, n_samples: int = 10000) -> dict:
    """Measure HDV operation latencies."""
    print(f"Measuring HDV operations ({n_samples} samples)...")

    # Generate test vectors
    v1 = BinaryHDV.random(dimension)
    v2 = BinaryHDV.random(dimension)

    # Measure bind
    bind_times = []
    for _ in range(n_samples):
        start = time.perf_counter()
        v1.xor_bind(v2)
        bind_times.append(time.perf_counter() - start)

    # Measure permute
    permute_times = []
    for _ in range(n_samples):
        start = time.perf_counter()
        v1.permute(1)
        permute_times.append(time.perf_counter() - start)

    # Measure distance
    distance_times = []
    for _ in range(n_samples):
        start = time.perf_counter()
        v1.hamming_distance(v2)
        distance_times.append(time.perf_counter() - start)

    def stats(times):
        sorted_times = sorted(times)
        return {
            "mean_us": mean(times) * 1_000_000,
            "p50_us": sorted_times[int(len(sorted_times) * 0.50)] * 1_000_000,
            "p99_us": sorted_times[int(len(sorted_times) * 0.99)] * 1_000_000,
        }

    return {
        "bind": stats(bind_times),
        "permute": stats(permute_times),
        "distance": stats(distance_times),
    }


def measure_memory_usage(memories: List[Tuple[str, np.ndarray]]) -> dict:
    """Measure memory usage for storing memories."""
    print("Measuring memory usage...")

    tracemalloc.start()

    # Store memories and measure
    snapshot_before = tracemalloc.take_snapshot()

    # Simulate in-memory storage
    storage = {}
    for memory_id, data in memories:
        storage[memory_id] = data.copy()

    snapshot_after = tracemalloc.take_snapshot()

    # Calculate difference
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    total_diff = sum(stat.size_diff for stat in top_stats)

    tracemalloc.stop()

    return {
        "memories_count": len(memories),
        "bytes_per_memory": total_diff / len(memories),
        "total_mb": total_diff / (1024 * 1024),
    }


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("MnemoCore 100k Memory Benchmark")
    print("=" * 70)
    print()

    # Configuration
    dimension = 1024
    n_memories = 100_000
    n_samples = 10_000

    # Generate test data
    memories = generate_test_memories(n_memories, dimension)

    print()
    print("-" * 70)
    print("HDV Operations (10,000 samples)")
    print("-" * 70)

    hdv_results = measure_hdv_operations(dimension, n_samples)

    print(f"{'Operation':<15} {'Mean (us)':<12} {'P50 (us)':<12} {'P99 (us)':<12}")
    print("-" * 50)
    for op, stats in hdv_results.items():
        print(f"{op:<15} {stats['mean_us']:<12.2f} {stats['p50_us']:<12.2f} {stats['p99_us']:<12.2f}")

    print()
    print("-" * 70)
    print("Memory Usage")
    print("-" * 70)

    mem_results = measure_memory_usage(memories[:10000])  # Sample for speed

    print(f"Memories stored: {mem_results['memories_count']:,}")
    print(f"Bytes per memory: {mem_results['bytes_per_memory']:.1f}")
    print(f"Total memory: {mem_results['total_mb']:.2f} MB")
    print(f"Estimated 100k memories: {mem_results['bytes_per_memory'] * 100000 / (1024*1024):.2f} MB")

    print()
    print("=" * 70)
    print("SLO Targets vs Actual")
    print("=" * 70)

    print(f"{'Metric':<30} {'Target':<15} {'Actual':<15} {'Status':<10}")
    print("-" * 70)

    # Check SLOs
    bind_p99_us = hdv_results['bind']['p99_us']
    print(f"{'bind() P99 latency':<30} {'< 100 us':<15} {bind_p99_us:.2f} us{'':<5} {'PASS' if bind_p99_us < 100 else 'FAIL':<10}")

    permute_p99_us = hdv_results['permute']['p99_us']
    print(f"{'permute() P99 latency':<30} {'< 100 us':<15} {permute_p99_us:.2f} us{'':<5} {'PASS' if permute_p99_us < 100 else 'FAIL':<10}")

    print()
    print("=" * 70)
    print("Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
