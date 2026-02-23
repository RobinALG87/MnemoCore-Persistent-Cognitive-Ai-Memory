"""
MnemoCore Benchmark Suite

Comprehensive benchmarking framework for MnemoCore memory system.

Provides:
- Latency benchmarks per tier (store, recall, synthesize)
- Throughput benchmarks (concurrent operations)
- Memory footprint analysis
- Regression detection
- Comparison framework for external systems
"""

# Re-export base classes
from .base import (
    BenchmarkResult,
    BenchmarkConfig,
    BenchmarkBase,
    SystemMetrics,
    compute_percentiles,
    save_result,
    load_result,
)

# Re-export runners
from .runner import BenchmarkRunner, BenchmarkSuiteResult

# Re-export benchmarks
from .latency import LatencyBenchmark, benchmark_latency_quick
from .throughput import ThroughputBenchmark, benchmark_throughput_quick
from .memory_footprint import MemoryFootprintBenchmark, benchmark_footprint_quick

# Re-export regression detection
from .regression import (
    RegressionDetector,
    RegressionAlert,
    RegressionReport,
    create_baselines_from_results,
)

# Re-export comparison framework
from .comparison import (
    ComparisonBenchmark,
    ComparisonResult,
    MemorySystemInterface,
    MnemoCoreAdapter,
)

__all__ = [
    # Base
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkBase",
    "SystemMetrics",
    "compute_percentiles",
    "save_result",
    "load_result",
    # Runner
    "BenchmarkRunner",
    "BenchmarkSuiteResult",
    # Benchmarks
    "LatencyBenchmark",
    "benchmark_latency_quick",
    "ThroughputBenchmark",
    "benchmark_throughput_quick",
    "MemoryFootprintBenchmark",
    "benchmark_footprint_quick",
    # Regression
    "RegressionDetector",
    "RegressionAlert",
    "RegressionReport",
    "create_baselines_from_results",
    # Comparison
    "ComparisonBenchmark",
    "ComparisonResult",
    "MemorySystemInterface",
    "MnemoCoreAdapter",
]
