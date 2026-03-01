"""
Base benchmark framework with utilities for MnemoCore benchmarking.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import psutil
from loguru import logger
from statistics import mean, median, stdev

# Add src to path so 'mnemocore' is importable
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.config import reset_config, get_config


T = TypeVar("T")


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    category: str
    timestamp: str
    duration_ms: float
    samples: int

    # Statistics
    mean_ms: float
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: Optional[float] = None

    # Throughput (operations per second)
    throughput_ops: Optional[float] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tier_stats: Optional[Dict[str, Any]] = None

    # System info
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def get_summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Benchmark: {self.name}",
            f"Category: {self.category}",
            f"Samples: {self.samples:,}",
            f"Duration: {self.duration_ms:.2f} ms",
            "",
            f"Latency (ms):",
            f"  Mean:   {self.mean_ms:.4f}",
            f"  Median: {self.median_ms:.4f}",
            f"  P50:    {self.p50_ms:.4f}",
            f"  P95:    {self.p95_ms:.4f}",
            f"  P99:    {self.p99_ms:.4f}",
            f"  Min:    {self.min_ms:.4f}",
            f"  Max:    {self.max_ms:.4f}",
        ]
        if self.stdev_ms is not None:
            lines.append(f"  Stdev:  {self.stdev_ms:.4f}")
        if self.throughput_ops is not None:
            lines.append(f"Throughput: {self.throughput_ops:.2f} ops/sec")
        if self.tier_stats:
            lines.append("")
            lines.append("Tier Statistics:")
            for tier, stats in self.tier_stats.items():
                lines.append(f"  {tier.upper()}: {stats}")
        return "\n".join(lines)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Dataset sizes
    n_memories: int = 10000
    n_warmup: int = 100
    n_samples: int = 1000

    # Engine config
    dimension: int = 16384
    persist_path: Optional[str] = None

    # Concurrency
    concurrent_workers: int = 10
    batch_size: int = 100

    # Tier sizes for benchmark
    hot_size: int = 5000
    warm_size: int = 50000
    cold_size: int = 100000

    # Output
    output_dir: str = "benchmark_results"
    save_results: bool = True

    # Memory tracking
    track_memory: bool = True
    track_cpu: bool = True

    def __post_init__(self):
        if self.persist_path is None:
            # Create a temp path for benchmarks
            temp_dir = Path(self.output_dir) / "temp_data"
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.persist_path = str(temp_dir / f"bench_{uuid.uuid4().hex[:8]}")


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute percentiles from a list of values."""
    if not values:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "stdev": 0.0,
        }

    sorted_values = sorted(values)
    n = len(sorted_values)

    return {
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": mean(values),
        "median": median(values),
        "p50": sorted_values[int(n * 0.50)] if n > 0 else 0.0,
        "p95": sorted_values[int(n * 0.95)] if n > 0 else 0.0,
        "p99": sorted_values[int(n * 0.99)] if n > 0 else 0.0,
        "stdev": stdev(values) if len(values) > 1 else 0.0,
    }


class SystemMetrics:
    """Track system resource usage during benchmarks."""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)

    def get_disk_usage_mb(self, path: str) -> float:
        """Get disk usage in MB for a given path."""
        if not os.path.exists(path):
            return 0.0
        total = 0.0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except OSError:
                    pass
        return total / 1024 / 1024


class BenchmarkBase(ABC):
    """Base class for all benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics = SystemMetrics()
        self._engine: Optional[HAIMEngine] = None
        self._results: List[BenchmarkResult] = []

    @property
    def engine(self) -> HAIMEngine:
        """Lazy initialize engine."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def _create_engine(self) -> HAIMEngine:
        """Create a fresh HAIMEngine instance."""
        os.environ["HAIM_DIMENSIONALITY"] = str(self.config.dimension)
        reset_config()
        return HAIMEngine(persist_path=self.config.persist_path)

    async def setup(self) -> None:
        """Setup benchmark environment."""
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize engine
        await self.engine.initialize()
        logger.info(f"Benchmark setup complete. Engine dimension: {self.config.dimension}")

    async def teardown(self) -> None:
        """Cleanup after benchmark."""
        if self._engine:
            await self._engine.close()
            self._engine = None
        reset_config()

    async def warmup(self, n: Optional[int] = None) -> None:
        """Warmup phase for the engine."""
        n = n or self.config.n_warmup
        logger.info(f"Warming up with {n} operations...")
        for i in range(n):
            await self.engine.store(
                f"warmup memory {i}",
                metadata={"warmup": True, "index": i}
            )
        logger.info("Warmup complete.")

    def generate_memories(self, count: int, prefix: str = "bench") -> List[str]:
        """Generate synthetic memory content."""
        return [
            f"{prefix}_memory_{i:06d}_with_signal_{i % 97}"
            for i in range(count)
        ]

    def generate_queries(self, count: int) -> List[str]:
        """Generate synthetic queries."""
        return [
            f"signal {(i * 7) % 97}"
            for i in range(count)
        ]

    @abstractmethod
    async def run(self) -> BenchmarkResult:
        """Run the benchmark. Must be implemented by subclasses."""
        pass

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a result to the collection."""
        self._results.append(result)

    def get_results(self) -> List[BenchmarkResult]:
        """Get all collected results."""
        return self._results.copy()


class TimedContext:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000.0


async def measure_async(
    func: Callable,
    *args,
    **kwargs
) -> Tuple[Any, float]:
    """Measure async function execution time."""
    start = time.perf_counter()
    result = await func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def save_result(result: BenchmarkResult, output_dir: str) -> str:
    """Save benchmark result to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{result.category}_{result.name}_{timestamp}.json"
    filepath = Path(output_dir) / filename

    with open(filepath, "w") as f:
        f.write(result.to_json())

    logger.info(f"Saved benchmark result to {filepath}")
    return str(filepath)


def load_result(filepath: str) -> BenchmarkResult:
    """Load benchmark result from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return BenchmarkResult(**data)


__all__ = [
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkBase",
    "SystemMetrics",
    "TimedContext",
    "compute_percentiles",
    "save_result",
    "load_result",
    "measure_async",
]
