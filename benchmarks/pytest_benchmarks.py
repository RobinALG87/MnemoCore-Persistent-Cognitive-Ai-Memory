"""
Pytest-based benchmarks for MnemoCore.

These benchmarks can be run with pytest and pytest-benchmark for CI/CD integration.

Usage:
    pytest benchmarks/pytest_benchmarks.py --benchmark-only
    pytest benchmarks/pytest_benchmarks.py::test_store_latency -v
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import Generator

import sys
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.config import reset_config
import os


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_persist_dir() -> Generator[str, None, None]:
    """Create a temporary directory for benchmark data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
async def engine(temp_persist_dir: str) -> Generator[HAIMEngine, None, None]:
    """Create a HAIMEngine instance for benchmarking."""
    os.environ["HAIM_DIMENSIONALITY"] = "4096"
    reset_config()

    engine = HAIMEngine(persist_path=temp_persist_dir)
    await engine.initialize()

    yield engine

    await engine.close()
    reset_config()


@pytest.fixture
def benchmark_memories() -> list[str]:
    """Generate test memory content."""
    return [f"benchmark memory #{i:06d} with signal {i % 97}" for i in range(1000)]


@pytest.fixture
def benchmark_queries() -> list[str]:
    """Generate test queries."""
    return [f"signal {(i * 7) % 97}" for i in range(100)]


# ============================================================================
# Store Latency Benchmarks
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_store_latency_single(engine: HAIMEngine, benchmark, benchmark_memories: list[str]):
    """Benchmark single store operation latency."""

    async def store_single():
        import time
        content = benchmark_memories[0]
        start = time.perf_counter()
        await engine.store(content)
        return (time.perf_counter() - start) * 1000

    result = benchmark(asyncio.run, store_single())
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_store_latency_batch(engine: HAIMEngine, benchmark, benchmark_memories: list[str]):
    """Benchmark batch store operation latency."""

    async def store_batch():
        tasks = [engine.store(m) for m in benchmark_memories[:100]]
        await asyncio.gather(*tasks)

    result = benchmark(asyncio.run, store_batch())
    assert result is not None


# ============================================================================
# Query/Recall Latency Benchmarks
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_query_latency(engine: HAIMEngine, benchmark, benchmark_memories: list[str], benchmark_queries: list[str]):
    """Benchmark query operation latency."""
    # First, store some memories
    for content in benchmark_memories[:100]:
        await engine.store(content)

    async def query_single():
        import time
        query = benchmark_queries[0]
        start = time.perf_counter()
        await engine.query(query, top_k=5)
        return (time.perf_counter() - start) * 1000

    result = benchmark(asyncio.run, query_single())
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_query_latency_top_k(engine: HAIMEngine, benchmark, benchmark_memories: list[str], benchmark_queries: list[str]):
    """Benchmark query with varying top_k values."""
    # Store memories
    for content in benchmark_memories[:100]:
        await engine.store(content)

    async def query_top10():
        await engine.query(benchmark_queries[0], top_k=10)

    result = benchmark(asyncio.run, query_top10())
    assert result is not None


# ============================================================================
# Throughput Benchmarks
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_concurrent_store_throughput(engine: HAIMEngine, benchmark, benchmark_memories: list[str]):
    """Benchmark concurrent store throughput."""

    async def concurrent_stores():
        tasks = [engine.store(m) for m in benchmark_memories[:50]]
        await asyncio.gather(*tasks)

    result = benchmark(asyncio.run, concurrent_stores())
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_concurrent_query_throughput(engine: HAIMEngine, benchmark, benchmark_memories: list[str], benchmark_queries: list[str]):
    """Benchmark concurrent query throughput."""
    # Store memories first
    for content in benchmark_memories[:100]:
        await engine.store(content)

    async def concurrent_queries():
        tasks = [engine.query(q, top_k=5) for q in benchmark_queries[:20]]
        await asyncio.gather(*tasks)

    result = benchmark(asyncio.run, concurrent_queries())
    assert result is not None


# ============================================================================
# Tier Operation Benchmarks
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_hot_tier_store(engine: HAIMEngine, benchmark, temp_persist_dir: str):
    """Benchmark HOT tier store operations."""
    # Small enough to stay in HOT tier
    hot_size = engine.tier_manager.config.tiers_hot.max_memories

    async def fill_hot_tier():
        for i in range(min(100, hot_size)):
            await engine.store(f"hot_tier_memory_{i}")

    result = benchmark(asyncio.run, fill_hot_tier())
    assert result is not None


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_hot_tier_query(engine: HAIMEngine, benchmark):
    """Benchmark HOT tier query operations."""
    # Fill HOT tier
    hot_size = engine.tier_manager.config.tiers_hot.max_memories
    for i in range(min(100, hot_size)):
        await engine.store(f"hot_tier_memory_{i}")

    async def query_hot():
        await engine.query("hot_tier_memory", top_k=5)

    result = benchmark(asyncio.run, query_hot())
    assert result is not None


# ============================================================================
# Memory Footprint (Basic)
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_memory_footprint_1k(engine: HAIMEngine, benchmark_memories: list[str]):
    """Measure memory footprint for 1K memories."""
    import psutil
    process = psutil.Process()

    baseline_mb = process.memory_info().rss / 1024 / 1024

    for content in benchmark_memories[:1000]:
        await engine.store(content)

    after_mb = process.memory_info().rss / 1024 / 1024
    delta_mb = after_mb - baseline_mb

    # Assert memory growth is reasonable (< 100MB for 1K memories)
    assert delta_mb < 100, f"Memory footprint too high: {delta_mb:.2f}MB"

    # Also assert per-memory overhead is reasonable
    per_memory_kb = (delta_mb * 1024) / 1000
    assert per_memory_kb < 50, f"Per-memory overhead too high: {per_memory_kb:.2f}KB"


# ============================================================================
# Reconstructive Recall Benchmarks
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_reconstructive_recall(engine: HAIMEngine, benchmark, benchmark_memories: list[str], benchmark_queries: list[str]):
    """Benchmark reconstructive recall operations."""
    # Store memories
    for content in benchmark_memories[:100]:
        await engine.store(content)

    async def reconstructive_query():
        await engine.reconstructive_recall(
            benchmark_queries[0],
            top_k=10,
            enable_synthesis=True
        )

    result = benchmark(asyncio.run, reconstructive_query())
    assert result is not None


# ============================================================================
# Regression Threshold Checks
# ============================================================================

@pytest.mark.regression
@pytest.mark.asyncio
async def test_regression_store_latency(engine: HAIMEngine, benchmark_memories: list[str]):
    """
    Regression test for store latency.

    Asserts that P99 latency is below 50ms for store operations.
    """
    import time

    latencies = []
    for content in benchmark_memories[:500]:
        start = time.perf_counter()
        await engine.store(content)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    p99 = latencies[int(len(latencies) * 0.99)]

    # Regression threshold: P99 < 50ms
    assert p99 < 50, f"Store latency P99 regression: {p99:.2f}ms > 50ms"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_regression_query_latency(engine: HAIMEngine, benchmark_memories: list[str], benchmark_queries: list[str]):
    """
    Regression test for query latency.

    Asserts that P99 latency is below 50ms for query operations.
    """
    import time

    # Store memories first
    for content in benchmark_memories[:100]:
        await engine.store(content)

    latencies = []
    for query in benchmark_queries[:100]:
        start = time.perf_counter()
        await engine.query(query, top_k=5)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    p99 = latencies[int(len(latencies) * 0.99)]

    # Regression threshold: P99 < 50ms
    assert p99 < 50, f"Query latency P99 regression: {p99:.2f}ms > 50ms"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_regression_throughput(engine: HAIMEngine, benchmark_memories: list[str]):
    """
    Regression test for throughput.

    Asserts minimum throughput of 100 ops/sec for concurrent stores.
    """
    import time

    batch_size = 100
    start = time.perf_counter()

    tasks = [engine.store(m) for m in benchmark_memories[:batch_size]]
    await asyncio.gather(*tasks)

    duration = time.perf_counter() - start
    throughput = batch_size / duration

    # Regression threshold: throughput > 100 ops/sec
    assert throughput > 100, f"Throughput regression: {throughput:.2f} ops/sec < 100"


# ============================================================================
# SLO (Service Level Objective) Tests
# ============================================================================

class SLO:
    """Service Level Objectives for MnemoCore."""

    STORE_P99_MS = 50.0
    QUERY_P99_MS = 50.0
    THROUGHPUT_MIN = 100.0
    MEMORY_PER_1K_MB = 50.0


@pytest.mark.slo
@pytest.mark.asyncio
async def test_slo_store_latency(engine: HAIMEngine, benchmark_memories: list[str]):
    """SLO test: Store P99 latency must be < 50ms."""
    import time

    latencies = []
    for content in benchmark_memories[:1000]:
        start = time.perf_counter()
        await engine.store(content)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    p99 = latencies[int(len(latencies) * 0.99)]

    assert p99 < SLO.STORE_P99_MS, f"SLO violation: store P99 = {p99:.2f}ms > {SLO.STORE_P99_MS}ms"


@pytest.mark.slo
@pytest.mark.asyncio
async def test_slo_query_latency(engine: HAIMEngine, benchmark_memories: list[str], benchmark_queries: list[str]):
    """SLO test: Query P99 latency must be < 50ms."""
    import time

    for content in benchmark_memories[:200]:
        await engine.store(content)

    latencies = []
    for query in benchmark_queries[:200]:
        start = time.perf_counter()
        await engine.query(query, top_k=5)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    p99 = latencies[int(len(latencies) * 0.99)]

    assert p99 < SLO.QUERY_P99_MS, f"SLO violation: query P99 = {p99:.2f}ms > {SLO.QUERY_P99_MS}ms"
