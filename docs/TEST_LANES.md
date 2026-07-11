# Test lanes

The default `python -m pytest` run collects only `tests/`. Performance
benchmarks are deliberately opt-in so a normal developer or CI test run cannot
accidentally execute timing-sensitive workloads.

Run the CI lanes independently:

```bash
# AgentMemory core (SQLite/local-only)
python -m pytest tests/agent_memory -q

# Legacy unit tests; service integration and stress cases stay out
python -m pytest tests --ignore=tests/agent_memory --ignore=tests/integrations --ignore=tests/test_integration_store_query_cycle.py -m "not integration and not slow" -q

# Offline integration adapters (no Redis or Qdrant probes)
python -m pytest tests/integrations -q

# Service integration (start Redis on :6379 and Qdrant on :6333 first)
python -m pytest tests/test_integration_store_query_cycle.py --run-integration -m "integration and not slow" -q

# Benchmark harness contracts and smoke coverage, without the performance suite
python -m pytest benchmarks/test_agent_memory_baseline.py benchmarks/test_benchmark_smoke.py -q
```

Run timing-sensitive lanes on dedicated, scheduled workers rather than on each
pull request:

```bash
python -m pytest benchmarks/pytest_benchmarks.py -m benchmark --benchmark-only -q
python -m pytest benchmarks/pytest_benchmarks.py -m regression -q
python -m pytest benchmarks/pytest_benchmarks.py -m slo -q
```

Slow service stress tests are a separate opt-in extension:

```bash
python -m pytest tests/test_integration_store_query_cycle.py --run-integration --run-slow -m "integration and slow" -q
```

`--run-integration` controls external service discovery. Without it, collection
does not connect to Redis or Qdrant. Availability is checked at most once per
pytest process when a selected test declares `requires_redis` or
`requires_qdrant`.
