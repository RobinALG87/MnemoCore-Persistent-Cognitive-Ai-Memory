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

# Offline/local integration (adapters plus the mocked store/query lifecycle)
python -m pytest tests/integrations tests/test_integration_store_query_cycle.py --run-integration -q

# Service integration; each test declares only the service(s) it actually needs
python -m pytest tests --run-integration -m "integration and (requires_redis or requires_qdrant) and not slow" -q

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
python -m pytest tests --run-integration --run-slow -m "integration and (requires_redis or requires_qdrant) and slow" -q
```

`--run-integration` controls external service discovery. Without it, collection
does not connect to Redis or Qdrant. Availability is checked at most once per
pytest process when a selected test declares `requires_redis` or
`requires_qdrant`.

An integration test without either service marker is offline/local. A service
marker without `integration` is rejected during collection so service tests
cannot become orphaned from the opt-in lane. The service lane may therefore
collect zero tests in a checkout that has no genuine external-service tests.
