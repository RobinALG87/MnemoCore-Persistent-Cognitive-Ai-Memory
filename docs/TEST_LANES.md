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

# Required offline adapters (no Redis or Qdrant probes)
python -m pytest tests/integrations -q

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

## Quarantined known-red lifecycle

`tests/test_integration_store_query_cycle.py` is an offline/local integration
module, but it is not part of required CI. It is quarantined because its fixture
still uses an obsolete `AsyncQdrantClient` patch target and several assertions
encode stale lifecycle behavior assumptions. Keep it discoverable, but do not
add it to a required green lane.

**Unquarantine Requirements:**
1. Migrate engine initialization and operations (e.g. `.store()`, `.query()`) from `HAIMEngine` to the modern `AgentMemory` interface.
2. Fix obsolete client/mock patching targets (e.g. `AsyncQdrantClient`).
3. Align all assertions with the updated bitemporal timelines and working memory service properties.
4. Ensure the entire file passes when running `pytest tests/test_integration_store_query_cycle.py`.

## Service lane: inactive

There are currently no tests carrying `integration` plus `requires_redis` or
`requires_qdrant`, so there is no required CI command for service integration.
Activate this lane only after its marker expression has a nonzero collection
and every collected test passes against its declared service or services. At
that point, each test must declare only the services it actually needs; the
lane must not impose a blanket requirement for both Redis and Qdrant.

`--run-integration` controls external service discovery. Without it, collection
does not connect to Redis or Qdrant. Availability is checked at most once per
pytest process when a selected test declares `requires_redis` or
`requires_qdrant`.

An integration test without either service marker is offline/local. A service
marker without `integration` is rejected during collection so service tests
cannot become orphaned from the future opt-in lane.
