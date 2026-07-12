# MnemoCore release checklist

## Status: working prototype, not production-ready

The authoritative evidence is recorded in
[the platform status](docs/status/2026-07-12-platform-baseline.md). Historical
remediation documents are context, not green release evidence.

## Verified prototype gates

- [x] AgentMemory core: 308 focused tests passed, including four physical
  erasure contract tests.
- [x] Offline adapters: 6 tests passed.
- [x] Benchmark contracts/smoke: 18 tests passed.
- [x] Combined AgentMemory, webhook security, adapter, benchmark, packaging,
  runtime, and deployment batch: 380 passed, 2 skipped because the local
  build-backend metadata prerequisite was unavailable.
- [x] Persistent webhook signing secrets are represented only by `secret_ref`;
  inline secrets, unsafe legacy files, and persistent custom headers fail
  closed. Mutations are serialized and atomically persisted.
- [x] Physical erase prototype enforces exact scope, protects supersession
  components unless cascade is explicit, removes dependent rows, validates
  integrity, and returns a content-free receipt.
- [x] Docker image builds a wheel and installs the package.
- [x] A real container starts `mnemocore.api.main:app` as a non-root user.
- [x] Container smoke returned `/health` (live/degraded without Redis), `/ready`
  (`ready`), and Prometheus output from `/metrics/` on port 8100.
- [x] Compose requires `HAIM_API_KEY`, `REDIS_PASSWORD`, and `QDRANT_API_KEY`.
- [x] CI includes Docker in its fail-closed summary and exercises the runtime
  endpoints.
- [x] Helm defaults are aligned to the single-node prototype and port 8100.

## Release blockers

- [ ] Resolve or explicitly accept current Bandit and dependency-audit findings.
- [ ] Extend physical erasure beyond its cooperative-lock SQLite prototype:
  prove arbitrary power-loss/failure points, uncoordinated-client handling,
  backup deletion, and external derived-artifact deletion.
- [ ] Migrate or retire the quarantined lifecycle integration file.
- [ ] Make the legacy unit lane terminate reliably and record its result.
- [ ] Add non-zero Redis/Qdrant service-lane coverage.
- [ ] Run the full supported Python matrix and build wheel/sdist from a clean
  environment without skipped packaging assertions.
- [ ] Validate Compose persistence/restart and backup/restore.
- [ ] Validate Helm dependency locking, secrets, storage, rollback, and recovery
  on a target cluster.
- [ ] Run load/soak tests and compare published benchmark artifacts against the
  accepted regression threshold.
- [ ] Complete a final public security review and secret-pattern scan.

## Candidate verification sequence

```bash
python -m build
python -m pytest tests/agent_memory tests/integrations \
  benchmarks/test_agent_memory_baseline.py benchmarks/test_benchmark_smoke.py -q
python -m pytest tests/events/test_webhook_secret_persistence.py -q
python -m pytest tests/test_version_contract.py tests/test_packaging_smoke.py \
  tests/test_api_health_runtime.py tests/test_docker_runtime_contract.py \
  tests/deployment/test_deployment_contracts.py -q
ruff check src tests benchmarks
bandit -r src/mnemocore -q
pip-audit -l
docker build -t mnemocore:release-candidate .
docker compose config
helm dependency build helm/mnemocore
helm lint helm/mnemocore
```

Then run a container and require successful responses from `/health`, `/ready`,
and `/metrics/` on port 8100. A release is not green merely because the focused
prototype checks pass; every blocker above needs direct evidence or a recorded
risk acceptance.

## Publication gate

Before tagging or publishing:

- [ ] Worktree and version authority are clean and intentional.
- [ ] Release notes distinguish stable AgentMemory from legacy/experimental
  surfaces.
- [ ] No real credentials, private runtime policies, or internal paths are in
  the public diff or artifacts.
- [ ] Upgrade, rollback, backup, restore, and erasure procedures are rehearsed.
- [ ] CI is green without unexpected skips.
- [ ] Image digest and Python artifacts are recorded and scanned.
