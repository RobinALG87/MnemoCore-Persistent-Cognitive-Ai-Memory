# MnemoCore platform status — updated 2026-07-13

This is the public status source for the PR #35 production-readiness work. The
repository now provides a functioning single-node prototype. It does not yet
provide enough evidence for a production-ready release claim.

## Current product boundary

- AgentMemory is the stable, local-first track: exact scopes, SQLite-backed
  persistence, bitemporal history, receipts, timeline queries, deterministic
  context compilation, and rebuildable projections.
- The REST/Docker surface is a working prototype and degrades to local operation
  when Redis or Qdrant is absent.
- The broader legacy cognitive stack remains available for compatibility and
  experimentation but is outside the currently proven release baseline.
- The public repository contains only public contracts and generic adapter
  boundaries. Private runtime policy and implementation details are excluded.

## Evidence

| Area | Status | Current evidence |
|---|---|---|
| AgentMemory core | PASS | `python -m pytest tests/agent_memory -q` — 308 passed, including 4 erasure tests |
| Offline adapters | PASS | `python -m pytest tests/integrations -q` — 6 passed |
| Benchmark contracts/smoke | PASS | Focused benchmark batch — 18 passed |
| Combined prototype verification | PASS WITH SKIPS | 2026-07-13 AgentMemory, webhook security, adapter, benchmark, packaging, runtime, and deployment batch — 380 passed, 2 packaging metadata tests skipped because the local build-backend prerequisite was unavailable |
| Webhook persistence security | PASS (prototype) | 28 tests prove `secret_ref`-only persistent signing, rejection of inline/legacy plaintext and persistent custom headers, fail-closed load, serialized mutation, atomic replace, and redacted delivery/log failures |
| Physical erasure | PASS (prototype) | 4 tests prove exact-scope ownership, supersession cascade protection, dependent-row cleanup, content-free receipt, reopen, FTS removal, and SQLite foreign-key/integrity checks |
| Docker build | PASS | `docker build -t mnemocore:prototype-docs .` completed using the Python 3.11 runtime image and installed wheel |
| Docker runtime | PASS (prototype) | Real container returned HTTP 200 from `/health`, `/ready`, and `/metrics/`; readiness was `ready`, while health correctly reported `degraded` without Redis |
| Metrics | PASS | Prometheus exposition is served at canonical `/metrics/` on API port 8100 |
| Compose contracts | PASS (static) | Required API/Redis/Qdrant secrets and internal service exposure are contract-tested; full-stack runtime is still pending |
| Helm contracts | PASS (static) | Single-node defaults, probes, shared port 8100, and ServiceMonitor path are contract-tested; cluster runtime is pending |
| Legacy non-integration lane | TIMEOUT | Historical command did not produce a reliable completion summary |
| Quarantined lifecycle file | QUARANTINED | 16 cases remain excluded pending migration or retirement |
| Required service lane | INACTIVE | No non-zero Redis/Qdrant service-marked test collection is proven |
| Security/dependency scans | FAIL | Historical Bandit and dependency-audit findings remain unresolved and must be rerun before release |
| Full production gate | NOT MET | Full security closure, erasure durability, legacy/service, recovery, and target-environment evidence remain incomplete |

## Prototype improvements now landed

- One package version authority and compatibility-preserving 2.x dependencies.
- Correct console entrypoints and wheel inclusion boundaries.
- Docker builds and installs the wheel, uses `mnemocore.api.main:app`, permits
  command overrides, runs unprivileged, and has a runtime healthcheck.
- `/health` is liveness/diagnostics; `/ready` represents initialized local
  runtime readiness; unavailable Redis does not make local readiness fail.
- Metrics, Compose, Helm, ServiceMonitor, and CI use `/metrics/` on port 8100.
- Compose requires explicit secrets and does not publish Redis/Qdrant ports.
- Helm defaults describe one prototype replica rather than implied HA behavior.
- CI runs image import/runtime smoke and includes Docker in its fail-closed
  summary.
- Persistent webhook configuration stores an opaque signing-secret reference,
  rejects persistent custom headers, resolves secrets at send time, and uses
  serialized atomic persistence with fail-closed loading.
- AgentMemory physical erase uses exact-scope checks, explicit supersession
  cascade, dependent-row cleanup, integrity verification, and a content-free
  receipt.

## Remaining gaps

1. Resolve or accept security and dependency findings with fresh evidence.
2. Harden the physical erase prototype beyond its cooperative sidecar-lock
   contract: uncoordinated raw SQLite clients are currently unsafe during file
   replacement, and arbitrary power-loss/failure points, backups, and external
   derived artifacts are not yet covered.
3. Migrate or retire the quarantined lifecycle test and make the legacy lane
   complete reliably.
4. Add genuine Redis/Qdrant service tests and run Compose persistence/restart,
   backup, and restore scenarios.
5. Validate Helm dependencies, secrets, storage, rollback, and recovery on a
   real target cluster; current chart evidence is static.
6. Run clean wheel/sdist builds across supported Python versions without skips.
7. Publish reproducible performance artifacts and complete load/soak testing.
8. Perform the final public security and release review before tagging.

## Honest release statement

The current branch is suitable for local evaluation and single-node prototype
deployment. AgentMemory has the strongest evidence. Docker runtime behavior is
now directly demonstrated. Multi-node durability, full legacy compatibility,
exhaustive erasure durability, production recovery, and full security closure
remain unproven.
