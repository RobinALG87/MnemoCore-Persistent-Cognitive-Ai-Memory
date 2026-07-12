# MnemoCore platform baseline — 2026-07-12

This is the current public source of truth for the Workstream 0 baseline. It
records what was actually executed and separates focused green lanes from
legacy, quarantined, unavailable, or unresolved areas. It does not claim that
the full platform is green.

## Scope and repository state

- Baseline reference: `main` at `77999311129fcd3cfdc0f94d4e401ee311b5bea2`.
- Implementation branch: `agent/pr35-workstream0`.
- Current branch HEAD after importing the planning document:
  `5607f9e96a591d13987d4b532eb1990ce6355def`.
- Working tree was clean before this status document was created.
- Production code was not changed for Workstream 0.
- Python: 3.12.10; pytest: 9.1.0; Ruff: 0.15.17; mypy: 2.1.0;
  Bandit: 1.9.4; pip-audit: 2.10.1; Docker CLI: 29.5.2.
- Dependency freeze fingerprint (`pip freeze --all`, SHA-256):
  `9a344bba7deb7ac45cac55e05ef3dd5ae2ee953f9c5362fb088653a54e347bc3`.

## Reproducible lane results

| Area | Status | Evidence |
|---|---|---|
| AgentMemory core | PASS | `python -m pytest tests/agent_memory -q` — 304 passed in 22.59s |
| Offline adapters | PASS | `python -m pytest tests/integrations -q` — 6 passed in 1.24s |
| Benchmark contracts/smoke | PASS | `python -m pytest benchmarks/test_agent_memory_baseline.py benchmarks/test_benchmark_smoke.py -q` — 18 passed in 11.18s |
| Legacy non-integration lane | TIMEOUT | The documented command exceeded 60 seconds without a reliable summary; no green claim is made |
| Quarantined lifecycle file | QUARANTINED | `python -m pytest tests/test_integration_store_query_cycle.py -q -rs` — 16 skipped because `--run-integration` was not enabled |
| Required service lane | NOT RUN / INACTIVE | No collected tests currently carry the required service markers |
| Full platform suite | NOT RUN | The legacy lane did not complete, so a full green result cannot be asserted |
| Focused Ruff | FAIL | `ruff check src/mnemocore/agent_memory tests/agent_memory` reports one unused `pytest` import in `tests/agent_memory/test_fingerprint.py` |
| Whole-repository Ruff | FAIL | `ruff check src tests benchmarks` reports 880 existing findings |
| Focused mypy | PASS (scoped) | `MYPYPATH=src python -m mypy --explicit-package-bases --follow-imports=skip src/mnemocore/agent_memory` succeeds; strict whole-repo typing is not proven |
| Bandit | FAIL | `bandit -r src/mnemocore -q` reports 48 findings (38 low, 9 medium, 1 high) and skips one legacy file with a syntax error |
| Dependency audit | FAIL | `pip-audit -l` reports 13 known vulnerabilities in 7 installed packages; the environment also contains local packages that cannot be resolved from PyPI |
| Secret-pattern scan | PASS with placeholders | The scan found only redacted/documentation examples; no credential material was detected |
| Docker build | NOT RUN | `docker build -t mnemocore-baseline:local .` could not connect because the Docker daemon was unavailable |

The exact commands above are intentionally kept in the document so a future
run can compare results without relying on a historical test count.

## Open risks and inconsistencies

These are release-blocking or scope-defining gaps observed on the baseline:

1. Version authority is contradictory: current packaging and API fallbacks use
   2.0.0 while older release/remediation history advertises 5.x.
2. README onboarding still describes the repository as private and uses an old
   repository URL.
3. The stable AgentMemory core is documented alongside a much larger legacy
   stack, while the default dependency set still pulls optional services.
4. The Docker entrypoint targets `src.api.main:app`; the maintained package
   target is `mnemocore.api.main:app`. A container smoke test is pending.
5. CI's summary failure loop does not include the Docker job even though Docker
   is listed in `needs`.
6. `tests/test_integration_store_query_cycle.py` still requires migration to
   the current AgentMemory lifecycle before it can leave quarantine.
7. No required Redis/Qdrant service lane is active because its marker set has
   zero collected tests.
8. Webhook secrets are still persisted by the legacy path without a physical
   erasure workflow for stored data and derived artifacts.
9. Default AgentMemory retrieval is lexical; hybrid providers, deterministic
   fusion, and calibrated scoring are future work.
10. Benchmark tests exist, but canonical raw baseline artifacts tied to a
    release are not yet published.

## Stale pull-request triage

The GitHub state was checked with `gh pr view`:

- PR #31 — draft, open, `DIRTY`, last updated 2026-02-20. Do not merge the
  stale branch directly; port only a current, test-backed fix if still needed.
- PR #34 — draft, open, `CLEAN`, last updated 2026-03-05. Treat as stale until
  its claims are revalidated against the baseline above.

## Workstream 0 exit gate

The repository now has one evidence-backed current-state document and a
reproducible command list. The exit gate is met for documentation triage, but
implementation work must remain stopped until this baseline is reviewed and
the unresolved release gates are explicitly accepted or assigned.

## Public safety boundary

This repository documents only public MnemoCore contracts and generic adapter
boundaries. Private runtime policies, formulas, thresholds, paths, prompts,
and implementation details are intentionally excluded.
