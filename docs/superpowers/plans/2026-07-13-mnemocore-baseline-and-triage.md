# MnemoCore baseline and stale-work triage

This child plan executes Workstream 0 from the next-phase master plan. It is
documentation/status work only; it must not modify production code.

## Files

- Create `docs/status/2026-07-12-platform-baseline.md`.
- Modify `PRODUCTION_REMEDIATION_PROGRESS.md` to point to the current source of
  truth and mark its older audit as historical.
- Modify `RELEASE_CHECKLIST.md` to link evidence and use explicit status words.
- Review `.github/workflows/ci.yml`, PR #31, and PR #34.

## Completed checks

- [x] Record baseline and implementation SHAs, clean-state evidence, tool
  versions, and dependency fingerprint.
- [x] Run AgentMemory, offline adapter, and benchmark contract/smoke lanes.
- [x] Run the documented legacy lane and record its timeout without claiming a
  green result.
- [x] Run the quarantined lifecycle file separately and record its skips.
- [x] Review CI summary coverage, Docker entrypoint, and public onboarding.
- [x] Triage PR #31 and PR #34 without merging either stale branch.
- [x] Record focused/whole-repository Ruff, scoped mypy, Bandit, dependency
  audit, and secret-scan results.

## Pending evidence before release claims

- [ ] Re-run the legacy lane to completion and triage its failure clusters.
- [ ] Migrate and unquarantine the lifecycle integration file.
- [ ] Add nonzero, service-specific Redis/Qdrant tests before activating the
  service lane.
- [ ] Start Docker and complete image, import, health, and persistence smoke
  tests.
- [ ] Resolve the existing security and dependency-audit findings through
  separate, focused implementation PRs.

## Exit criteria

The baseline document is authoritative, all later workstreams link to it, and
no later workstream is started until the owner reviews the unresolved gates.
