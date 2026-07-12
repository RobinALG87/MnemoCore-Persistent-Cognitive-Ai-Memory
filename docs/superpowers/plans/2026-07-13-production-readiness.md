# MnemoCore Production Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AgentMemory core and REST adapter production-ready with reproducible packaging, container startup, health/metrics, CI gates, deployment manifests, and public operational documentation.

**Architecture:** AgentMemory plus REST is the stable release track; legacy HAIM remains experimental. Work is split into independently revertible drones, each with focused tests and Sol review before dependent work starts. No private runtime implementation or proprietary integration detail enters this repository.

**Tech Stack:** Python 3.10–3.12, pyproject packaging, FastAPI/Uvicorn, SQLite AgentMemory, Docker, Docker Compose, Helm, GitHub Actions, pytest, Ruff, mypy, Bandit, pip-audit.

## Global Constraints

- Exact `MemoryScope` isolation remains mandatory.
- Existing AgentMemory, REST, MCP, CLI, and HAIM compatibility surfaces remain backward compatible.
- Minimal installation must not require Redis, Qdrant, FAISS, FastAPI, MCP, or model providers.
- Private runtime policies, formulas, thresholds, paths, prompts, and datasets remain outside the public repository.
- No new mandatory database or service is introduced.

## Current status

The single-node prototype is functioning and reproducible. This is not a
production-readiness claim: full security closure, exhaustive erasure
durability, legacy/service coverage, recovery exercises, supported-Python
artifact validation, and target-cluster evidence remain open.

## Tasks

- [x] Packaging/version prototype: central version resolution, repaired console
  entrypoints, compatibility-preserving optional extras, wheel boundaries, and
  packaging contracts are implemented. Clean wheel/sdist validation across the
  full supported Python matrix remains a release gate.
- [x] Docker/API prototype: the image builds and installs the wheel, uses
  `mnemocore.api.main:app`, supports command overrides, runs unprivileged, and
  exposes `/health`, `/ready`, and canonical `/metrics/` on port 8100.
- [x] CI prototype gates: packaging and Docker runtime lanes are included in the
  fail-closed summary, and the image import plus public runtime endpoints are
  exercised.
- [x] Compose/Helm prototype: placeholder secrets are rejected, ports and probes
  are aligned, Helm defaults are explicitly single-node, optional embedded
  services are disabled by default, and static render/contract checks exist.
- [x] Persistent webhook hardening: persisted signing material is `secret_ref`
  only; inline/legacy plaintext and persistent custom headers fail closed;
  mutations are serialized and atomically replaced.
- [x] Physical-erasure prototype: exact scope, explicit supersession cascade,
  dependent-row cleanup, integrity checks, and a content-free receipt are
  implemented under a cooperative sidecar-lock contract.
- [x] Public prototype documentation: README, deployment guide, platform status,
  and release checklist reflect the implemented runtime and remaining gaps.

## Runtime evidence

On 2026-07-13, after commit `7e82a93`, a clean Docker Compose recreate/build
completed successfully. Redis and Qdrant became healthy, the API became healthy,
and direct checks returned HTTP 200 from `/health`, `/ready`, and `/metrics/`;
the metrics response contained Prometheus `HELP` exposition. This proves the
local Compose prototype path, not production durability or scale.

## Remaining production gates

- [ ] Resolve or explicitly accept fresh Bandit and dependency-audit findings,
  then complete the final public security review.
- [ ] Extend erasure verification beyond cooperating `SQLiteMemoryStore`
  clients to power-loss/failure injection, backup deletion, external derived
  artifacts, and documented recovery behavior.
- [ ] Migrate or retire the quarantined lifecycle tests, make the legacy lane
  terminate reliably, and add non-zero Redis/Qdrant service-lane coverage.
- [ ] Build and install wheel/sdist artifacts on every supported Python version
  without unexpected skips.
- [ ] Exercise Compose persistence/restart, backup/restore, upgrade/rollback,
  load, and soak scenarios.
- [ ] Validate Helm secrets, storage classes, rollback, disruption, recovery,
  and observability on the actual target cluster before any multi-node claim.
- [ ] Publish versioned benchmark evidence and complete release/tag review.
