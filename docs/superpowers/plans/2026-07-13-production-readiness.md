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

## Tasks

- [ ] Packaging/version: inventory release evidence, centralize version resolution, repair CLI entrypoints, split optional extras, and add clean-wheel tests.
- [ ] Docker/API: install the wheel in the image, correct the Uvicorn target, add readiness semantics, align metrics exposure, and test persistence reopen.
- [ ] CI: include Docker and cancellation/skipped states in fail-closed summary checks; add wheel and runtime smoke lanes.
- [ ] Compose/Helm: lock dependencies, remove accepted placeholder secrets, align ports/probes/ServiceMonitor, and add render/lint smoke tests.
- [ ] Security/lifecycle: replace plaintext secret persistence and specify physical-erasure behavior with WAL/SHM, crash, rollback, and rebuild tests.
- [ ] Documentation/release: update runbooks, stability/product boundary, baseline, deployment instructions, and release evidence.

