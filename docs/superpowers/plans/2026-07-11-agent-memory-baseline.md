# AgentMemory Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, local-only AgentMemory benchmark for latency, retrieval semantics, storage size, and process RSS.

**Architecture:** A parent CLI creates a fixed corpus manifest and launches one fresh Python worker per repetition. Each worker owns a temporary SQLite database, exercises only `AgentMemory`, checkpoints WAL, and returns raw measurements; the parent aggregates without discarding raw samples.

**Tech Stack:** Python standard library, psutil, SQLite, pytest, MnemoCore AgentMemory.

## Global Constraints

- Default baseline repetitions are at least five; smoke mode may use one.
- No HAIMEngine, telemetry, private text, private paths, or private formulas.
- Corpus, scopes, seed, dependency state, repository state, SQLite settings, and raw samples must be recorded.

---

### Task 1: Public result contract and deterministic corpus

**Files:**
- Create: `benchmarks/agent_memory_baseline.py`
- Test: `benchmarks/test_agent_memory_baseline.py`

- [x] Write failing tests for stable corpus SHA-256, generic corpus text, default repetitions, and JSON schema.
- [x] Run `python -m pytest benchmarks/test_agent_memory_baseline.py -q` and confirm missing-module failure.
- [x] Implement immutable configuration, corpus generation, manifest creation, aggregation, and JSON serialization.
- [x] Re-run focused tests.

### Task 2: Isolated benchmark worker

**Files:**
- Modify: `benchmarks/agent_memory_baseline.py`
- Test: `benchmarks/test_agent_memory_baseline.py`

- [x] Add a failing smoke test invoking one fresh subprocess.
- [x] Confirm failure because the worker path is absent.
- [x] Implement remember, lexical hit/miss/selectivity, bitemporal recall, context compilation, checkpointed main/WAL/SHM sizes, and RSS baseline/peak/delta.
- [x] Re-run smoke and schema tests.

### Task 3: Documentation and delivery

**Files:**
- Modify: `benchmarks/README.md`
- Modify: `.gitignore`

- [x] Document baseline and smoke commands plus interpretation boundaries.
- [x] Run focused tests, inspect the diff, and verify no private strings or HAIMEngine imports.
