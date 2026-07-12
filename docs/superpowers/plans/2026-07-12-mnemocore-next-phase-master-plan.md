# MnemoCore Next-Phase Platform Improvement Master Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:writing-plans` to create one child implementation plan per workstream, then use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement it task-by-task. Track every step with checkboxes and stop at each release gate.

**Goal:** Turn MnemoCore into a trustworthy, reproducible, local-first, open-source memory and context infrastructure for long-lived AI agents, while correcting current release, packaging, security, testing, retrieval, documentation, and legacy-boundary weaknesses.

**Architecture:** The stable product is the focused `mnemocore.agent_memory` layer: an exact-scope, append-only, bitemporal memory ledger with explainable retrieval and bounded context compilation. Optional retrieval, API, distributed-storage, and legacy HAIM capabilities plug into that core without becoming mandatory. MnemoCore remains a memory/context system; task planning, model routing, tool execution, verification loops, autonomous workflow orchestration, and compute management belong to external runtimes and must not be implemented here.

**Tech Stack:** Python 3.10–3.12, SQLite WAL/FTS5, Hatch/Hatchling, optional Redis/Qdrant/FAISS, FastAPI, MCP, Ruff, mypy, pytest, Hypothesis, GitHub Actions, Docker/OCI, OpenTelemetry/OpenInference-compatible traces.

## Global Constraints

1. **Open-source boundary:** MnemoCore may store, version, retrieve, explain, compact, and expose memory/context. It must not become an agent runtime, planner, model router, tool runner, verifier, task-graph engine, cloud-escalation layer, or autonomous workflow controller.
2. **No private-product leakage:** Do not add private orchestration policies, prompts, task IRs, routing thresholds, skill-execution logic, compute-management algorithms, or proprietary evaluation corpora to this repository. External products integrate only through stable MnemoCore APIs.
3. **Local-first default:** `mnemocore.agent_memory` must work with Python's standard library and SQLite. Network services, vector databases, LLM providers, background daemons, and distributed infrastructure remain optional extras.
4. **Authorization before ranking:** Exact scope predicates are mandatory storage and retrieval boundaries. Embeddings, HDV masks, similarity scores, graph edges, and rerankers are never authorization mechanisms.
5. **Evidence before confidence:** Derived facts, summaries, procedures, and context items must carry source lineage. Confidence without evidence must not be promoted as truth.
6. **Temporal correctness:** Validity time and knowledge time remain separate. Recency is a ranking signal, never a proxy for truth.
7. **Fail visibly:** Persistence, migration, retrieval-provider, authorization, index, and extraction failures return typed errors. Do not silently fall back in ways that alter correctness.
8. **No unsafe deserialization:** Runtime code must not deserialize untrusted pickle data. Legacy migration, if retained, must be explicit, offline, isolated, and opt-in.
9. **No content telemetry by default:** Metrics and traces may record IDs, counts, timings, provider names, and policy versions; private memory content is excluded unless a caller explicitly enables a documented development-only mode.
10. **Backward compatibility:** Existing public imports remain operational until a documented deprecation window expires. New stable code must not import the legacy HAIM engine.
11. **Schema discipline:** Every schema change is versioned, transactionally migrated, covered by rollback/rebuild tests, and tested from the oldest supported schema.
12. **Benchmark honesty:** Do not publish leadership or superiority claims until the exact dataset, model, prompt, token budget, dependency versions, commit SHA, and raw outputs are reproducible.
13. **One responsibility per PR:** Avoid sweeping mechanical and behavioral changes in the same pull request. Every PR must be independently reviewable and revertible.
14. **TDD and frequent commits:** Write a failing test, verify the failure, implement the smallest change, verify the lane, then commit.
15. **No direct-to-main implementation:** Use one branch per task or tightly coupled task set. Require green checks and review before merge.

---

## 1. Executive Decision

MnemoCore should now operate as two clearly separated tracks:

### Stable product track: AgentMemory

The stable product promise is:

> Durable, exact-scoped, temporally correct, evidence-bearing memory and bounded context for long-lived agents.

The stable surface is centered on:

- `MemoryScope`
- `MemoryRecord`
- `MemoryEvent`
- `AgentMemory` / `SyncAgentMemory`
- sessions and episodes
- bitemporal `remember`, `supersede`, `recall`, `explain`, `history`, `forget`, and `rebuild`
- explainable candidate retrieval
- bounded context compilation
- feedback and procedure **storage/proposal**, not execution
- optional adapters that map directly to the public memory contract

### Experimental compatibility track: legacy HAIM

The legacy cognitive engine remains available for research and backward compatibility, but it is not the stability anchor for new releases. It must be isolated behind optional dependencies and explicit imports. No new stable AgentMemory functionality may depend on the legacy engine.

### Explicitly outside MnemoCore

The following are not open-source MnemoCore responsibilities:

| Outside responsibility | Boundary reason |
|---|---|
| User-task decomposition and Task IR | Agent runtime concern |
| Global workspace / active task state | Orchestration concern |
| Model selection, routing, confidence escalation | Inference/runtime concern |
| Local-vs-cloud escalation | Product policy concern |
| Tool selection and execution | Agent motor-system concern |
| Code patching, shell execution, deployment | Action/runtime concern |
| Deterministic verifier loops | Workflow/runtime concern |
| Autonomous skill execution | Agent behavior concern |
| Compute budgeting and neural-path selection | Inference/runtime concern |
| Proprietary prompts, policies, and evaluation sets | Private product IP |

MnemoCore may expose neutral primitives consumed by these systems, but must not implement their decision logic.

---

## 2. Baseline Snapshot and Known Gaps

**Planning baseline:** `main` around commit `77999311129fcd3cfdc0f94d4e401ee311b5bea2`, reviewed on 2026-07-12.

The next implementation worker must verify the exact current SHA before acting.

### Strong foundation already present

- Focused `mnemocore.agent_memory` package.
- Exact tenant/user/agent/project/session scope model.
- SQLite WAL, foreign keys, immutable event ledger, materialized projections, FTS5, and rebuild support.
- Bitemporal validity/knowledge-time queries and supersession.
- Memory receipts and deterministic bounded context compilation.
- Async and explicit sync clients.
- Session lifecycle and episode recording.
- Reproducible local-only AgentMemory benchmark harness with subprocess isolation, fixed seed/corpus, resource samples, and environment manifest.
- Separated CI lanes for AgentMemory, legacy, adapters, benchmark smoke, security, property tests, and Docker.

### Immediate inconsistencies and risks to resolve

1. **Version history is contradictory.** Earlier repository history advertises v5.0.0, while the current package, README, API fallback, Docker defaults, and release checklist use 2.0.0. Version authority is not trustworthy.
2. **README onboarding is incorrect.** It calls the public repository private and points installation/clone commands at a different repository name.
3. **Product positioning is split.** README leads with the legacy HAIM/VSA "brain" narrative while the strongest current product is the focused AgentMemory foundation.
4. **Base installation is too heavy.** The standard-library AgentMemory package pulls FastAPI, Redis, Qdrant, FAISS, MCP, background-worker, and other dependencies through the default project dependency list.
5. **Docker packaging appears inconsistent.** The image copies `src/` without installing the package and starts `uvicorn src.api.main:app`; the intended import is `mnemocore.api.main:app`.
6. **CI truth is incomplete.** The summary job depends on Docker but does not include Docker in its explicit failure loop. Mypy remains non-blocking, and formatting/lint tooling is duplicated across Ruff, Black, isort, and flake8.
7. **Full-platform green status is unclear.** The current release checklist says work is in progress and references remaining failures, while the new focused lanes are greener. Release scope is ambiguous.
8. **Integration proof is incomplete.** A known-red local integration file is quarantined, and there is no active required Redis/Qdrant service lane with nonzero test collection.
9. **Retrieval is foundation-only.** Current AgentMemory recall is scoped FTS5 lexical retrieval; hybrid candidate union, semantic providers, fusion, and calibrated ranking remain future work.
10. **Webhook secrets are persisted in plaintext.** File permissions reduce exposure but do not solve secret-at-rest risk.
11. **Privacy erasure is not implemented.** `forget()` is correctly logical/auditable, but users need an explicit, separate physical-erasure workflow.
12. **Stale open PRs exist.** PR #31 and PR #34 are old, draft, non-mergeable, and based on pre-current-main snapshots. They must not be merged directly.
13. **Benchmark artifacts are not published.** The harness is strong, but the repository does not yet provide canonical raw baseline outputs tied to releases.
14. **Legacy and stable contracts are intermingled in documentation, dependencies, and release claims.** This increases maintenance cost and weakens user trust.

---

## 3. Program Outcomes and Quantitative Gates

The program is complete only when all mandatory gates pass.

### Correctness gates

- Cross-scope leakage tests: **0 leaked records across all tenant/user/agent/project/session combinations**.
- Bitemporal boundary tests: **100% expected results before, at, and after validity and knowledge-time boundaries**.
- Event-ledger rebuild parity: **100% equality between committed projections and projections rebuilt from immutable events**.
- Idempotency: **same scope + same idempotency key never creates a duplicate; different scopes remain independent**.
- Context receipts: **100% of emitted context items identify source memory, scope, score components, reason, and estimated tokens**.
- Migration: **every supported schema fixture migrates transactionally and survives reopen/rebuild**.

### Packaging and operability gates

- `pip install mnemocore` installs the local AgentMemory core without Redis, Qdrant, FAISS, FastAPI, MCP, OpenAI, Anthropic, or background-worker dependencies.
- Importing `mnemocore.agent_memory` performs no network calls, service discovery, telemetry, or legacy-engine initialization.
- `pip install "mnemocore[server]"` supports the maintained REST deployment.
- Docker image builds, imports `mnemocore`, starts, passes `/health`, and persists/reopens an AgentMemory database.
- Python 3.10, 3.11, and 3.12 lanes pass for the supported core.

### Quality gates

- AgentMemory package line coverage: **>=90%**.
- New or materially changed modules: **>=90% branch coverage**.
- `mnemocore.agent_memory` passes strict mypy.
- Ruff lint and format are blocking.
- No unresolved critical/high findings from dependency audit, Bandit, container scan, or the release threat-model checklist.

### Retrieval and context gates

- Default local, non-LLM recall p95: **<200 ms** on the pinned reference dataset and hardware profile recorded in the benchmark manifest.
- Default compiled context: **<=2,000 estimated tokens**.
- Hybrid retrieval never reduces exact lexical-hit correctness relative to the lexical baseline.
- Every score is finite and every normalized probability remains within `[0.0, 1.0]`.
- Retrieval reports candidate counts, filters, scoring-profile version, selected items, and latency without logging content by default.

### Benchmark and release gates

- Raw JSON outputs, configuration, manifest, and report are attached to each proof release.
- External comparisons use identical model, prompt, token budget, and dataset split.
- README claims are generated from or directly traceable to committed benchmark artifacts.
- All documentation quickstarts are executed in CI.

---

## 4. Target Repository Shape

Do not perform a repository-wide move immediately. Evolve toward this structure incrementally while preserving compatibility:

```text
src/mnemocore/
├── __init__.py                     # Lazy, minimal public exports only
├── agent_memory/                   # Stable product core
│   ├── __init__.py
│   ├── client.py                   # Async/sync public facades
│   ├── models.py                   # Stable typed public models
│   ├── errors.py                   # Typed public exceptions
│   ├── store.py                    # Store protocol
│   ├── sqlite_store.py             # SQLite implementation
│   ├── schema.py                   # Versioned schema/migrations
│   ├── timeline.py                 # Bitemporal semantics
│   ├── sqlite_codecs.py            # Private persistence codecs
│   ├── context.py                  # Bounded context compiler
│   ├── feedback.py                 # Outcome/reliability events
│   ├── compaction.py               # Evidence-preserving compaction plans
│   ├── erasure.py                  # Explicit physical-erasure workflow
│   └── retrieval/
│       ├── __init__.py
│       ├── base.py                 # Candidate-provider interfaces
│       ├── policy.py               # Versioned retrieval policy
│       ├── lexical.py              # FTS5 provider
│       ├── fusion.py               # Deterministic rank fusion
│       ├── embedding.py            # Optional semantic provider adapter
│       ├── hdv.py                  # Optional HDV provider adapter
│       └── explain.py              # Score/reason construction
├── api/                            # Optional server adapter
├── cli/                            # Optional CLI adapter
├── mcp/                            # Optional MCP adapter
├── integrations/                   # Thin framework adapters only
└── core/, cognitive/, subconscious/, ...
    # Legacy/experimental HAIM compatibility surface; no dependency from agent_memory

tests/
├── agent_memory/
│   ├── contract/
│   ├── retrieval/
│   ├── migrations/
│   ├── privacy/
│   └── concurrency/
├── api/
├── integrations/
└── legacy/

benchmarks/
├── agent_memory/
│   ├── harness.py
│   ├── manifests.py
│   ├── local_performance.py
│   ├── temporal_suite.py
│   ├── scope_isolation_suite.py
│   ├── provenance_suite.py
│   ├── external/
│   │   ├── locomo.py
│   │   ├── longmemeval.py
│   │   └── beam.py
│   └── schemas/
└── legacy/

docs/
├── PRODUCT_BOUNDARY.md
├── STABILITY.md
├── SECURITY.md
├── RELEASE.md
├── MIGRATION_AGENT_MEMORY.md
├── LEGACY_HAIM_POLICY.md
├── adr/
└── superpowers/plans/
```

---

## 5. Program Dependency Graph

```text
W0 Baseline/triage
        |
        +--> W1 Version/release integrity
        +--> W2 Product boundary/governance
        |
        +--> W3 Packaging/dependency isolation
        +--> W4 CI/test truth
        +--> W5 Security/privacy
                    |
                    +--> W6 AgentMemory contract hardening
                                |
                                +--> W7 Hybrid retrieval
                                +--> W8 Context/feedback/compaction
                                            |
                                            +--> W9 API/MCP/CLI parity
                                            +--> W10 Benchmarks/proof
                                                        |
                                                        +--> W11 Legacy containment
                                                        +--> W12 Release/documentation
```

Workstreams W3, W4, and W5 may proceed in parallel only after W0 establishes a truthful baseline and W1 fixes version authority decisions. W7 and W8 may proceed in parallel after W6 freezes shared public types.

---

# Workstream 0 — Establish a Truthful Baseline and Triage Stale Work

**Priority:** P0, serial, first work only.

**Child plan:** `docs/superpowers/plans/2026-07-13-mnemocore-baseline-and-triage.md`

**Files:**

- Create: `docs/status/2026-07-12-platform-baseline.md`
- Modify: `PRODUCTION_REMEDIATION_PROGRESS.md`
- Modify: `RELEASE_CHECKLIST.md`
- Review: `.github/workflows/ci.yml`
- Review: open PR #31 and PR #34

**Required deliverable:** One evidence-backed status document that distinguishes passing focused lanes, failing/quarantined legacy lanes, active security risks, package/install failures, Docker status, and stale documentation. Do not claim the full platform is green unless a fresh full run proves it.

- [ ] Record `git rev-parse HEAD`, `git status --porcelain`, Python/platform versions, dependency hashes, and test commands.
- [ ] Run the focused AgentMemory lane on Python 3.11.
- [ ] Run benchmark contract/smoke tests without timing-sensitive suites.
- [ ] Run offline adapter tests.
- [ ] Run the legacy non-integration lane exactly as documented in `docs/TEST_LANES.md`.
- [ ] Run the quarantined integration file separately and record each current failure cluster.
- [ ] Build the Docker image and run an import plus `/health` smoke test with an explicit test key.
- [ ] Run Ruff/format, mypy, Bandit, pip-audit, and secret scanning with exact versions recorded.
- [ ] Inspect PR #31 and PR #34 against current `main`. Port only still-relevant, minimal, test-backed changes to fresh branches; otherwise close them as superseded. Never merge either stale branch directly.
- [ ] Replace stale counts and outdated assertions in `PRODUCTION_REMEDIATION_PROGRESS.md` with links to the new baseline document.
- [ ] Change `RELEASE_CHECKLIST.md` from aspirational checkboxes to evidence links and explicit `PASS`, `FAIL`, `QUARANTINED`, or `NOT RUN` statuses.
- [ ] Commit only documentation/status changes in the first PR.

**Exit gate:** The repository has one authoritative current-state document, and every later workstream can cite a reproducible failing or passing command.

---

# Workstream 1 — Restore Version, Release, and Repository Integrity

**Priority:** P0.

**Child plan:** `docs/superpowers/plans/2026-07-13-version-and-release-integrity.md`

**Files:**

- Create: `docs/adr/0001-version-authority-and-release-history.md`
- Modify: `pyproject.toml`
- Modify: `src/mnemocore/__init__.py`
- Modify: `src/mnemocore/api/version.py`
- Modify: `Dockerfile`
- Modify: `helm/mnemocore/Chart.yaml`
- Modify: `helm/mnemocore/values.yaml`
- Modify: `config.yaml`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `RELEASE_CHECKLIST.md`
- Modify: `.github/workflows/docker-publish.yml`
- Test: `tests/test_version_contract.py`

**Decision:** Git tags become the release-version authority through Hatch VCS. Runtime code obtains the installed version through `importlib.metadata`; source checkouts use the generated VCS fallback. Hardcoded independent version strings are removed.

- [ ] Inventory Git tags, GitHub releases, PyPI versions, GHCR tags, and documentation history.
- [ ] Select the next **monotonically increasing** SemVer. Default recommendation: `6.0.0a1` if any `5.x` artifact was publicly distributed; otherwise choose the next monotonic version proven by the inventory. Record the decision and evidence in ADR-0001.
- [ ] Configure Hatch VCS dynamic versioning and generated fallback module.
- [ ] Make `mnemocore.__version__` and API version read installed metadata only, with one generated fallback.
- [ ] Remove duplicate/contradictory changelog release headings and preserve historical entries without rewriting history.
- [ ] Replace hardcoded README version badges with a latest-release badge.
- [ ] Correct every clone/install/homepage URL to `RobinALG87/MnemoCore-Persistent-Cognitive-Ai-Memory`.
- [ ] Remove language describing the repository as private.
- [ ] Replace `mnemocore = "mnemocore.api.main:app"` with valid CLI entry points. The API entry point must call a real zero-argument CLI function that starts Uvicorn; it must not expose the ASGI object as a console script.
- [ ] Pass the resolved version into Docker labels and OCI metadata from the release workflow.
- [ ] Add a contract test asserting package, API, CLI, Docker label input, and generated documentation version resolve to the same value.
- [ ] Add a release dry-run job that builds wheel/sdist, installs them into a clean environment, imports the package, and checks metadata.

**Exit gate:** One authoritative version exists, repository URLs are correct, and a clean wheel/sdist install reports the same release version everywhere.

---

# Workstream 2 — Publish and Enforce the Open-Source Product Boundary

**Priority:** P0 strategic guardrail.

**Child plan:** `docs/superpowers/plans/2026-07-13-product-boundary-and-governance.md`

**Files:**

- Create: `docs/PRODUCT_BOUNDARY.md`
- Create: `docs/STABILITY.md`
- Create: `.github/CODEOWNERS`
- Create: `.github/pull_request_template.md`
- Create: `.github/ISSUE_TEMPLATE/feature.yml`
- Create: `.github/ISSUE_TEMPLATE/bug.yml`
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `ROADMAP.md`

**Public positioning:**

> MnemoCore is open-source infrastructure for durable, scoped, temporally correct, explainable agent memory and context.

- [ ] Document the allowed scope: memory ingestion, temporal versioning, retrieval, provenance, context compilation, feedback, compaction, storage, and thin adapters.
- [ ] Document explicit non-goals: planning, task orchestration, model routing, tool execution, verifier loops, cloud escalation, autonomous skill execution, and compute/inference management.
- [ ] Define stability tiers:
  - `Stable`: `mnemocore.agent_memory` public contract.
  - `Beta`: hybrid retrieval, compaction, and selected adapters until benchmarked.
  - `Experimental`: legacy HAIM/cognitive/subconscious modules.
- [ ] Add a PR checklist item: “Does this change implement memory/context infrastructure rather than agent-runtime behavior?”
- [ ] Require owner review for public models, schemas, migrations, release workflows, licensing, and product-boundary documents.
- [ ] Add feature-request fields requiring the proposer to identify storage/retrieval/context semantics and to explain why the feature is not agent orchestration.
- [ ] Rewrite README's first screen around AgentMemory, exact scope, truth timeline, memory receipts, local SQLite, and optional extensions.
- [ ] Move the HAIM/HDC research narrative to a clearly labeled experimental/research section.
- [ ] Keep the existing open-source license decision explicit. Do not add private runtime code under a different license in this repository.

**Exit gate:** A contributor or coding agent can decide from repository documentation whether a proposed feature belongs in MnemoCore without learning any private product architecture.

---

# Workstream 3 — Make AgentMemory the Minimal Install and Isolate Optional Capabilities

**Priority:** P0/P1.

**Child plan:** `docs/superpowers/plans/2026-07-14-minimal-packaging-and-extras.md`

**Files:**

- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `requirements-dev.txt`
- Modify: `requirements-optional.txt`
- Create: `uv.lock`
- Modify: `src/mnemocore/__init__.py`
- Modify: optional-import boundaries under `src/mnemocore/api/`, `mcp/`, `integrations/`, `core/`, and `subconscious/`
- Modify: `Dockerfile`
- Create: `tests/packaging/test_minimal_install.py`
- Create: `tests/packaging/test_extras_imports.py`
- Create: `tests/packaging/test_wheel_contents.py`

**Target extras:**

```text
mnemocore                    -> AgentMemory + SQLite/FTS5 only
mnemocore[api]               -> FastAPI, Uvicorn, Pydantic, Loguru, Prometheus
mnemocore[distributed]       -> Redis, Qdrant
mnemocore[vector]            -> FAISS and optional vector adapters
mnemocore[mcp]               -> MCP SDK
mnemocore[llm]               -> aiohttp, tokenizer/provider clients
mnemocore[legacy]            -> NumPy and complete HAIM compatibility stack
mnemocore[server]            -> maintained API deployment dependency set
mnemocore[benchmark]         -> benchmark-only dependencies
mnemocore[dev]               -> tests, types, lint, build, security tools
mnemocore[all]               -> all supported optional capabilities
```

- [ ] Make the base `[project.dependencies]` empty unless a direct import from the stable AgentMemory package proves a dependency is required.
- [ ] Move all non-core libraries into explicit optional extras.
- [ ] Ensure `mnemocore.agent_memory` imports successfully in a clean venv with only the wheel installed.
- [ ] Ensure root `mnemocore` import remains lazy and does not initialize legacy modules.
- [ ] Add typed, actionable errors for missing optional extras, including the exact install command.
- [ ] Use `uv.lock` as the reproducible development/release lock and `uv export` to generate Docker-compatible requirements where needed.
- [ ] Stop hand-maintaining three contradictory dependency lists. Add CI that regenerates exports and fails on drift.
- [ ] Make Docker install the built wheel with the `server` and required legacy extras instead of copying an uninstalled `src/` tree.
- [ ] Verify wheel contents exclude tests, local data, debug artifacts, private files, and benchmark outputs.
- [ ] Add a dependency-boundary test that fails if importing AgentMemory loads FastAPI, Redis, Qdrant, FAISS, OpenAI, Anthropic, or legacy engine modules.

**Exit gate:** The default package is genuinely lightweight and local; every optional capability is explicit, testable, and isolated.

---

# Workstream 4 — Make CI, Tests, and Release Status Tell the Truth

**Priority:** P0/P1.

**Child plan:** `docs/superpowers/plans/2026-07-14-ci-and-test-truth.md`

**Files:**

- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/docker-publish.yml`
- Modify: `.pre-commit-config.yaml`
- Modify: `pyproject.toml`
- Modify or remove: `setup.cfg`
- Modify: `docs/TEST_LANES.md`
- Create: `.github/workflows/nightly.yml`
- Create: `.github/workflows/benchmark.yml`
- Create: `tests/test_pytest_harness.py`

**Tooling decision:** Use Ruff for formatting and linting, mypy for type checking, pytest/Hypothesis for correctness, Bandit/pip-audit for source/dependency checks, and Trivy for OCI scanning. Remove duplicate Black/isort/flake8 execution after Ruff parity is verified.

- [ ] Put Ruff configuration in `pyproject.toml`; remove overlapping formatter/import-sorter/linter hooks.
- [ ] Make `ruff check`, `ruff format --check`, and strict AgentMemory mypy blocking.
- [ ] Add a legacy type-error baseline that can only shrink; do not hide new errors with broad `ignore` changes.
- [ ] Require AgentMemory tests on Python 3.10, 3.11, and 3.12 with minimal dependencies.
- [ ] Create separate optional-extra lanes for API, MCP, distributed/vector providers, and legacy.
- [ ] Activate a Redis-only lane and a Qdrant-only lane only when each has nonzero correctly marked test collection; reject orphan service markers.
- [ ] Keep timing-sensitive benchmarks off ordinary PRs; run benchmark contracts on PRs and full performance suites on dedicated/scheduled runners.
- [ ] Fix `build-status` so Docker and every required job are explicitly checked.
- [ ] Build the actual wheel first, then test that exact wheel in core, extras, and Docker jobs.
- [ ] Add coverage thresholds: AgentMemory >=90%; new modules >=90% branch coverage; legacy may use a ratcheted baseline that never decreases.
- [ ] Add a generated CI summary showing required, optional, quarantined, and not-run lanes.
- [ ] Add branch-protection documentation requiring review, green status checks, up-to-date branches, and no direct pushes to protected release branches.

**Exit gate:** A green status means the declared release scope is actually green, and a red/quarantined lane is visible rather than hidden in prose.

---

# Workstream 5 — Security, Privacy, and Data-Lifecycle Hardening

**Priority:** P0/P1.

**Child plan:** `docs/superpowers/plans/2026-07-15-security-privacy-and-erasure.md`

**Files:**

- Create: `docs/SECURITY.md`
- Create: `docs/THREAT_MODEL.md`
- Create: `src/mnemocore/agent_memory/erasure.py`
- Create: `src/mnemocore/cli/erasure.py`
- Modify: `src/mnemocore/events/webhook_manager.py`
- Modify: `src/mnemocore/storage/binary_vector_compression.py`
- Modify: API exception handlers and logging configuration
- Modify: Docker/Compose/Helm security defaults
- Create: `tests/agent_memory/privacy/test_erasure.py`
- Create: `tests/security/test_no_pickle_runtime.py`
- Create: `tests/security/test_secret_redaction.py`
- Create: `tests/security/test_scope_leakage_matrix.py`

**Security decisions:**

- Runtime secret persistence stores references, not secret material.
- `forget()` remains logical and auditable.
- Physical erasure is a separate explicit offline rewrite workflow that produces a new verified database without erased content, atomically replaces the active file, and emits a content-free audit receipt.
- Runtime code contains no `pickle.loads` path.

- [ ] Threat-model local library, REST deployment, MCP, Redis/Qdrant, background workers, adapters, backups, logs, and benchmark workers.
- [ ] Replace persisted webhook secret values with `secret_ref` identifiers resolved from environment or a caller-provided secrets backend. Add migration that refuses to copy plaintext secrets forward.
- [ ] Remove runtime legacy pickle fallback. Provide an isolated migration command only if users demonstrably need it; warn that it must run in a disposable environment on trusted files.
- [ ] Implement exact-scope physical erasure by streaming verified retained events/projections into a new SQLite database, rebuilding indexes, verifying row/content exclusion, fsyncing, and atomically replacing the original file. Never mutate immutable ledger rows in place.
- [ ] Emit an erasure receipt containing scope, record counts, timestamp, old/new database hashes, and no erased content.
- [ ] Document that backups and replicated stores require separate deletion policies.
- [ ] Sanitize all API error responses; internal exceptions and backend addresses must not be returned to callers.
- [ ] Ensure logs and traces redact API keys, webhook secrets, authorization headers, memory content, and free-form metadata by default.
- [ ] Validate secure CORS, API-key startup, trusted-proxy handling, request-size limits, and per-route rate limits in tests.
- [ ] Pin GitHub Actions by commit SHA for release/security workflows; pin container base images by digest for release builds.
- [ ] Generate SBOM and provenance attestations and scan the built image before publishing.

**Exit gate:** No high/critical release findings remain, secret material is not persisted by MnemoCore, and users have a truthful distinction between forget and physical erasure.

---

# Workstream 6 — Freeze and Harden the Stable AgentMemory Contract

**Priority:** P1 and prerequisite for retrieval/context expansion.

**Child plan:** `docs/superpowers/plans/2026-07-16-agent-memory-contract-hardening.md`

**Files:**

- Modify: `src/mnemocore/agent_memory/models.py`
- Modify: `src/mnemocore/agent_memory/client.py`
- Modify: `src/mnemocore/agent_memory/store.py`
- Modify: `src/mnemocore/agent_memory/sqlite_store.py`
- Modify: `src/mnemocore/agent_memory/schema.py`
- Modify: `src/mnemocore/agent_memory/timeline.py`
- Create: `tests/agent_memory/contract/`
- Create: `tests/agent_memory/migrations/`
- Create: `docs/API_STABILITY_AGENT_MEMORY.md`

**Stable invariants:**

- Exact scope is always explicit.
- Public models are immutable, validated, and JSON-serializable through documented codecs.
- Async and sync clients have behavior parity.
- Sync clients reject active event loops.
- All writes are atomic and idempotent where an idempotency key is supplied.
- Explain and context outputs carry provenance.
- Rebuild can reproduce projections from events.

- [ ] Define the stable public import list and freeze it in contract tests.
- [ ] Define serialization JSON schemas for public request/response models.
- [ ] Add golden tests for every operation across async and sync clients.
- [ ] Add property tests for scope keys, timestamp normalization, half-open validity windows, idempotency, lifecycle transitions, and invalid values.
- [ ] Add multi-process SQLite tests for concurrent remember/recall/supersede/rebuild operations.
- [ ] Add failure-injection tests for disk-full, locked database, interrupted migration, malformed rows, and damaged FTS projection.
- [ ] Make rebuild verify counts, hashes, lifecycle rows, evidence, relations, history, and FTS parity before commit.
- [ ] Create schema fixtures for every supported version and test upgrade/reopen/rebuild.
- [ ] Add a capability-report API so adapters can detect supported schema version, retrieval providers, erasure, and context features without probing private attributes.
- [ ] Document deprecation policy: stable public operations require at least two minor releases of warning before removal, except security emergency removal.

**Exit gate:** New retrieval and integration code can depend on a tested, documented, stable contract rather than private SQLite implementation details.

---

# Workstream 7 — Implement Explainable Hybrid Retrieval Without Making Services Mandatory

**Priority:** P1, major product-quality work.

**Child plan:** `docs/superpowers/plans/2026-07-17-hybrid-retrieval.md`

**Files:**

- Create: `src/mnemocore/agent_memory/retrieval/base.py`
- Create: `src/mnemocore/agent_memory/retrieval/policy.py`
- Create: `src/mnemocore/agent_memory/retrieval/lexical.py`
- Create: `src/mnemocore/agent_memory/retrieval/fusion.py`
- Create: `src/mnemocore/agent_memory/retrieval/embedding.py`
- Create: `src/mnemocore/agent_memory/retrieval/hdv.py`
- Create: `src/mnemocore/agent_memory/retrieval/explain.py`
- Modify: `src/mnemocore/agent_memory/models.py`
- Modify: `src/mnemocore/agent_memory/store.py`
- Modify: `src/mnemocore/agent_memory/sqlite_store.py`
- Modify: `src/mnemocore/agent_memory/client.py`
- Create: `tests/agent_memory/retrieval/`

**Public interfaces to introduce in the child plan:**

```python
@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    text: str
    kinds: tuple[MemoryKind, ...]
    limit: int
    valid_at: datetime | None
    known_at: datetime
    include_ancestors: bool

@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    memory_id: str
    provider: str
    rank: int
    raw_score: float
    evidence_ids: tuple[str, ...]

class CandidateProvider(Protocol):
    name: str
    async def retrieve(
        self,
        scope: MemoryScope,
        query: RetrievalQuery,
        policy: RetrievalPolicy,
    ) -> Sequence[RetrievalCandidate]: ...
```

The child plan may refine field names only before the first implementation PR; after merge, later plans must consume the exact contract.

- [ ] Extract current FTS5 behavior into a lexical provider without changing baseline results.
- [ ] Gather a union of candidates before ranking; providers may introduce candidates independently.
- [ ] Implement deterministic weighted reciprocal-rank fusion with a versioned scoring profile and finite-score validation.
- [ ] Apply scope and temporal policy before final ranking and before returning any content.
- [ ] Add reliability, contradiction/status, diversity, and optional recency adjustments as named score components.
- [ ] Add optional caller-supplied embedding provider with explicit model ID, vector dimension, and embedding-version registry. No embedding model is installed by default.
- [ ] Add optional HDV candidate/rerank provider behind the `vector` or `legacy` extra; never use it as an authorization boundary.
- [ ] Keep lexical-only behavior as the default when no optional providers are configured.
- [ ] Return human-readable retrieval reasons plus machine-readable score components and provider ranks.
- [ ] Add adversarial tests for ambiguous queries, no lexical overlap, stale facts, contradictions, ancestor scopes, and provider failure.
- [ ] If an optional provider fails, return a typed degraded-mode trace; never silently present partial retrieval as full hybrid success.
- [ ] Benchmark lexical baseline vs candidate union vs fusion vs optional rerank under identical datasets.

**Exit gate:** Hybrid retrieval improves measured recall/answer quality without weakening lexical exact hits, scope isolation, temporal correctness, local defaults, or explanation fidelity.

---

# Workstream 8 — Upgrade Context Compilation and Add Safe Learning Primitives

**Priority:** P1/P2.

**Child plan:** `docs/superpowers/plans/2026-07-18-context-feedback-and-compaction.md`

**Files:**

- Modify: `src/mnemocore/agent_memory/context.py`
- Modify: `src/mnemocore/agent_memory/models.py`
- Create: `src/mnemocore/agent_memory/feedback.py`
- Create: `src/mnemocore/agent_memory/compaction.py`
- Modify: `src/mnemocore/agent_memory/schema.py`
- Modify: `src/mnemocore/agent_memory/client.py`
- Create: `tests/agent_memory/context/`
- Create: `tests/agent_memory/feedback/`
- Create: `tests/agent_memory/compaction/`

**Boundary:** MnemoCore may record outcomes, calculate evidence-backed memory reliability, propose procedures, and compile context. It must not decide a user's task, select a model/tool, or execute a procedure.

- [ ] Introduce a versioned `ContextPolicy` with total token budget, per-level allocations, min/max items, pinned kinds, diversity, and truncation rules.
- [ ] Preserve fixed levels: core, working, episodic, semantic, and procedural.
- [ ] Make the token estimator injectable; keep deterministic `ceil(len/4)` as the zero-dependency default.
- [ ] Add stable serialization of ContextPack into plain text and structured JSON without losing receipts.
- [ ] Add `feedback()` that appends an outcome event and updates evidence-backed reliability projections atomically.
- [ ] Add procedure **proposals** sourced from repeated successful episodes, including trigger, steps, source episode IDs, success/failure counts, confidence, and review policy.
- [ ] Add failure-warning proposals sourced from repeated failed episodes.
- [ ] Default high-impact proposal policy to `propose`; do not auto-promote procedures without an explicit configured policy and validation evidence.
- [ ] Add bounded compaction plans: select source records, create proposed summary, validate complete lineage/coverage, commit atomically, retain rollback metadata, and rebuild affected indexes in the same transaction.
- [ ] Never delete source evidence during normal compaction. Use status/projection policy to keep compacted sources out of default context while retaining explainability.
- [ ] Add compaction rollback and rebuild tests.
- [ ] Add context-quality tests for source coverage, duplicate prevention, token-budget compliance, priority stability, and maliciously long content.

**Exit gate:** MnemoCore returns smaller, policy-driven, receipt-bearing context and learns safe memory metadata from outcomes without crossing into agent execution.

---

# Workstream 9 — Align Python, REST, MCP, CLI, and Framework Adapters

**Priority:** P2 after stable contract.

**Child plan:** `docs/superpowers/plans/2026-07-19-interface-parity.md`

**Files:**

- Modify/create route modules under `src/mnemocore/api/routes/`
- Modify: `src/mnemocore/api/models.py`
- Modify: `src/mnemocore/cli/main.py`
- Modify: `src/mnemocore/mcp/server.py`
- Modify: `src/mnemocore/integrations/bridge.py`
- Modify thin adapters: `langgraph.py`, `crewai.py`, `openclaw.py`, `mcp.py`
- Create: `tests/contracts/`
- Create: `docs/AGENT_MEMORY_API.md`
- Create: `docs/AGENT_MEMORY_MCP.md`
- Create: `docs/AGENT_MEMORY_CLI.md`

**Rules:**

- Every adapter calls public `AgentMemory`/store contracts only.
- No adapter imports legacy engine internals.
- No adapter performs agent planning or tool execution.
- The same fields and semantics appear across Python, REST, MCP, and CLI.

- [ ] Publish versioned `/v1/memory` routes for remember, recall, supersede, explain, history, compile-context, forget, rebuild, feedback, and capabilities.
- [ ] Use shared request/response schemas generated from or mapped directly to the public models.
- [ ] Add CLI groups with machine-readable `--json` output and human-readable default output.
- [ ] Expose minimal MCP tools/resources for memory operations; do not expose internal cognitive engine controls under the stable AgentMemory namespace.
- [ ] Add contract fixtures that execute the same scenario through Python async, Python sync, REST, MCP, and CLI and compare normalized outputs.
- [ ] Add cancellation, timeout, and error-mapping tests.
- [ ] Ensure adapters accept an already-created client or endpoint configuration rather than creating hidden global singletons.
- [ ] Mark framework adapters experimental until their contract suite and one end-to-end example pass.

**Exit gate:** Users can change transport or framework without changing memory semantics or losing scope, temporal, provenance, or error guarantees.

---

# Workstream 10 — Build Reproducible Quality, Performance, and Competitive Proof

**Priority:** P1/P2; no major public claims before completion.

**Child plan:** `docs/superpowers/plans/2026-07-20-agent-memory-evaluation.md`

**Files:**

- Refactor: `benchmarks/agent_memory_baseline.py` into `benchmarks/agent_memory/`
- Create: `benchmarks/agent_memory/schemas/result.schema.json`
- Create: `benchmarks/agent_memory/temporal_suite.py`
- Create: `benchmarks/agent_memory/scope_isolation_suite.py`
- Create: `benchmarks/agent_memory/provenance_suite.py`
- Create: `benchmarks/agent_memory/context_suite.py`
- Create external adapters for LoCoMo, LongMemEval, and BEAM
- Create: `benchmark_configs/`
- Create: `benchmark_results/reference/README.md`
- Modify: `.github/workflows/benchmark.yml`
- Create: `docs/BENCHMARK_METHODOLOGY.md`

**Metric hierarchy:**

1. Retrieval and answer correctness.
2. Scope isolation and temporal truth.
3. Provenance fidelity.
4. Context tokens and source coverage.
5. Retrieval/ingestion latency.
6. Index/database size and RSS.
7. Model calls and cost only when an explicit benchmark uses a model.

- [ ] Preserve fresh-process, fresh-database, fixed-corpus, no-telemetry baseline behavior.
- [ ] Version the result JSON schema and validate every emitted artifact.
- [ ] Pin dataset revisions, model IDs, prompts, token budgets, seeds, dependencies, and container image digests.
- [ ] Add custom truth-timeline cases with corrections, delayed knowledge, contradictions, and exact boundary instants.
- [ ] Add exhaustive scope-leakage matrices and ancestor-scope policy cases.
- [ ] Add provenance tests that verify every answer/context item traces to immutable evidence.
- [ ] Add compaction tests for coverage and rollback.
- [ ] Add repeated-task tests that measure whether procedure/failure memory reduces repeated errors without executing procedures inside MnemoCore.
- [ ] Compare against selected open-source systems only through clean adapters and identical budgets. Do not copy private prompts or implementation.
- [ ] Publish raw runs and a generated Markdown report as workflow artifacts and release assets.
- [ ] Add regression thresholds only after a stable reference baseline exists; use statistically defensible repetitions and report distributions, not one number.
- [ ] Keep performance results separate by hardware profile.

**Exit gate:** A third party can reproduce every published MnemoCore claim from the repository and raw artifacts.

---

# Workstream 11 — Contain, Stabilize, and Rationalize Legacy HAIM

**Priority:** P2; protect the stable core from legacy drag.

**Child plan:** `docs/superpowers/plans/2026-07-21-legacy-haim-containment.md`

**Files:**

- Create: `docs/LEGACY_HAIM_POLICY.md`
- Modify: `README.md`
- Modify: `pyproject.toml` extras
- Modify: root lazy exports
- Modify: legacy test layout/markers
- Review: `core/`, `cognitive/`, `subconscious/`, `storage/`, `events/`
- Create: `tests/architecture/test_dependency_direction.py`

**Policy:**

- Legacy HAIM remains available and open-source as experimental research/compatibility functionality.
- It does not define AgentMemory correctness or release readiness.
- It receives security fixes, data-integrity fixes, and bounded maintenance; new agent-runtime features are rejected.

- [ ] Add an architecture test proving `mnemocore.agent_memory` does not import legacy modules.
- [ ] Move legacy dependencies behind the `legacy` extra.
- [ ] Relabel legacy tests by unit, service integration, slow, stress, and known-red quarantine.
- [ ] Convert quarantine from filename-specific exclusion to a maintained manifest containing owner, reason, reproduction command, and unquarantine criteria.
- [ ] Fix or remove obsolete tests that encode behavior no longer supported; never make them green by broad assertion weakening.
- [ ] Stop adding new cognitive modules until existing end-to-end quality is measured.
- [ ] Produce a decision record after the proof release: retain in-tree, extract to a separate `mnemocore-haim` distribution, or deprecate specific modules. Base the decision on users, maintenance load, security surface, and benchmark value.
- [ ] Preserve import compatibility through documented shims if extraction is chosen.

**Exit gate:** The stable AgentMemory product can release independently, while legacy research remains usable without silently expanding the supported surface.

---

# Workstream 12 — Documentation, Examples, Release Candidate, and Community Readiness

**Priority:** Final release workstream.

**Child plan:** `docs/superpowers/plans/2026-07-22-proof-release.md`

**Files:**

- Rewrite: `README.md`
- Update: `CHANGELOG.md`
- Update: `ROADMAP.md`
- Update: `RELEASE_CHECKLIST.md`
- Create: `docs/MIGRATION_AGENT_MEMORY.md`
- Create: `docs/RELEASE.md`
- Update examples under `examples/`
- Create: `tests/docs/test_examples.py`
- Modify release and container-publish workflows

**README first-screen order:**

1. One-sentence product promise.
2. Minimal install.
3. 15-line AgentMemory quickstart.
4. Truth Timeline example.
5. Explainable recall/context receipt example.
6. Supported/stability matrix.
7. Optional extras.
8. Benchmarks with direct artifact references.
9. Legacy/research section.

- [ ] Correct all install paths, package names, URLs, version labels, and public/private language.
- [ ] Test every Python, CLI, and Docker quickstart in CI.
- [ ] Add migration guidance from legacy `Memory`/HAIM APIs to AgentMemory.
- [ ] Publish a concise compatibility and deprecation matrix.
- [ ] Generate API docs from stable models and contract tests where possible.
- [ ] Add security-reporting instructions and supported-version policy.
- [ ] Build wheel, sdist, SBOM, signatures/attestations, GHCR image, benchmark raw JSON, and generated report from the same tagged commit.
- [ ] Run the complete release checklist against the exact artifacts, not the working tree.
- [ ] Create a release candidate first; promote only after clean-install, upgrade, rollback, Docker, benchmark, and documentation verification.
- [ ] State limitations explicitly: optional provider maturity, benchmark scope, legacy experimental status, and physical-erasure backup responsibilities.

**Exit gate:** A new user can install the minimal package, complete the quickstart, understand stability boundaries, reproduce evidence, and deploy the maintained server without relying on undocumented repository knowledge.

---

## 6. Recommended Pull-Request Sequence

| Order | PR | Scope | Depends on | Merge gate |
|---:|---|---|---|---|
| 1 | Baseline/status reconciliation | Docs and evidence only | None | Commands and status verified |
| 2 | Version authority + URL hotfix | Release metadata only | PR 1 | Clean wheel version contract |
| 3 | Product boundary + stability policy | Governance/docs | PR 1 | Owner approval |
| 4 | Minimal package + extras | Packaging | PR 2–3 | Minimal-install contract passes |
| 5 | Ruff/mypy/CI truth | Tooling/CI | PR 4 | All declared required lanes accurate |
| 6 | Docker/release artifact correctness | Build/deploy | PR 2,4,5 | Built wheel works in image |
| 7 | Security/privacy hardening part A | Secrets, pickle, errors | PR 4–6 | Security lanes green |
| 8 | AgentMemory contract freeze | Public models/contracts | PR 4–5 | Contract/migration/concurrency tests |
| 9 | Physical erasure workflow | Data lifecycle | PR 7–8 | Rewrite, verification, recovery tests |
| 10 | Retrieval interfaces + lexical extraction | No behavior change | PR 8 | Lexical baseline parity |
| 11 | Candidate union + rank fusion | Retrieval behavior | PR 10 | Quality and safety tests |
| 12 | Optional embedding/HDV providers | Optional retrieval | PR 11 | Minimal install unchanged |
| 13 | ContextPolicy v2 | Context | PR 8,11 | Token/receipt invariants |
| 14 | Feedback/procedure proposals | Learning primitives | PR 8,13 | No execution behavior |
| 15 | Evidence-preserving compaction | Memory lifecycle | PR 14 | Coverage/rollback/rebuild tests |
| 16 | REST/MCP/CLI parity | Adapters | PR 8,13–15 | Cross-transport contract suite |
| 17 | Evaluation framework | Benchmarks | PR 10–16 | Reproducible raw artifacts |
| 18 | Legacy containment | Architecture/deps | PR 4–5,8 | Dependency-direction test |
| 19 | Proof-release docs and artifacts | Release | All | Full checklist on built artifacts |

Do not combine PRs 8–15 into one large feature branch. Public model, schema, retrieval, context, feedback, and compaction changes need separate review and migration gates.

---

## 7. GitHub Project Setup After This Plan Is Approved

Create milestones:

1. `M0 — Release Integrity`
2. `M1 — AgentMemory Foundation`
3. `M2 — Hybrid Retrieval and Context`
4. `M3 — Reproducible Proof Release`

Create labels:

```text
priority:P0
priority:P1
priority:P2
area:agent-memory
area:retrieval
area:context
area:security
area:privacy
area:packaging
area:ci
area:api
area:mcp
area:benchmark
area:legacy
kind:bug
kind:feature
kind:refactor
kind:docs
status:blocked
status:needs-design
status:ready
stability:stable
stability:beta
stability:experimental
```

Issue creation rules:

- One issue per independently testable deliverable.
- Every issue lists exact files, interfaces, tests, migration effect, benchmark effect, and public-boundary classification.
- Parent tracking issues may group a workstream but contain no code implementation themselves.
- Security-sensitive issues omit exploit details until a fix is available; use private advisories when warranted.

---

## 8. Global PR Acceptance Checklist

Every implementation PR must satisfy all applicable items:

- [ ] Linked issue and child implementation plan.
- [ ] Change belongs inside the documented MnemoCore boundary.
- [ ] No private runtime/IP details added.
- [ ] Failing test written and observed before implementation.
- [ ] Exact affected lane passes locally.
- [ ] Minimal-install test still passes.
- [ ] Scope and temporal invariants remain covered.
- [ ] New public fields/types have validation, serialization, docs, and compatibility analysis.
- [ ] Schema changes include migration, reopen, rebuild, rollback/recovery, and fixture tests.
- [ ] New optional dependency is assigned to an explicit extra.
- [ ] Errors are typed and no silent correctness-changing fallback was added.
- [ ] Logs/traces exclude content and secrets by default.
- [ ] Hot-path changes include benchmark before/after artifacts.
- [ ] Documentation and examples are updated and executed.
- [ ] Changelog entry added under the correct stability category.
- [ ] No broad linter/type ignores added to hide new problems.
- [ ] PR is small enough for line-by-line review and safe revert.

---

## 9. Risk Register and Countermeasures

| Risk | Severity | Leading indicator | Countermeasure |
|---|---:|---|---|
| Scope creep into a full agent runtime | 5 | Task/model/tool logic proposed in MnemoCore | Enforce PRODUCT_BOUNDARY, PR checklist, CODEOWNERS |
| Legacy platform blocks stable releases | 5 | Release scope tied to unrelated known-red modules | Stable/experimental tiers and optional legacy extra |
| Wrong memory harms agent behavior | 5 | Low retrieval precision, stale/contradictory hits | Temporal policy, candidate explanations, evidence, abstention tests |
| Cross-tenant/project leakage | 5 | Any record returned outside exact authorized path | Storage predicates, exhaustive matrix tests, no similarity auth |
| Irreversible schema/data corruption | 5 | Migration mutates without verified backup/rebuild | Transactional migration, fixtures, rebuild parity, atomic replacement |
| Benchmark gaming or non-reproducibility | 4 | Claims without raw artifacts or pinned budgets | Versioned result schema and release artifacts |
| Dependency/supply-chain bloat | 4 | Minimal install pulls server/model stack | Extras, lock, wheel boundary tests, SBOM |
| Secret/content exposure | 5 | Plaintext secret persistence or verbose logs | Secret references, redaction, threat model, scans |
| Background compaction invents facts | 5 | Summary without complete source membership | Proposal mode, lineage validation, rollback, source retention |
| Public API churn | 4 | Adapters depend on private implementation | Stable contract tests and deprecation policy |
| Performance optimizations break correctness | 4 | Faster scores outside valid range or changed truth results | Correctness-first benchmarks and numeric property tests |
| Stale generated plans mislead future agents | 3 | Counts/line numbers no longer match main | Baseline date/SHA, status docs, child plans per workstream |

---

## 10. Stop Conditions for Codex

Codex must stop and request owner review rather than continue when any of these occur:

1. Public release history cannot prove the next monotonic version.
2. A schema migration cannot demonstrate transactional rollback or rebuild parity.
3. A proposed feature implements task planning, model routing, tool execution, verification, or other external-runtime behavior.
4. A change requires private prompts, policies, datasets, or product code to be committed.
5. A retrieval provider can bypass scope authorization or temporal filtering.
6. Physical erasure cannot prove erased content is absent from the replacement database and generated indexes.
7. A new default dependency breaks the local standard-library AgentMemory goal.
8. A benchmark comparison cannot use identical budgets and pinned artifacts.
9. Security tooling reports a critical/high issue in a release artifact.
10. A PR requires broad test exclusion, linter suppression, or assertion weakening to appear green.

---

## 11. First Codex Handoff Prompt

Use this exact prompt to begin execution:

```text
You are Codex working in RobinALG87/MnemoCore-Persistent-Cognitive-Ai-Memory.

Read, in order:
1. docs/superpowers/plans/2026-07-12-mnemocore-next-phase-master-plan.md
2. docs/superpowers/specs/2026-07-10-agent-memory-vnext-design.md
3. docs/AGENT_MEMORY_QUICKSTART.md
4. docs/TEST_LANES.md
5. RELEASE_CHECKLIST.md
6. PRODUCTION_REMEDIATION_PROGRESS.md
7. pyproject.toml
8. .github/workflows/ci.yml

Operate only Workstream 0. Do not modify production code.

Create:
docs/superpowers/plans/2026-07-13-mnemocore-baseline-and-triage.md

The child plan must:
- record the current main SHA;
- give exact commands and expected outcomes for every existing lane;
- define how to inspect and resolve stale PR #31 and PR #34;
- define the structure of docs/status/2026-07-12-platform-baseline.md;
- distinguish PASS, FAIL, QUARANTINED, and NOT RUN;
- use TDD where code is eventually required, but Workstream 0 itself changes documentation only;
- end at the Workstream 0 exit gate.

Do not start Workstream 1 until the baseline/status PR is reviewed and merged.
```

---

## 12. Final Program Definition of Done

MnemoCore's next phase is complete when:

1. The repository has one monotonic, automated version authority and correct public onboarding.
2. `mnemocore.agent_memory` is a minimal, stable, exact-scoped local package.
3. Stable, beta, and experimental surfaces are explicit and mechanically enforced.
4. CI status accurately represents the declared release scope.
5. Docker and release artifacts install and run the built wheel, not an uninstalled source tree.
6. Security, secret handling, logging, and physical-erasure semantics are documented and tested.
7. The AgentMemory contract is frozen, typed, migration-safe, and transport-independent.
8. Hybrid retrieval is candidate-union-first, explainable, temporally correct, and optional-provider friendly.
9. Context compilation is bounded, policy-driven, deduplicated, and receipt-bearing.
10. Outcome feedback, procedure proposals, failure warnings, and compaction remain memory primitives and never execute agent actions.
11. Python, REST, MCP, CLI, and framework adapters preserve the same semantics.
12. Benchmark claims are reproducible from pinned raw artifacts.
13. Legacy HAIM is contained as experimental compatibility/research functionality and cannot block stable AgentMemory releases.
14. The open repository contains no private agent-runtime implementation or proprietary orchestration IP.
