# MnemoCore Agent Memory vNext Design

**Status:** Approved direction  
**Date:** 2026-07-10  
**Objective:** Build the strongest verifiable open-source memory layer for long-lived agents.

## 1. Product Promise

An agent using MnemoCore should begin each task with the right prior context, avoid repeating known failures, preserve changing truths over time, learn reusable procedures from outcomes, and explain every injected memory.

MnemoCore is a memory and context system, not an agent runtime or an LLM provider.

## 2. Success Criteria

- Reproducibly compare against OSS baselines on LoCoMo, LongMemEval, and BEAM using identical model and token budgets.
- Track retrieval quality, answer quality, context tokens, model cost, ingestion latency, retrieval latency, and index size.
- Prevent cross-scope retrieval by construction; retrieval masks and embeddings are never authorization boundaries.
- Attach provenance to every derived memory and every context item returned to an agent.
- Preserve immutable evidence through contradiction, supersession, consolidation, and deletion workflows.
- Keep the default context pack below 2,000 tokens and local non-LLM retrieval below 200 ms p95 on the reference dataset.
- Support a genuinely local default using Python standard-library SQLite; server, vector database, graph database, and LLM integrations remain optional.

## 3. Design Principles

1. **One canonical ledger:** raw observations and lifecycle events are append-only.
2. **Derived projections:** facts, episodes, procedures, profiles, and graph links can be rebuilt from ledger events.
3. **Temporal truth:** distinguish ingestion time from real-world validity time.
4. **Evidence before confidence:** confidence without source lineage is invalid.
5. **Candidate union before ranking:** lexical, semantic, graph, temporal, HDV, and procedural signals may each introduce candidates.
6. **Bounded context:** retrieval and context compilation obey explicit latency, token, and cost policies.
7. **Fail visibly:** persistence, initialization, index, extraction, and authorization failures return typed errors.
8. **Clean-room adaptation:** concepts from other OSS systems may inform schemas and tests; do not copy their prompts or implementation.

## 4. Architecture

```text
Agent events/messages/outcomes
            |
            v
Immutable Memory Ledger (SQLite local, PostgreSQL scale)
            |
            v
Extraction + policy + idempotency
            |
            +--> temporal facts
            +--> episodes
            +--> procedures
            +--> profiles/preferences
            +--> entities/relationships
                        |
                        v
Hybrid candidate union -> explainable rank fusion -> context compiler
                        |
                        v
                 Agent briefing
                        |
                        v
                 outcome feedback
```

### 4.1 Evolution Strategy

Create a focused `mnemocore.agent_memory` package alongside the legacy HAIM engine. The new package owns the public vNext contract and uses legacy components only through explicit adapters. Migrate proven components incrementally; do not rewrite or expose the existing engine internals as the new contract.

### 4.2 Canonical Storage

The local backend is SQLite in WAL mode with foreign keys enabled.

Core tables:

- `memory_events`: immutable event ledger with idempotency key, scope, event type, payload, source, and timestamps.
- `memories`: current materialized records with type, content, temporal validity, confidence, status, and scope.
- `memory_evidence`: many-to-many links from derived records to source events or memories.
- `memory_relations`: typed, temporal, provenance-bearing relationships.
- `memory_history`: materialized lifecycle transitions for fast inspection.
- `memory_fts`: FTS5 lexical projection maintained transactionally.

All destructive-looking operations append an event and update projections in one transaction. Raw evidence is not silently removed. Privacy erasure is a separate explicit operation that records an audit tombstone without retaining erased content.

### 4.3 Scope and Authorization

Every event and memory carries a canonical scope:

```text
tenant -> user -> agent -> project -> session
```

Storage predicates enforce permitted scope paths. Missing scope is rejected for agent-facing writes. Shared memory requires an explicit capability grant. XOR/HDV masking may partition retrieval but never grants access.

### 4.4 Memory Types

- `observation`: immutable source material.
- `fact`: self-contained proposition with validity interval and supersession links.
- `episode`: goal, ordered events, outcome, reward, and lessons.
- `procedure`: trigger, ordered steps, evidence, success/failure counts, and reliability.
- `preference`: explicit or inferred preference with subject and confidence.
- `summary`: derived compression with complete source membership.

### 4.5 Ingestion Pipeline

1. Validate scope, size, timestamps, and idempotency key.
2. Persist the raw observation before optional model work.
3. Select extraction policy: `none`, `rules`, `llm_sync`, or `llm_background`.
4. Produce typed memory proposals with source spans and confidence.
5. Resolve exact and semantic duplicates.
6. Detect compatible facts, contradictions, and supersession.
7. Commit projections, evidence, relations, lexical index, and history atomically.

LLM extraction is optional, provider-neutral, structured, batchable, and observable.

### 4.6 Retrieval Pipeline

1. Enforce authorized scopes and memory-type policy.
2. Classify query needs without requiring an LLM.
3. Gather a union of candidates from FTS5, embeddings, entity/graph links, temporal filters, procedures, and optional HDV similarity.
4. Normalize and fuse signals using a versioned scoring profile.
5. Apply validity, contradiction, reliability, diversity, and privacy policies.
6. Optionally rerank under explicit cost/latency policy.
7. Return typed results with content, provenance, validity, confidence, score components, and a human-readable retrieval reason.

Recency is one signal, never a proxy for current truth.

### 4.7 Context Compiler

Compile context into explicit levels:

- `core`: pinned identity, policy, and critical preferences.
- `working`: current session goals and recent state.
- `episodic`: relevant prior attempts and outcomes.
- `semantic`: facts and relationships.
- `procedural`: proven methods and failure warnings.

Each level has a token allocation, minimum/maximum items, visibility policy, and truncation strategy. Every emitted item carries a memory receipt.

### 4.8 Learning and Compaction

Outcome feedback updates evidence-backed reliability rather than blindly mutating embeddings.

Repeated successful episodes may propose a procedure. Repeated failures generate a failure warning. Contradictions create temporal versions. Compaction runs as a bounded plan:

1. select candidates;
2. create proposed summary;
3. validate coverage and lineage;
4. commit atomically;
5. retain rollback metadata;
6. repair all projections in the same transaction.

High-impact procedural or profile changes support `automatic`, `propose`, and `manual` policies.

## 5. Public Contract

Primary async API with a separate sync wrapper; no nested event-loop thread bridging.

```python
memory = AgentMemory.open(
    path="./data/mnemocore.db",
    scope=MemoryScope(user_id="robin", agent_id="codex", project_id="mnemocore"),
)

session = await memory.start_session(goal="Improve retrieval")
await session.remember("BM25 must introduce candidates", kind="decision")
context = await session.recall("How should retrieval fusion work?")
await session.finish(outcome="success", reward=1.0)
```

Stable operations:

- `remember`, `recall`, `get`, `list`, `history`, `forget`
- `start_session`, `observe`, `finish_session`
- `feedback`, `reflect`, `explain`
- `compile_context`

REST, MCP, CLI, Python sync, and future TypeScript clients map to the same request and response models.

## 6. Agent Experience

- **Mission Brief:** pre-task context with decisions, risks, and relevant procedures.
- **Failure Shield:** warns about evidence-backed failed approaches.
- **Skill Forge:** proposes procedures from repeated successful episodes.
- **Truth Timeline:** queries what is true now or at a historical time.
- **Memory Receipts:** exposes source, confidence, validity, and retrieval explanation.
- **Dream Review:** presents safe background consolidation and association results.

## 7. Observability

Emit OpenTelemetry/OpenInference-compatible spans for validation, storage, extraction, deduplication, contradiction handling, candidate generation, fusion, reranking, context compilation, feedback, and compaction.

Each retrieval trace records candidate counts per signal, filters, scoring profile version, token budget, chosen items, latency, and errors without logging private content by default.

## 8. Evaluation

Create a reproducible benchmark harness with pinned datasets, model configuration, prompts, and budgets.

External suites:

- LoCoMo
- LongMemEval
- BEAM

MnemoCore-specific suites:

- temporal currentness and point-in-time recall;
- contradiction resolution and abstention;
- cross-agent and cross-project leakage;
- provenance fidelity;
- compaction rollback and evidence coverage;
- repeated-error reduction and time-to-solve for coding agents;
- token, latency, cost, and storage growth.

Do not publish "best" claims until results are reproducible from the OSS repository.

## 9. Migration and Delivery

### Sub-project A: Foundation

Typed models, scope enforcement, SQLite ledger, FTS5 projection, history, and minimal async/sync facade.

### Sub-project B: Hybrid Retrieval

Candidate providers, rank fusion, score explanations, temporal policies, and optional embeddings/HDV.

### Sub-project C: Agent Lifecycle

Sessions, episodes, outcome feedback, failure warnings, and procedures.

### Sub-project D: Extraction and Temporal Knowledge

Provider-neutral extraction, deduplication, contradiction handling, temporal facts, entity links, and provenance.

### Sub-project E: Context and Compaction

Context compiler, memory receipts, transactional compaction, reflection, and proposal workflow.

### Sub-project F: Ecosystem and Proof

MCP/REST/CLI, OpenClaw/LangGraph/Codex integrations, benchmark adapters, reproducible results, and migration documentation.

Each sub-project must ship independently working software, focused tests, documentation, and a benchmark or behavioral acceptance test.

## 10. Compatibility and Non-Goals

- Keep the legacy API operational during migration.
- Do not make Redis, Qdrant, a graph database, or an LLM mandatory.
- Do not expose legacy feature flags in the new public API.
- Do not copy Mem0, Graphiti, Letta, or other projects' prompts or code.
- Do not add new cognitive modules until the existing lifecycle has measurable end-to-end quality.
