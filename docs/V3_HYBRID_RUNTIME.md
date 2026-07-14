# v3 Hybrid Runtime Milestone

This document describes the implemented but **unreleased** v3 hybrid-runtime
milestone. The published package remains v2.0.0. It is a safe migration layer,
not a second memory database.

## What is available

- **AgentMemory is the only durable truth source.** Memory records, events,
  timeline state, and writes remain in its scoped SQLite store.
- **`HybridMemoryRuntime`** is the primary asynchronous runtime. Its
  `SyncHybridMemoryRuntime` facade offers the same behavior to synchronous
  callers and refuses nested event-loop use.
- **Retrieval is exact-scope and deterministic.** It combines lexical
  candidates with BinaryHDV reranking, pages candidates under a bounded budget,
  and exposes the scoring version (`hybrid-lexical-binary-hdv-v2`) and candidate
  counts without logging memory content.
- **Tier and graph projections are derivatives.** They can be rebuilt from the
  records in the same complete scope and are not an alternate write path.
- **Cognitive jobs propose plans, not direct writes.** Plans and their proposals
  require provenance and confidence. The runtime marks every persisted
  cognitive output as synthetic, then validates the whole plan again immediately
  before an atomic AgentMemory batch write; invalid scope, insufficient
  confidence, malformed provenance, or conflicts leave no partial change.
- **`HAIMEngineAdapter` is temporary.** It bridges legacy store/query/delete
  calls to a scope-bound AgentMemory client and emits a deprecation warning.
  The direct `HAIMEngine`, legacy REST application, CLI, and MCP server remain
  v2 compatibility surfaces; their global JSONL/tiering lifecycle is not
  AgentMemory-backed. Use `HybridMemoryRuntime` or `create_v3_app` for v3.
  `LiteEngine` and every `Memory(...)` invocation are removed; migrate to an
  explicit `MemoryScope`.

## Scope rule

Every operation requires the exact `tenant_id`, `user_id`, `agent_id`, and any
present `project_id` and `session_id`. Runtime binding, retrieval, projections,
plans, and API authorization do not broaden or infer scopes.

## HTTP deployment boundary

Use `create_v3_app(sqlite_path, scope_authorizer=...)` for a standalone v3 API.
The injected authorizer must authenticate a caller and approve its complete
requested scope. The legacy application leaves these routes disabled by default,
and a v3 application without an authorizer fails closed. This prevents a
credential that is valid in one scope from selecting another scope.

## Remaining staged work

1. Expand graph associations and projection rebuild evidence while retaining
   exact-scope provenance.
2. Add reconstructive recall and gap-detection planners with abstention,
   provenance, and conflict validation.
3. Add dream and anticipation planners only as reviewable, rollback-safe plans.
4. Define distributed ports and complete deployment-specific
   credential-to-scope authorizer wiring.
5. Record reproducible retrieval benchmarks and complete full CI, Docker, and
   upgrade/rollback validation before making production claims.

The public API and documentation intentionally describe only the generic,
standalone runtime boundary.
