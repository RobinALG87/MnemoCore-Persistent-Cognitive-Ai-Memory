# MnemoCore Self-Improvement Deep Dive

Status: Design document (pre-implementation)  
Date: 2026-02-18  
Scope: Latent, always-on memory self-improvement loop that runs safely in production-like beta.

## 1. Purpose

This document defines a production-safe design for a latent self-improvement loop in MnemoCore.  
The goal is to continuously improve memory quality over time without corrupting truth, overloading resources, or breaking temporal-memory behavior.

Primary outcomes:
- Better memory quality (clarity, consistency, retrieval utility).
- Better long-term structure (less duplication, stronger semantic links).
- Preserved auditability and rollback.
- Compatibility with temporal timelines (`previous_id`, `unix_timestamp`, time-range search).

## 2. Current System Baseline

Relevant existing mechanisms already in code:
- `HAIMEngine.store/query` orchestration and subconscious queue (`src/core/engine.py`).
- Background dream strengthening and synaptic binding (`src/core/engine.py`).
- Gap detection and autonomous gap filling (`src/core/gap_detector.py`, `src/core/gap_filler.py`).
- Semantic consolidation workers (`src/core/semantic_consolidation.py`, `src/subconscious/consolidation_worker.py`).
- Subconscious daemon loop with LLM-powered cycles (`src/subconscious/daemon.py`).
- Temporal memory fields in node model (`src/core/node.py`): `previous_id`, `unix_timestamp`, `iso_date`.
- Tiered persistence and time-range aware search (`src/core/tier_manager.py`, `src/core/qdrant_store.py`).

Implication: Self-improvement should reuse these pathways, not bypass them.

## 3. Problem Definition

Without a dedicated self-improvement loop, memory quality drifts:
- Duplicate or near-duplicate content accumulates.
- Weakly structured notes remain unnormalized.
- Conflicting memories are not actively reconciled.
- Query utility depends too much on initial storage quality.

At the same time, naive autonomous rewriting is risky:
- Hallucinated edits can reduce truth quality.
- Over-aggressive rewriting can erase provenance.
- Continuous background jobs can starve main workloads.

## 4. Design Principles

1. Append-only evolution, never destructive overwrite.
2. Improvement proposals must pass validation gates before commit.
3. Full provenance and rollback path for every derived memory.
4. Temporal consistency is mandatory (timeline must remain navigable).
5. Resource budgets and kill switches must exist from day 1.

## 5. Target Architecture

## 5.1 New Component

Add `SelfImprovementWorker` as a background worker (similar lifecycle style to consolidation/gap-filler workers).

Suggested location:
- `src/subconscious/self_improvement_worker.py`

Responsibilities:
- Select candidates from HOT/WARM.
- Produce improvement proposals (rule-based first, optional LLM later).
- Validate proposals.
- Commit accepted proposals via `engine.store(...)`.
- Link provenance metadata.
- Emit metrics and decision logs.

## 5.2 Data Flow

1. Candidate Selection  
2. Proposal Generation  
3. Validation & Scoring  
4. Commit as New Memory  
5. Link Graph/Timeline  
6. Monitor + Feedback Loop

No in-place mutation of existing memory content.

## 5.3 Integration Points

- Read candidates: `TierManager` (`hot`, optional warm sampling).
- Commit: `HAIMEngine.store(...)` so all normal indexing/persistence paths apply.
- Timeline compatibility: preserve `previous_id` semantics and set provenance fields.
- Optional post-effects: trigger low-priority synapse/link updates.

## 6. Memory Model Additions (Metadata, not schema break)

Use metadata keys first (backward compatible):
- `source: "self_improvement"`
- `improvement_type: "normalize" | "summarize" | "deduplicate" | "reconcile"`
- `derived_from: "<node_id>"`
- `derived_from_many: [node_ids...]` (for merge/reconcile)
- `improvement_score: float`
- `validator_scores: { ... }`
- `supersedes: "<node_id>"` (logical supersedence, not deletion)
- `version_tag: "vN"`
- `safety_mode: "strict" | "balanced"`

Note: Keep temporal fields from `MemoryNode` untouched and naturally generated on store.

## 7. Candidate Selection Strategy

Initial heuristics (cheap and deterministic):
- High access + low confidence retrieval history.
- Conflicting memories in same topical cluster.
- Redundant near-duplicates.
- Old high-value memories needing compaction.

Selection constraints:
- Batch cap per cycle.
- Max candidates per source cluster.
- Cooldown per `node_id` to avoid thrashing.

## 8. Proposal Generation Strategy

Phase A (no LLM dependency):
- Normalize formatting.
- Metadata repair/completion.
- Deterministic summary extraction.
- Exact/near duplicate merge suggestion.

Phase B (LLM-assisted, guarded):
- Rewrite for clarity.
- Multi-memory reconciliation draft.
- Explicit uncertainty markup if conflict unresolved.

All proposals must include rationale + structured diff summary.

## 9. Validation Gates (Critical)

A proposal is committed only if all required gates pass:

1. Semantic drift gate  
- Similarity to origin must stay above threshold unless `improvement_type=reconcile`.

2. Fact safety gate  
- No new unsupported claims for strict mode.
- If unresolved conflict: enforce explicit uncertainty markers.

3. Structure gate  
- Must improve readability/compactness score beyond threshold.

4. Policy gate  
- Block forbidden metadata changes.
- Block sensitive tags crossing trust boundaries.

5. Resource gate  
- Cycle budget, latency budget, queue/backpressure checks.

Rejected proposals are logged but not committed.

## 10. Interaction with Temporal Memory (Hard Requirement)

This design must not break timeline behavior introduced around:
- `previous_id` chaining
- `unix_timestamp` payload filtering
- Qdrant time-range retrieval

Rules:
- Every improved memory is a new timeline event (new node id).
- `derived_from` models lineage; `previous_id` continues temporal sequence.
- Query paths that use `time_range` must continue functioning identically.
- Do not bypass `TierManager.add_memory` or Qdrant payload generation.

## 11. Safety Controls & Operations

Mandatory controls:
- Config kill switch: `self_improvement_enabled: false` by default initially.
- Dry-run mode: generate + validate, but do not store.
- Strict mode for early rollout.
- Per-cycle hard caps (count, wall-clock, token budget).
- Circuit breaker on repeated validation failures.

Operational observability:
- Attempted/accepted/rejected counters.
- Rejection reasons cardinality-safe labels.
- End-to-end cycle duration.
- Queue depth and backlog age.
- Quality delta trend over time.

## 12. Suggested Config Block

Add under `haim.dream_loop` or sibling block `haim.self_improvement`:

```yaml
self_improvement:
  enabled: false
  dry_run: true
  safety_mode: "strict"          # strict | balanced
  interval_seconds: 300
  batch_size: 8
  max_cycle_seconds: 20
  max_candidates_per_topic: 2
  cooldown_minutes: 120
  min_improvement_score: 0.15
  min_semantic_similarity: 0.82
  allow_llm_rewrite: false
```

## 13. Metrics (Proposed)

- `mnemocore_self_improve_attempts_total`
- `mnemocore_self_improve_commits_total`
- `mnemocore_self_improve_rejects_total`
- `mnemocore_self_improve_cycle_duration_seconds`
- `mnemocore_self_improve_candidates_in_cycle`
- `mnemocore_self_improve_quality_delta`
- `mnemocore_self_improve_backpressure_skips_total`

## 14. Phased Implementation Plan

Phase 0: Instrumentation + dry-run only  
- Add worker scaffold + metrics + decision logs.  
- No writes.

Phase 1: Deterministic improvements only  
- Metadata normalization, duplicate handling suggestions.  
- Strict validation.  
- Commit append-only derived nodes.

Phase 2: Controlled LLM improvements  
- Enable `allow_llm_rewrite` behind feature flag.  
- Add stricter validation and capped throughput.

Phase 3: Reconciliation and adaptive policies  
- Multi-memory conflict reconciliation.
- Learning policies from acceptance/rejection outcomes.

## 15. Test Strategy

Unit tests:
- Candidate selection determinism and cooldown behavior.
- Validation gates (pass/fail matrices).
- Provenance metadata correctness.

Integration tests:
- Store/query behavior unchanged under disabled mode.
- Time-range query still correct with improved nodes present.
- Qdrant payload contains expected temporal + provenance fields.

Soak/load tests:
- Worker under sustained ingest.
- Backpressure behavior.
- No unbounded queue growth.

Regression guardrails:
- No overwrite of original content.
- No bypass path around `engine.store`.

## 16. Risks and Mitigations

Risk: hallucinated improvements  
Mitigation: strict mode, no-LLM phase first, fact safety gate.

Risk: timeline noise from too many derived nodes  
Mitigation: cooldown, batch caps, minimum score thresholds.

Risk: resource contention  
Mitigation: cycle time caps, skip when main queue/backlog high.

Risk: provenance complexity  
Mitigation: standardized metadata contract and audit logs.

## 17. Open Decisions

1. Should self-improved nodes be visible by default in top-k query, or weighted down unless requested?  
2. Should `supersedes` influence retrieval ranking automatically?  
3. Do we need a dedicated “truth tier” for validated reconciled memories?

## 18. Recommended Next Step

Implement Phase 0 only:
- Worker skeleton
- Config block
- Metrics
- Dry-run reports

Then review logs for 1-2 weeks before enabling any writes.
