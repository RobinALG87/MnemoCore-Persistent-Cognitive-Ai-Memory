# MnemoCore Beta Roadmap

## Scope and Intent

This roadmap describes current known gaps and likely direction.
It is not a promise, delivery guarantee, or commitment to specific timelines.

## Known Gaps in Beta

- Query path is still primarily HOT-tier-centric in current engine behavior.
- Some consolidation pathways are partial or under active refinement.
- Certain integrations (LLM/Nightlab) are intentionally marked as TODO.
- Distributed-scale behavior from long-form blueprints is not fully productized in this public beta.

## Near-Term Priorities

1. Improve cross-tier retrieval consistency.
2. Harden consolidation and archival flow.
3. Improve deletion semantics and API consistency.
4. Expand tests around degraded dependency modes (Redis/Qdrant outages).
5. Stabilize API contracts and publish versioned compatibility notes.
6. Introduce MCP server integration for agent tool access (see `docs/MCP_IMPLEMENTATION_PLAN.md`).

## Mid-Term Priorities

1. Better batch operations and indexing performance.
2. More robust observability and operational diagnostics.
3. Cleaner migration paths for evolving storage formats.
4. Optional integration interfaces for external LLM/reasoning systems.

## Not a Commitment

Items above are directional only.
Order, scope, and implementation details can change during beta.

