# MnemoCore MCP Implementation Plan (Beta)

## Goal

Expose MnemoCore capabilities through a Model Context Protocol (MCP) server so external LLM agents can safely store, query, and inspect memory with predictable contracts.

## Scope (Phase 1)

### In Scope

- MCP server process for local/dev use.
- Read/write memory tools mapped to existing engine/API capabilities.
- Basic auth + request limits aligned with existing API policy.
- Test coverage for MCP tool contracts and degraded dependencies.

### Out of Scope (Phase 1)

- Multi-tenant policy engine.
- Full distributed consensus workflows.
- New memory semantics beyond existing endpoints.

## Architecture Decision

Prefer **adapter-first** design:

- Keep `src/core` and `src/api` as source of truth.
- Add `src/mcp/server.py` (MCP transport + tool registry).
- Add `src/mcp/adapters/api_adapter.py` to reuse validated API contracts.
- Add `src/mcp/schemas.py` for tool input/output validation.

Reason: minimizes behavior drift and reuses existing validation/security paths.

## Proposed MCP Tools (Phase 1)

1. `memory_store`
   - Input: `content`, `metadata?`, `agent_id?`, `ttl?`
   - Backend: `POST /store`
2. `memory_query`
   - Input: `query`, `top_k?`, `agent_id?`
   - Backend: `POST /query`
3. `memory_get`
   - Input: `memory_id`
   - Backend: `GET /memory/{memory_id}`
4. `memory_delete`
   - Input: `memory_id`
   - Backend: `DELETE /memory/{memory_id}`
5. `memory_stats`
   - Input: none
   - Backend: `GET /stats`
6. `memory_health`
   - Input: none
   - Backend: `GET /health`

Optional (Phase 1.1):
- `concept_define` and `analogy_solve` once primary tools are stable.

## Security and Operational Guardrails

- Require API key passthrough from MCP server to MnemoCore API.
- Allowlist MCP tools (disable dangerous or experimental operations by default).
- Enforce per-tool timeout and payload limits.
- Structured logs with `trace_id`, `tool_name`, latency, status.
- Fail closed for auth errors; fail open only where existing API already degrades by design.

## Delivery Milestones

### M0: Foundations (1-2 days)

- Add MCP package structure.
- Add config section for MCP host/port/timeouts/tool allowlist.
- Add local run command and basic health check tool.

Exit criteria:
- MCP server starts and responds to health tool.

### M1: Core Read/Write Tools (2-4 days)

- Implement `memory_store`, `memory_query`, `memory_get`, `memory_delete`.
- Map errors to stable MCP error format.
- Add contract tests with mocked API responses.

Exit criteria:
- Core memory flow works end-to-end from MCP client.

### M2: Observability + Hardening (1-2 days)

- Add metrics counters/histograms for MCP tools.
- Add retry/backoff only for transient failures.
- Add degraded-mode tests (Redis/Qdrant unavailable).

Exit criteria:
- Clear diagnostics for failures and latency.

### M3: Extended Cognitive Tools (optional, 1-2 days)

- Add `concept_define` and `analogy_solve`.
- Add docs examples for agent orchestration flows.

Exit criteria:
- Conceptual tools pass contract tests and are documented.

## Test Strategy

- Unit tests: schema validation, adapter mapping, error translation.
- Functional tests: MCP client -> server -> API in local integration mode.
- Resilience tests: upstream timeout, 403 auth fail, 404 memory miss, degraded Redis.
- Regression gate: existing `tests/` suite remains green.

## Rollout Plan

1. Ship behind `mcp.enabled: false` default.
2. Enable in beta environments only.
3. Observe for one sprint (latency, error rate, tool usage).
4. Promote to default-on after stability criteria are met.

## Success Metrics

- >= 99% successful MCP tool calls in healthy environment.
- P95 MCP tool latency <= 300 ms for read operations (local setup target).
- Zero contract-breaking changes without changelog entry.

## Minimal Backlog Tasks

1. Create `src/mcp/server.py` bootstrap.
2. Create adapter + schemas.
3. Add MCP config in `config.yaml` + typed config model.
4. Add tests in `tests/test_mcp_server.py` and `tests/test_mcp_contracts.py`.
5. Add documentation section in README + API docs.
