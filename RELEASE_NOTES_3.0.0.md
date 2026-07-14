# MnemoCore 3.0.0 release notes

## Release status

**Public beta.** MnemoCore 3.0.0 is ready for local evaluation and scoped,
single-node AgentMemory deployments. It is not a claim of distributed
durability, full recovery validation, or general production certification.

## What ships

- **AgentMemory as the durable truth source.** Scoped SQLite records, events,
  history, and timeline state remain the only writable persistence path for
  the v3 runtime.
- **Explicit hybrid runtime.** `HybridMemoryRuntime` and
  `SyncHybridMemoryRuntime` provide behaviorally aligned lexical and BinaryHDV
  retrieval over one explicit `AgentMemory` and one exact `MemoryScope`.
- **Deterministic retrieval.** Lexical and HDV candidates are unioned and
  ranked deterministically. The algorithm is identified as
  `hybrid-lexical-binary-hdv-v2`; telemetry records counts and scoring version,
  never memory content.
- **Safe cognitive writes.** Reconstructive or planned outputs are persisted
  only through validated atomic plans. Plans require confidence and provenance;
  persisted outputs are marked synthetic and retain validated source-memory
  identifiers.
- **Rebuildable derivatives.** Tier and graph projections derive from
  AgentMemory data and can be rebuilt deterministically; they are not separate
  memory stores.
- **Fail-closed HTTP composition.** `create_v3_app()` requires an injected
  `ScopeAuthorizer`. Each operation authorizes the complete scope before its
  database is opened. Without an authorizer, v3 memory routes are unavailable.

## Breaking changes

- `Memory(...)` is removed as an implicit-memory facade and always raises a
  v3 migration error.
- LiteEngine and `profile="lite"` are removed.
- New code must supply an explicit `MemoryScope`; cross-scope fallback is not
  supported.
- The legacy HAIM engine, REST surface, MCP surface, and CLI remain v2
  compatibility paths. They are not the v3 persistence boundary.

## Upgrade path

1. Open `AgentMemory` with the complete tenant, user, and agent scope.
2. Compose `HybridMemoryRuntime` or its synchronous facade over that client.
3. For HTTP deployment, compose `create_v3_app()` with a credential-to-scope
   authorizer owned by the deploying application.
4. Keep legacy HAIM persistence isolated while migrating; do not mix it with
   the v3 AgentMemory data path.

See [the v3 guide](docs/V3_HYBRID_RUNTIME.md) and the
[AgentMemory quickstart](docs/AGENT_MEMORY_QUICKSTART.md).

## Security and verification

- Scope checks are exact for remember, recall, history, projections, plans,
  and API routes.
- Plan source records are checked for exact ownership and active status inside
  the same SQLite transaction as the resulting write.
- Dependency audit reported no known vulnerabilities. The code scan reported
  no high-severity findings; its remaining medium findings are tracked legacy
  SQL-construction and network-binding review items.
- The release-candidate verification set passed 437 tests with 2 expected
  source-checkout metadata skips. Wheel and source distributions build, a
  clean wheel import succeeds, and the container smoke passes health, ready,
  and metrics checks.

## Known limits

- A deployment-specific credential-to-scope authorizer is required before
  exposing the v3 HTTP API.
- Graph jobs, reconstructive recall jobs, dream jobs, distributed ports,
  benchmark baselines, and v3 CLI support are staged follow-up work.
- The legacy compatibility lane remains separate from the v3 release claim.

## Publication gate

Before tagging `v3.0.0`, require the GitHub Actions matrix, package, security,
and container jobs to be green. Do not publish if a required CI job fails or
if the intended HTTP deployment lacks a complete-scope authorizer.
