# MnemoCore 3.0.0 release notes

**Release class:** public beta. This release is suitable for local evaluation
and single-node deployments; it is not a claim of multi-node or production
recovery readiness.

## Highlights

- AgentMemory is the sole durable store for the new scoped hybrid runtime.
- `HybridMemoryRuntime` provides deterministic lexical and BinaryHDV retrieval
  with scoring version `hybrid-lexical-binary-hdv-v2`.
- Cognitive outputs are applied only through validated, atomic plans with
  confidence, provenance, source-memory validation, and synthetic markers.
- `create_v3_app()` is a standalone, fail-closed API composition root. It
  requires an injected `ScopeAuthorizer` for every complete `MemoryScope`.

## Breaking changes

- `Memory(...)`, including every former profile, now raises a migration error.
- LiteEngine and `profile="lite"` are removed.
- Legacy HAIM, REST, MCP, and CLI surfaces remain v2 compatibility paths. They
  are not the scoped v3 persistence boundary.

## Upgrade

Use `AgentMemory` with an explicit `MemoryScope`, then compose
`HybridMemoryRuntime` (or `create_v3_app()` with a credential-to-scope
authorizer). See [the v3 guide](docs/V3_HYBRID_RUNTIME.md) and
[AgentMemory quickstart](docs/AGENT_MEMORY_QUICKSTART.md).

## Publication gate

Before tagging `v3.0.0`, run the required GitHub Actions matrix and confirm
the security, package, Docker, and supported-runtime gates are green. The
legacy lane remains visible as compatibility debt and is not a v3 persistence
claim. Do not publish the release if the scope authorizer deployment is absent
or any required CI job fails.
