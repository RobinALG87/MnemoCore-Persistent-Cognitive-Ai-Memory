# ADR 0001: Version authority and optional runtime surfaces

- Status: accepted
- Date: 2026-07-13

## Context

The repository contains historical beta tags while the current package metadata
declares version `2.0.0`. Duplicated literals in package metadata and runtime
modules can drift and make release artifacts difficult to identify.

The stable AgentMemory API is local-first and does not require a running
server, vector store, or legacy-engine service. Existing 2.x installations,
however, may import HAIM/REST functionality immediately after
`pip install mnemocore`.

## Decision

- `src/mnemocore/_version.py` is the source version used by Hatch.
- Runtime code reads installed distribution metadata and falls back to that
  source value for an uninstalled checkout.
- The current version remains `2.0.0`; this ADR does not create a new release.
- The 2.x base installation retains its established runtime dependencies for
  backward compatibility.
- Server, CLI, and legacy extras remain explicit installation profiles for
  deployments that want their dependency intent documented.
- A dependency-minimal AgentMemory distribution is deferred to a future major
  release with a migration guide and clean-install compatibility matrix.
- Console scripts resolve to zero-argument callables with lazy optional imports.

## Consequences

AgentMemory remains service-independent at runtime, while existing HAIM/REST
installations remain compatible in 2.x. Future releases must update
`_version.py`, create a matching tag, and verify wheel metadata plus runtime
`__version__` before publishing. The future dependency split must not ship
without its migration guide and compatibility tests.
