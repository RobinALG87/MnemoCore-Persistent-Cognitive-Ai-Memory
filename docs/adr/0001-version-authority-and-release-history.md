# ADR 0001: Version authority and optional runtime surfaces

- Status: accepted
- Date: 2026-07-13

## Context

The repository contains historical beta tags while the current package metadata
declares version `2.0.0`. Duplicated literals in package metadata and runtime
modules can drift and make release artifacts difficult to identify.

The stable AgentMemory API is local-first and does not require server, vector,
or legacy-engine services. Those integrations still need to remain available
without becoming mandatory dependencies of the base wheel.

## Decision

- `src/mnemocore/_version.py` is the source version used by Hatch.
- Runtime code reads installed distribution metadata and falls back to that
  source value for an uninstalled checkout.
- The current version remains `2.0.0`; this ADR does not create a new release.
- Base installation has no third-party runtime dependencies.
- Server, CLI, and legacy integrations are explicit optional extras.
- Console scripts resolve to zero-argument callables with lazy optional imports.

## Consequences

Clean AgentMemory installs remain small and service-independent. Deployments
that use the REST server or legacy CLI must install the corresponding extra.
Future releases must update `_version.py`, create a matching tag, and verify
wheel metadata plus runtime `__version__` before publishing.

