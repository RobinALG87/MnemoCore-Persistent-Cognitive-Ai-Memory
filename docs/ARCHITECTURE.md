# MnemoCore Architecture (Beta)

## Beta Context

This document describes the current implementation direction in beta.
It is not a guarantee of final architecture, performance, or feature completeness.

## Core Components

- `src/core/engine.py`: Main orchestration for memory storage, encoding, query, and synaptic augmentation.
- `src/core/binary_hdv.py`: Binary hyperdimensional vector operations.
- `src/core/tier_manager.py`: HOT/WARM/COLD placement and movement logic.
- `src/core/config.py`: Typed config loading from YAML + env overrides.
- `src/core/async_storage.py`: Async Redis metadata operations.
- `src/api/main.py`: FastAPI interface.

## Memory Model

MnemoCore represents memory as high-dimensional vectors and metadata-rich nodes:

1. Encode input text into vector representation.
2. Store node in HOT tier initially.
3. Apply reinforcement/decay dynamics (LTP-related logic).
4. Move between tiers based on thresholds and access patterns.

## Tiering Model

- **HOT**: In-memory dictionary for fastest access.
- **WARM**: Qdrant-backed where available; filesystem fallback when unavailable.
- **COLD**: Filesystem archival path for long-lived storage.

## Query Flow (Current Beta)

Current query behavior prioritizes HOT tier recall and synaptic score augmentation.
Cross-tier retrieval is still evolving and should be treated as beta behavior.

## Async + External Services

- Redis is used for async metadata and event stream operations.
- API startup checks Redis health and can operate in degraded mode.
- Qdrant usage is enabled through tier manager and can fall back to local files.

## Observability

- Prometheus metrics endpoint mounted at `/metrics` in API server.
- Logging behavior controlled through config.

## Practical Limitations

- Some roadmap functionality remains TODO-marked in code.
- Interface contracts may change across beta releases.
- Performance can vary significantly by hardware and data profile.

For active limitations and next work items, see `docs/ROADMAP.md`.
