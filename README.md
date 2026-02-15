# MnemoCore

> Infrastructure for Persistent Cognitive Memory.

[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Architecture](https://img.shields.io/badge/Memory-Binary%20VSA%2FHDC-blue.svg)]()

## ⚠️ Beta Notice (Read First)

MnemoCore is currently a **public beta / development preview**.

- No SLA, no uptime guarantees, no stability guarantees.
- No claims of production readiness.
- Features and APIs may change without backward compatibility.
- Results are experimental and should be independently validated.
- The software is provided "as is" under MIT (see `LICENSE`).

## What MnemoCore Is

MnemoCore is a memory engine for agentic systems built around **binary hyperdimensional vectors** and **tiered memory management**.

The project focuses on:
- Fast semantic memory encoding and retrieval.
- Memory lifecycle (HOT/WARM/COLD) with biologically inspired reinforcement/decay signals.
- API integration for external agent runtimes.

## Current Scope (Beta Reality)

What exists now:
- Binary HDV encoding and similarity search in the core engine.
- HOT tier in RAM plus WARM/COLD persistence paths.
- FastAPI service with async wrappers and Redis metadata integration.
- Qdrant integration with fallback behavior when unavailable.
- Unit tests for core components.

What is still evolving:
- Full cross-tier query coverage at scale.
- Consolidation and distributed workflows.
- Some advanced integrations marked TODO in code.

## Project Structure

- `src/core` – Core memory engine, vectors, tiering, synapses, config.
- `src/api` – FastAPI surface (`/store`, `/query`, `/memory/*`, `/health`, `/stats`).
- `src/subconscious` – Background processing loop.
- `src/nightlab` – Experimental orchestration and research tooling.
- `tests` – Unit/integration-focused test suite.
- `vector_core` – Experimental research tracks.
- `data` – Local runtime data for warm/cold persistence.

## Quick Start

### 1) Requirements
- Python 3.10+
- Docker + Docker Compose (for Redis/Qdrant)

### 2) Install

```bash
git clone https://github.com/RobinALG87/haim-public-v1.git
cd haim-public-v1
pip install -r requirements.txt
```

### 3) Start dependencies

```bash
docker-compose up -d
```

### 4) Start API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8100
```

### 5) Optional worker

```bash
python src/subconscious/daemon.py
```

## API Example

Store a memory:

```bash
curl -X POST http://localhost:8100/store \
    -H "Content-Type: application/json" \
    -d '{"content":"Stockholm is cold in winter","metadata":{"source":"demo"}}'
```

Query memories:

```bash
curl -X POST http://localhost:8100/query \
    -H "Content-Type: application/json" \
    -d '{"query":"winter in Sweden","top_k":5}'
```

Health check:

```bash
curl http://localhost:8100/health
```

## Configuration

Main runtime config is in `config.yaml` with `HAIM_*` environment variable overrides.
Compatibility note: env keys keep the `HAIM_*` prefix in this beta for backward compatibility.

Common examples:
- `HAIM_DIMENSIONALITY`
- `HAIM_ENCODING_MODE`
- `HAIM_REDIS_URL`
- `HAIM_QDRANT_URL`
- `HAIM_LOG_LEVEL`

See `src/core/config.py` for all settings and defaults.

## Testing

```bash
python -m pytest tests -v
```

## Documentation Map

- `docs/BETA_POLICY.md` – Beta constraints and expectation management.
- `docs/ARCHITECTURE.md` – Practical architecture overview.
- `docs/API.md` – Endpoint reference and examples.
- `docs/ROADMAP.md` – Known limitations and planned direction.
- `SECURITY.md` – Vulnerability reporting and disclosure policy.
- `RELEASE_CHECKLIST.md` – Safe release workflow for this repo.
- `studycase.md` – Background and design context.
- `HAIM Phase 3 5 Infinite.md` – Long-form scaling blueprint.

## Security and Responsible Use

- Do not store secrets or regulated data without your own controls.
- Validate outputs before decision-making in critical workflows.
- If you discover vulnerabilities, report privately before public disclosure at Robin@veristatesystems.com.

## License

This project is released under the MIT License.
See `LICENSE`.
