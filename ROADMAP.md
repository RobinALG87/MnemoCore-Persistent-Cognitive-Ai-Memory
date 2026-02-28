# MnemoCore Roadmap

**Open Source Infrastructure for Persistent Cognitive Memory**

Version: 5.0.0 | Updated: 2026-03-01

---

## Vision

MnemoCore provides the foundational memory layer for cognitive AI systems —
a production-ready, self-hosted alternative to cloud-dependent memory solutions.

---

## Current Status (v5.0.0)

| Component | Status |
|-----------|--------|
| Binary HDV Engine (16384-dim VSA) | ✅ Stable |
| Tiered Storage (HOT/WARM/COLD) | ✅ Production |
| HNSW Index | ✅ Working |
| Store/Query/Feedback API | ✅ Operational |
| Qdrant Integration | ✅ Available |
| MCP Server | ✅ Functional |
| Cognitive Services (Phase 5) | ✅ Complete |
| Research Services (Phase 6) | ✅ Complete |
| PyPI Distribution | ✅ Published |
| Test Suite | ✅ 1291+ tests |

---

## Completed Phases

### Phase 3: Core Architecture ✅
- Binary HDV core (XOR bind / bundle / permute / Hamming)
- Three-tier HOT/WARM/COLD memory lifecycle with LTP-driven eviction
- Synaptic connections with Hebbian learning

### Phase 4.0–4.5: Cognitive Enhancements ✅
- XOR attention masking for project isolation
- Bayesian reliability feedback loop
- Semantic consolidation (dream-phase synthesis)
- Auto-associative cleanup (vector immunology)
- Knowledge gap detection and filling
- Episodic chaining + chrono-weighted temporal recall
- Subconscious daemon with LLM-powered dream synthesis
- Dependency-injection container pattern
- HNSW index, batch operations, meta-cognition layer

### Phase 5.0–5.4: Cognitive Memory Architecture ✅
- **Working Memory** — Active slot buffer (7±2 items, TTL-based pruning)
- **Episodic Store** — Temporal chain verification/repair, LTP calculation, agent-scoped history
- **Semantic Store** — Qdrant persistence, CLS-style consolidation (hippocampus→neocortex)
- **Procedural Store** — JSON persistence, word-overlap matching, outcome tracking
- **Meta Memory** — Anomaly detection, LLM-driven proposals
- **Self-Improvement Worker** — Phase 0 dry-run with 5 validation gates
- **Pulse Loop** — 7 cognitive phases fully implemented
- **65 tests** for cognitive services

### Phase 5 Hardening ✅
- 136 additional tests (self-improvement, pulse phases, store integration)
- Config maintainability review (§1–§9 section organization)
- Subconscious exports verification

### Phase 6: Research-Backed Cognitive Services ✅
- **StrategyBank** — 5-phase closed loop (Evaluate→Select→Apply→Distill→Store), Bayesian confidence, 60/40 balance
- **KnowledgeGraph** — Bidirectional edges, spreading activation, community detection, decay
- **MemoryScheduler** — Priority queue, memory interrupts, load shedding, health scoring (STM→MTM→LTM)
- **SAMEP** — Multi-agent memory exchange, HMAC integrity, access control, tier-based visibility
- **Pulse Extended** — 4 new research phases (strategy, graph, scheduler, exchange)
- **85 tests** for Phase 6 services

---

## Phase 7: Production Hardening (Planned)

**Goal:** Enterprise-ready deployment and operational excellence

### 7.1 Reliability & Scale
- [ ] Distributed/clustered HOT-tier coordination
- [ ] CUDA kernels for batch HDV operations
- [ ] Extended observability (`mnemocore_*` metric prefix everywhere)
- [ ] Helm chart production hardening (autoscaling, PDB)
- [ ] Chaos engineering tests (network failures, disk full)

### 7.2 Self-Improvement Phase 1
- [ ] Enable writes after dry-run observation period
- [ ] Staged rollout with rollback capability
- [ ] Human-in-the-loop approval for high-impact proposals

### 7.3 Developer Experience
- [ ] Complete OpenAPI spec documentation
- [ ] Jupyter notebook tutorials
- [ ] Quickstart guide for common patterns

**ETA:** 4–6 weeks

---

## Phase 8: Feature Expansion (Planned)

### 8.1 Advanced Retrieval
- [ ] Multi-hop associative recall
- [ ] Contextual ranking (personalized relevance)
- [ ] Negation queries ("NOT about project X")

### 8.2 Multi-Modal Support
- [ ] Image embedding storage (CLIP encoder)
- [ ] Audio transcript indexing (Whisper)
- [ ] Cross-modal VSA binding

### 8.3 Emotional/Affective Layer
- [ ] Valence/arousal tagging
- [ ] Emotion-weighted LTP decay
- [ ] Flashbulb memory formation

**ETA:** 8–12 weeks

---

## Phase 9: Ecosystem (Planned)

### 9.1 Integrations
- [ ] LangChain memory adapter
- [ ] LlamaIndex vector store
- [ ] CrewAI shared memory backend
- [ ] AutoGen conversation memory

### 9.2 SDKs
- [ ] TypeScript/JavaScript SDK
- [ ] Go SDK

### 9.3 Community
- [ ] Contributing guide
- [ ] Feature request process
- [ ] Regular release cadence

**ETA:** 12–16 weeks

---

## Release History

| Version | Date | Focus |
|---------|------|-------|
| v3.x | 2025 | Core architecture (Binary HDV, 3-tier, LTP) |
| v4.0 | 2026-01 | Cognitive enhancements (XOR attention, Bayesian LTP, gaps) |
| v4.3 | 2026-02 | Temporal recall (episodic chaining, chrono-weighting) |
| v4.5 | 2026-02 | Subconscious daemon, container DI, HNSW, batch ops |
| **v5.0** | **2026-03** | **Cognitive architecture (Phase 5+6), 1291 tests** |

---

## Contributing

MnemoCore is open source under MIT license.

- **GitHub:** https://github.com/RobinALG87/MnemoCore-Infrastructure-for-Persistent-Cognitive-Memory
- **PyPI:** `pip install mnemocore`
- **Issues:** Use GitHub Issues for bugs and feature requests
- **PRs:** Welcome — performance, algorithms, integrations, tests

---

*Roadmap maintained by Robin Granberg & Omega*
