---
title: MnemoCore
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
language:
  - en
tags:
  - memory
  - vector-database
  - cognitive-architecture
  - agents
  - hyperdimensional-computing
  - vsa
---

# MnemoCore

### Infrastructure for Persistent Cognitive Memory

> *"Memory is not a container. It is a living process — a holographic continuum where every fragment contains the whole."*

<p align="center">
  <img src="https://img.shields.io/badge/Status-Release%202.0.0-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Async%20Ready-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/HDV-16384--dim-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Vectors-Binary%20VSA-critical?style=for-the-badge" />
</p>

---

## Quick Install

**Install from the public repository:**

```bash
pip install git+https://github.com/RobinALG87/MnemoCore-Persistent-Cognitive-Ai-Memory.git@main
```

**Install from source (development):**

```bash
git clone https://github.com/RobinALG87/MnemoCore-Persistent-Cognitive-Ai-Memory.git
cd MnemoCore-Persistent-Cognitive-Ai-Memory
python -m venv .venv && .\.venv\Scripts\activate   # Windows
# source .venv/bin/activate                        # Linux / macOS
pip install -e .          # editable install
pip install -e ".[dev]"   # + pytest, mypy, black, etc.
```

> **For the REST server, set your API key before starting:**
> ```bash
> # Windows PowerShell
> $env:HAIM_API_KEY = "your-secure-key"
> # Linux / macOS
> # export HAIM_API_KEY="your-secure-key"
> ```
> Then start the legacy v2 API locally: `mnemocore`.

`mnemocore` starts the legacy v2 REST server; `mnemocore-cli` is the legacy
Click CLI. Neither is the v3 AgentMemory persistence boundary. For v3, compose
`HybridMemoryRuntime` over an explicit scope or deploy `create_v3_app` with a
credential-to-scope authorizer. AgentMemory itself is local-first and does not
require a running Redis or Qdrant service. Full legacy setup, Docker, and
configuration details are in [Installation](#installation) below.

---

## What is MnemoCore?

**MnemoCore** is persistent memory and context infrastructure for agents.
Its stable product track is **AgentMemory**: local-first, scope-isolated,
bitemporal memory with receipts, timeline queries, deterministic context
compilation, and rebuildable SQLite projections. The **unreleased** v3
hybrid-runtime milestone builds on that store without adding a second
persistence path.
`HybridMemoryRuntime` adds deterministic lexical/HDV retrieval and safe
cognitive projections over AgentMemory. The legacy HAIM stack remains available
only as a temporary, deprecated compatibility surface; it is not the production
baseline.

The current 2.x package keeps its established dependencies so existing HAIM
and REST users are not broken. A dependency-minimal AgentMemory distribution
will arrive only in a future major release with a migration guide.

Traditional vector stores retrieve. MnemoCore **thinks**. It is built on the mathematical framework of **Binary Hyperdimensional Computing (HDC)** and **Vector Symbolic Architectures (VSA)**, principles rooted in Pentti Kanerva's landmark 2009 theory of cognitive computing. Every memory is encoded as a **16,384-dimensional binary holographic vector** — a format that is simultaneously compact (2,048 bytes), noise-tolerant (Hamming geometry), and algebraically rich (XOR binding, majority bundling, circular permutation).

At its core lives the **Holographic Active Inference Memory (HAIM) Engine** — a system that does not merely answer queries, but:

- **Evaluates** the epistemic novelty of every incoming memory before deciding to store it
- **Dreams** — strengthening synaptic connections between related memories during idle cycles
- **Reasons by analogy** — if `king:man :: ?:woman`, the VSA soul computes `queen`
- **Self-organizes** into tiered storage based on biologically-inspired Long-Term Potentiation (LTP)
- **Scales** from a single process to distributed nodes targeting 1B+ memories

Phase 4.x introduces cognitive enhancements including contextual masking, reliability feedback loops, semantic consolidation, gap detection/filling, temporal recall (episodic chaining + chrono-weighted query), a Subconscious Daemon with LLM-powered dream synthesis, and a full dependency-injection container pattern for clean modularity.

Phase 5 transforms MnemoCore into a true cognitive architecture with Working Memory, Episodic/Semantic/Procedural stores, Meta Memory, and a Self-Improvement Worker — all orchestrated by an 11-phase Pulse loop.

Phase 6 adds four research-backed services: **StrategyBank** (reasoning bank with Bayesian confidence), **KnowledgeGraph** (spreading activation + community detection), **MemoryScheduler** (OS-level priority queue with interrupts), and **SAMEP** (multi-agent memory exchange with HMAC integrity).

---

## Table of Contents

- [Architecture](#architecture)
- [Core Technology](#core-technology-binary-hdv--vsa)
- [The Memory Lifecycle](#the-memory-lifecycle)
- [Tiered Storage](#tiered-storage-hotwarmcold)
- [Phase 4.0 Cognitive Enhancements](#phase-40-cognitive-enhancements)
- [Phase 4.4–4.5 Subconscious Daemon & LLM Integration](#phase-4445-subconscious-daemon--llm-integration)
- [API Reference](#api-reference)
- [Python Library Usage](#python-library-usage)
- [Installation](#installation)
- [Configuration](#configuration)
- [MCP Server Integration](#mcp-server-integration)
- [Observability](#observability)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MnemoCore Stack                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              REST API  (FastAPI / Async)                  │  │
│   │   /store  /query  /feedback  /insights/gaps  /stats      │  │
│   │   Rate Limiting · API Key Auth · Prometheus Metrics      │  │
│   └─────────────────────────┬────────────────────────────────┘  │
│                             │                                   │
│   ┌─────────────────────────▼────────────────────────────────┐  │
│   │                  HAIM Engine                              │  │
│   │                                                          │  │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│   │   │ Text Encoder │  │  EIG / Epist │  │  Subconsc.   │  │  │
│   │   │ (token→HDV)  │  │  Drive       │  │  Dream Loop  │  │  │
│   │   └──────────────┘  └──────────────┘  └──────────────┘  │  │
│   │                                                          │  │
│   │   ┌──────────────────────────────────────────────────┐  │  │
│   │   │            Binary HDV Core (VSA)                 │  │  │
│   │   │  XOR bind · majority_bundle · permute · Hamming  │  │  │
│   │   └──────────────────────────────────────────────────┘  │  │
│   └─────────────────────────┬────────────────────────────────┘  │
│                             │                                   │
│   ┌─────────────────────────▼────────────────────────────────┐  │
│   │                  Tier Manager                             │  │
│   │                                                          │  │
│   │   🔥 HOT         🌡 WARM            ❄️ COLD               │  │
│   │   In-Memory      Redis / mmap       Qdrant / Disk / S3   │  │
│   │   ≤2,000 nodes   ≤100,000 nodes     ∞ nodes              │  │
│   │   <1ms           <10ms              <100ms               │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  Conceptual Layer ("The Soul")           │   │
│   │   ConceptualMemory · Analogy Engine · Symbol Algebra     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | File | Responsibility |
|-----------|------|----------------|
| **HAIM Engine** | `src/mnemocore/core/engine.py` | Central cognitive coordinator — store, query, dream, delete |
| **BinaryHDV** | `src/mnemocore/core/binary_hdv.py` | 16384-dim binary vector math (XOR, Hamming, bundle, permute) |
| **TextEncoder** | `src/mnemocore/core/binary_hdv.py` | Token→HDV pipeline with positional permutation binding |
| **MemoryNode** | `src/mnemocore/core/node.py` | Memory unit with LTP, epistemic values, tier state |
| **TierManager** | `src/mnemocore/core/tier_manager.py` | HOT/WARM/COLD orchestration with LTP-driven eviction |
| **SynapticConnection** | `src/mnemocore/core/synapse.py` | Hebbian synapse with strength, decay, and fire tracking |
| **SynapseIndex** | `src/mnemocore/core/synapse_index.py` | Fast synapse lookup index for associative spreading |
| **ConceptualMemory** | `src/mnemocore/core/holographic.py` | VSA soul for analogy and cross-domain symbolic reasoning |
| **AsyncRedisStorage** | `src/mnemocore/core/async_storage.py` | Async Redis backend (WARM tier + pub/sub) |
| **BayesianLTP** | `src/mnemocore/core/bayesian_ltp.py` | Bayesian reliability scoring on top of LTP strength |
| **SemanticConsolidation** | `src/mnemocore/core/semantic_consolidation.py` | Memory deduplication via majority-bundle prototyping |
| **ConsolidationWorker** | `src/mnemocore/core/consolidation_worker.py` | Async worker scheduling nightly consolidation |
| **GapDetector** | `src/mnemocore/core/gap_detector.py` | Temporal co-occurrence analysis for knowledge gaps |
| **GapFiller** | `src/mnemocore/core/gap_filler.py` | Bridge detected gaps via synapse creation |
| **Immunology** | `src/mnemocore/core/immunology.py` | Auto-associative attractor cleanup for vector drift |
| **Attention** | `src/mnemocore/core/attention.py` | XOR context masking / project isolation |
| **BatchOps** | `src/mnemocore/core/batch_ops.py` | Vectorized bulk store / query operations |
| **HNSWIndex** | `src/mnemocore/core/hnsw_index.py` | In-process HNSW approximate nearest-neighbour index |
| **QdrantStore** | `src/mnemocore/core/qdrant_store.py` | Async Qdrant COLD tier backend |
| **RecursiveSynthesizer** | `src/mnemocore/core/recursive_synthesizer.py` | Deep concept synthesis via iterative VSA composition |
| **RippleContext** | `src/mnemocore/core/ripple_context.py` | Cascading context propagation across synaptic graph |
| **SubconsciousAI** | `src/mnemocore/core/subconscious_ai.py` | LLM-guided dream synthesis worker |
| **SubconsciousDaemon** | `src/mnemocore/subconscious/daemon.py` | Background process orchestrating dream/consolidation cycles |
| **LLMIntegration** | `src/mnemocore/llm_integration.py` | Agent-facing LLM connector (OpenAI / Anthropic compatible) |
| **Container** | `src/mnemocore/core/container.py` | Dependency-injection wiring for all core components |
| **GoalTree** | `src/mnemocore/meta/goal_tree.py` | Hierarchical goal / task tracking for meta-cognition |
| **LearningJournal** | `src/mnemocore/meta/learning_journal.py` | Persistent log of what the agent has learned over time |
| **API** | `src/mnemocore/api/main.py` | Legacy v2 HAIM REST interface; v3 uses `create_v3_app` with scope authorization |
| **MCP Server** | `src/mnemocore/mcp/server.py` | Legacy v2 HAIM REST adapter; v3 uses scoped AgentMemory integration helpers |

---

## Core Technology: Binary HDV & VSA

MnemoCore's mathematical foundation is **Hyperdimensional Computing** — a computing paradigm that encodes information in very high-dimensional binary vectors (HDVs), enabling noise-tolerant, distributed, and algebraically composable representations.

### The Vector Space

Every piece of information — a word, a sentence, a concept, a goal — is encoded as a **16,384-dimensional binary vector**:

```
Dimension D = 16,384 bits = 2,048 bytes per vector
Storage:      packed as numpy uint8 arrays
Similarity:   Hamming distance (popcount of XOR result)
Random pair:  ~50% similarity (orthogonality by probability)
```

At this dimensionality, two random vectors will differ in ~50% of bits. This near-orthogonality is the foundation of the system's expressive power — related concepts cluster together while unrelated ones remain maximally distant.

### VSA Algebra

Four primitive operations make the entire system work:

#### Binding — XOR `⊕`
Creates an association between two concepts. Crucially, the result is **dissimilar to both inputs** (appears as noise), making it a true compositional operation.

```python
# Bind content to its context
bound = content_vec.xor_bind(context_vec)  # content ⊕ context

# Self-inverse: unbind by re-binding
recovered = bound.xor_bind(context_vec)   # ≈ content (XOR cancels)
```

Key mathematical properties:
- **Self-inverse**: `A ⊕ A = 0` (XOR cancels itself)
- **Commutative**: `A ⊕ B = B ⊕ A`
- **Distance-preserving**: `hamming(A⊕C, B⊕C) = hamming(A, B)`

#### Bundling — Majority Vote
Creates a **prototype** that is similar to all inputs. This is how multiple memories combine into a concept.

```python
from mnemocore.core.binary_hdv import majority_bundle

# Create semantic prototype from related memories
concept = majority_bundle([vec_a, vec_b, vec_c, vec_d])  # similar to all inputs
```

#### Permutation — Circular Shift
Encodes **sequence and roles** without separate positional embeddings.

```python
# Positional encoding: token at position i
positioned = token_vec.permute(shift=i)  # circular bit-shift

# Encode "hello world" with order information
hello_positioned = encoder.get_token_vector("hello").permute(0)
world_positioned = encoder.get_token_vector("world").permute(1)
sentence_vec = majority_bundle([hello_positioned, world_positioned])
```

#### Similarity — Hamming Distance
Fast comparison using vectorized popcount over XOR results:

```python
# Normalized similarity: 1.0 = identical, 0.5 = unrelated
sim = vec_a.similarity(vec_b)  # 1.0 - hamming(a, b) / D

# Batch nearest-neighbor search (no Python loops)
distances = batch_hamming_distance(query, database_matrix)
```

### Text Encoding Pipeline

The `TextEncoder` converts natural language to HDVs using a token-position binding scheme:

```
"Python TypeError" →
  token_hdv("python") ⊕ permute(0)  =  positioned_0
  token_hdv("typeerror") ⊕ permute(1)  =  positioned_1
  majority_bundle([positioned_0, positioned_1])  =  final_hdv
```

Token vectors are **deterministic** — seeded via SHAKE-256 hash — meaning the same word always produces the same base vector, enabling cross-session consistency without a vocabulary file.

---

## The Memory Lifecycle

Every memory passes through a defined lifecycle from ingestion to long-term storage:

```
Incoming Content
      │
      ▼
 ┌─────────────┐
 │ TextEncoder │  → 16,384-dim binary HDV
 └──────┬──────┘
        │
        ▼
 ┌──────────────────┐
 │ Context Binding  │  → XOR bind with goal_context if present
 │  (XOR)           │     bound_vec = content ⊕ context
 └──────┬───────────┘
        │
        ▼
 ┌──────────────────┐
 │  EIG Evaluation  │  → Epistemic Information Gain
 │  (Novelty Check) │     eig = normalized_distance(vec, context_vec)
 └──────┬───────────┘     tag "epistemic_high" if eig > threshold
        │
        ▼
 ┌─────────────────┐
 │  MemoryNode     │  → id, hdv, content, metadata
 │  Creation       │     ltp_strength = I × log(1+A) × e^(-λT)
 └──────┬──────────┘
        │
        ▼
 ┌─────────────────┐
 │  HOT Tier       │  → In-memory dict (max 2000 nodes)
 │  (RAM)          │     LTP eviction: low-LTP nodes → WARM
 └──────┬──────────┘
        │     (background)
        ▼
 ┌─────────────────┐
 │ Subconscious    │  → Dream cycle fires
 │ Dream Loop      │     Query similar memories
 └──────┬──────────┘     Strengthen synapses (Hebbian)
        │
        ▼
 ┌─────────────────┐
 │  WARM Tier      │  → Redis-backed persistence
 │  (Redis/mmap)   │     async dual-write + pub/sub events
 └──────┬──────────┘
        │     (scheduled, nightly)
        ▼
 ┌─────────────────┐
 │  COLD Tier      │  → Qdrant / Disk / S3
 │  (Archival)     │     ANN search, long-term persistence
 └─────────────────┘
```

### Long-Term Potentiation (LTP)

Memories are not equal. Importance is computed dynamically using a biologically-inspired LTP formula:

```
S = I × log(1 + A) × e^(-λ × T)

Where:
  S = LTP strength (determines tier placement)
  I = Importance (derived from epistemic + pragmatic value)
  A = Access count (frequency of retrieval)
  λ = Decay lambda (configurable, default ~0.01)
  T = Age in days
```

Memories with high LTP remain in HOT tier. Those that decay are automatically promoted to WARM, then COLD — mirroring how biological memory consolidates from working memory to long-term storage.

### Synaptic Connections

Memories are linked by `SynapticConnection` objects that implement Hebbian learning: *"neurons that fire together, wire together."*

Every time two memories are co-retrieved (via the background dream loop or explicit binding), their synaptic strength increases. During query time, synaptic spreading amplifies scores of connected memories even when they do not directly match the query vector — enabling **associative recall**.

```python
# Explicit synapse creation
engine.bind_memories(id_a, id_b, success=True)

# Associative spreading: query top seeds spread activation to neighbors
# neighbor_score += seed_score × synapse_strength × 0.3
```

---

## Tiered Storage: HOT / WARM / COLD

| Tier | Backend | Capacity | Latency | Eviction Trigger |
|------|---------|----------|---------|------------------|
| 🔥 **HOT** | Python dict (RAM) | 2,000 nodes | < 1ms | LTP < threshold |
| 🌡 **WARM** | Redis + mmap | 100,000 nodes | < 10ms | Age + low access |
| ❄️ **COLD** | Qdrant / Disk / S3 | Unlimited | < 100ms | Manual / scheduled |

Promotion is automatic: accessing a WARM or COLD memory re-promotes it to HOT based on recalculated LTP. Eviction is LRU-weighted by LTP strength — the most biologically active memories always stay hot.

---

## Phase 4.0 Cognitive Enhancements

MnemoCore Phase 4.0 introduces five architectural enhancements that elevate the system from **data retrieval** to **cognitive reasoning**.

---

### 1. Contextual Query Masking *(XOR Attention)*

**Problem**: Large multi-project deployments suffer from cross-context interference. A query for `"Python error handling"` returns memories from all projects equally, diluting precision.

**Solution**: Bidirectional XOR context binding — apply the same context vector at both **storage** and **query** time:

```
Store:  bound_vec   = content ⊕ context_vec
Query:  masked_query = query   ⊕ context_vec

Result: (content ⊕ C) · (query ⊕ C) ≈ content · query
        (context cancels, cross-project noise is suppressed)
```

```python
# Store memories in a project context
engine.store("API rate limiting logic", goal_id="ProjectAlpha")
engine.store("Garden watering schedule", goal_id="HomeProject")

# Query with context mask — only ProjectAlpha memories surface
results = engine.query("API logic", top_k=5, context="ProjectAlpha")
```

**Expected impact**: +50–80% query precision (P@5) in multi-project deployments.

---

### 2. Reliability Feedback Loop *(Self-Correcting Memory)*

**Problem**: Wrong or outdated memories persist with the same retrieval weight as correct ones. The system has no mechanism to learn from its own mistakes.

**Solution**: Bayesian reliability scoring with real-world outcome feedback:

```
reliability = (successes + 1) / (successes + failures + 2)  # Laplace smoothing

LTP_enhanced = I × log(1+A) × e^(-λT) × reliability
```

```python
# After using a retrieved memory:
engine.provide_feedback(memory_id, outcome=True)   # Worked → boost reliability
engine.provide_feedback(memory_id, outcome=False)  # Failed → reduce reliability

# System auto-tags consistently wrong memories as "unreliable"
# and verified memories (>5 successes, >0.8 score) as "verified"
```

The system converges toward **high-confidence knowledge** — memories that have demonstrably worked in practice rank above theoretically similar but unproven ones.

---

### 3. Semantic Memory Consolidation *(Dream-Phase Synthesis)*

**Problem**: Episodic memory grows without bound. 1,000 memories about `"Python TypeError"` are semantically equivalent but consume 2MB of vector space and slow down linear scan queries.

**Solution**: Nightly `ConsolidationWorker` clusters similar WARM tier memories and replaces them with a **semantic anchor** — a majority-bundled prototype:

```
BEFORE consolidation:
  mem_001: "Python TypeError in line 45"    (2KB vector)
  mem_002: "TypeError calling function"     (2KB vector)
  ...   ×100 similar memories              (200KB total)

AFTER consolidation:
  anchor_001: "Semantic pattern: python typeerror function"
              metadata: {source_count: 100, confidence: 0.94}
              hdv: majority_bundle([mem_001.hdv, ..., mem_100.hdv])  (2KB)
```

```python
# Manual trigger (runs automatically at 3 AM)
stats = engine.trigger_consolidation()
# → {"abstractions_created": 12, "memories_consolidated": 847}

# Via API (admin endpoint)
POST /admin/consolidate
```

**Expected impact**: 70–90% memory footprint reduction, 10x query speedup at scale.

---

### 4. Auto-Associative Cleanup Loop *(Vector Immunology)*

**Problem**: Holographic vectors degrade over time through repeated XOR operations, noise accumulation, and long-term storage drift. After months of operation, retrieved vectors become "blurry" and similarity scores fall.

**Solution**: Iterative attractor dynamics — when a retrieved vector appears noisy, snap it to the nearest stable concept in a **codebook** of high-confidence prototypes:

```
noisy_vec → find K nearest in codebook
         → majority_bundle(K neighbors)
         → check convergence (Hamming distance < 5%)
         → iterate until converged or max iterations reached
```

```python
# Cleanup runs automatically on retrieval when noise > 15%
node = engine.get_memory(memory_id, auto_cleanup=True)
# node.metadata["cleaned"] = True  (if cleanup was triggered)
# node.metadata["cleanup_iterations"] = 3

# Codebook is auto-populated from most-accessed, high-reliability memories
```

**Expected impact**: Maintain >95% similarity fidelity even after years of operation.

---

### 5. Knowledge Gap Detection *(Proactive Curiosity)*

**Problem**: The system is entirely reactive — it answers queries but never identifies what it *doesn't know*. True cognitive autonomy requires self-directed learning.

**Solution**: Temporal co-occurrence analysis — detect concepts that are frequently accessed **close in time** but have **no synaptic connection**, flagging them as knowledge gaps:

```python
# Automatically runs hourly
gaps = engine.detect_knowledge_gaps(time_window_seconds=300)

# Returns structured insight:
# [
#   {
#     "concept_a": "Python asyncio event loop",
#     "concept_b": "FastAPI dependency injection",
#     "suggested_query": "How does asyncio relate to FastAPI dependency injection?",
#     "co_occurrence_count": 4
#   }
# ]

# Query endpoint
GET /insights/gaps?lookback_hours=24

# Fill gap manually (or via LLM agent)
POST /insights/fill-gap
{"concept_a_id": "mem_xxx", "concept_b_id": "mem_yyy",
 "explanation": "FastAPI uses asyncio's event loop internally..."}
```

The system becomes capable of **saying what it doesn't understand** and requesting clarification — the first step toward genuine cognitive autonomy.

---

## Phase 4.4–4.5: Subconscious Daemon & LLM Integration

### Subconscious Daemon *(Autonomous Background Mind)*

Phase 4.4 introduced `SubconsciousAI` — a worker that fires during idle cycles and calls an external LLM to generate **synthetic dream memories**: structured insights derived by reasoning over existing memory clusters, rather than through direct observation.

Phase 4.5 hardened this into a full `SubconsciousDaemon` — an independently managed asyncio process that orchestrates dream cycles, consolidation scheduling, and subconscious queue processing:

```python
# The daemon is started automatically when the API starts up.
# It coordinates:
#   - Dream synthesis: SubconsciousAI → LLM → synthetic insights stored back
#   - Consolidation scheduling: ConsolidationWorker fired on a configurable interval
#   - Subconscious queue: novelty detection from Redis pub/sub stream
```

Configure in `config.yaml`:

```yaml
haim:
  subconscious_ai:
    enabled: true
    api_url: "https://api.openai.com/v1/chat/completions"  # or Anthropic
    model: "gpt-4o-mini"
    # api_key: set via SUBCONSCIOUS_AI_API_KEY env var
    dream_interval_seconds: 300
    batch_size: 5
```

### Dependency Injection Container

All major services (TierManager, AsyncRedisStorage, QdrantStore, SubconsciousAI, etc.) are now wired through `src/mnemocore/core/container.py`. This eliminates global singleton state and makes every subsystem testable in isolation:

```python
from mnemocore.core.container import build_container

container = build_container(config)
engine   = container.engine()
tier_mgr = container.tier_manager()
```

### LLM Agent Integration

`src/mnemocore/llm_integration.py` provides a high-level interface for attaching MnemoCore to any OpenAI/Anthropic-style LLM agent loop:

```python
from mnemocore.llm_integration import MnemoCoreAgent

agent = MnemoCoreAgent(engine)

# Store agent observations
agent.observe("User prefers concise answers over verbose ones")

# Recall relevant context before a response
context = agent.recall("user preference", top_k=3)
```

---

## API Reference (v2 compatibility)

This section documents the legacy HAIM REST service (`mnemocore.api.main:app`).
It remains available for v2 clients but its global JSONL/tiering lifecycle is
not AgentMemory-backed and must not be presented as v3 persistence. For v3,
use `create_v3_app(sqlite_path, scope_authorizer=...)`; the authorizer must
approve the caller's complete requested scope before each operation.

### Authentication

All endpoints require an API key via the `X-API-Key` header:

```bash
export HAIM_API_KEY="your-secure-key"
curl -H "X-API-Key: $HAIM_API_KEY" ...
```

### Endpoints

#### `POST /store`
Store a new memory with optional context binding.

```json
Request:
{
  "content": "FastAPI uses Pydantic v2 for request validation.",
  "metadata": {"source": "docs", "tags": ["python", "fastapi"]},
  "context": "ProjectAlpha",
  "agent_id": "agent-001",
  "ttl": 3600
}

Response:
{
  "ok": true,
  "memory_id": "mem_1739821234567",
  "message": "Stored memory: mem_1739821234567"
}
```

#### `POST /query`
Query memories by semantic similarity with optional context masking.

```json
Request:
{
  "query": "How does FastAPI handle request validation?",
  "top_k": 5,
  "context": "ProjectAlpha"
}

Response:
{
  "ok": true,
  "query": "How does FastAPI handle request validation?",
  "results": [
    {
      "id": "mem_1739821234567",
      "content": "FastAPI uses Pydantic v2 for request validation.",
      "score": 0.8923,
      "metadata": {"source": "docs"},
      "tier": "hot"
    }
  ]
}
```

#### `POST /feedback`
Report outcome of a retrieved memory (Phase 4.0 reliability loop).

```json
Request:
{
  "memory_id": "mem_1739821234567",
  "outcome": true,
  "comment": "This solution worked perfectly."
}

Response:
{
  "ok": true,
  "memory_id": "mem_1739821234567",
  "reliability_score": 0.714,
  "success_count": 4,
  "failure_count": 1
}
```

#### `GET /memory/{memory_id}`
Retrieve a specific memory with full metadata.

```json
Response:
{
  "id": "mem_1739821234567",
  "content": "...",
  "metadata": {...},
  "created_at": "2026-02-17T20:00:00Z",
  "ltp_strength": 1.847,
  "epistemic_value": 0.73,
  "reliability_score": 0.714,
  "tier": "hot"
}
```

#### `DELETE /memory/{memory_id}`
Delete memory from all tiers and clean up synapses.

#### `POST /concept`
Define a symbolic concept for analogical reasoning.

```json
{"name": "king", "attributes": {"gender": "man", "role": "ruler", "domain": "royalty"}}
```

#### `POST /analogy`
Solve analogies using VSA algebra: `source:value :: target:?`

```json
Request:  {"source_concept": "king", "source_value": "man", "target_concept": "queen"}
Response: {"results": [{"value": "woman", "score": 0.934}]}
```

#### `GET /insights/gaps`
Detect knowledge gaps from recent temporal co-activity (Phase 4.0).

```json
Response:
{
  "gaps_detected": 3,
  "knowledge_gaps": [
    {
      "concept_a": "asyncio event loop",
      "concept_b": "FastAPI middleware",
      "suggested_query": "How does event loop relate to middleware?",
      "co_occurrence_count": 5
    }
  ]
}
```

#### `POST /admin/consolidate`
Trigger manual semantic consolidation (normally runs automatically at 3 AM).

#### `GET /stats`
Engine statistics - tiers, synapse count, consolidation state. Requires `X-API-Key` by default.

#### `GET /health`
Public liveness and dependency diagnostics. A local-only runtime can report
`degraded` when Redis is unavailable while remaining live.

#### `GET /ready`
Readiness of the initialized local runtime. Redis is reported diagnostically
but is not required for the single-node prototype to become ready.

#### `GET /metrics/`
Prometheus metrics endpoint on the API listener (port 8100). The trailing slash
is canonical.

---

## Python Library Usage

For durable SQLite-backed memory with exact agent scopes, async/sync APIs,
bitemporal recall, supersession, and evidence receipts, see the
[Persistent Agent Memory Quickstart](docs/AGENT_MEMORY_QUICKSTART.md). Retrieval
in this foundation is SQLite FTS5 lexical search only; it is not semantic or
embedding retrieval.

### Preparing for the unreleased v3 migration

```python
from mnemocore.agent_memory import AgentMemory, MemoryScope
from mnemocore.hybrid import HybridMemoryRuntime
import asyncio

async def main():
    scope = MemoryScope(tenant_id="local", user_id="demo", agent_id="assistant")
    async with await AgentMemory.open("memory.db", scope=scope) as memory:
        runtime = HybridMemoryRuntime(memory, scope=scope)
        await memory.remember("User prefers concise answers")
        results = await runtime.recall(scope, "concise", limit=3)
        print(results)

asyncio.run(main())
```

The published package remains v2.0.0. In the unreleased v3 migration path,
`LiteEngine` and the entire unscoped `Memory(...)` facade are removed and fail
with a migration error; use `AgentMemory` with an explicit `MemoryScope`.
Every runtime operation uses the complete scope tuple; there is no broader
scope fallback. `HAIMEngine` remains available only for v2 compatibility: its
global JSONL/tiering lifecycle is not AgentMemory-backed and must not be used
as v3 persistence. Use `HybridMemoryRuntime` (or `create_v3_app` for HTTP)
for v3.
For a temporary bridge of its legacy `store`/`query`/`delete_memory` shape, use
the deprecated `HAIMEngineAdapter` with an already-open, scope-bound
`AgentMemory` client.

The hybrid runtime also has an explicit synchronous facade for synchronous
applications. Do not call it from an already-running event loop; use the async
runtime there. See the [v3 hybrid-runtime guide](docs/V3_HYBRID_RUNTIME.md) for
retrieval metadata, cognitive-plan safeguards, API composition, and the staged
remaining work.

### Advanced / Low-level (v2 compatibility only)

`HAIMEngine` below is retained for existing v2 integrations. It is deprecated
for new work, does not enforce `MemoryScope`, and is not the v3 persistence
layer. New v3 code must use the scoped runtime shown above.

```python
from mnemocore.core.engine import HAIMEngine

engine = HAIMEngine(persist_path="./data/memory.jsonl")

# Store memories
engine.store("Python generators are lazy iterators", metadata={"topic": "python"})
engine.store("Use 'yield' to create generator functions", metadata={"topic": "python"})
engine.store("Redis XADD appends to a stream", goal_id="infrastructure")

# Query (global)
results = engine.query("How do Python generators work?", top_k=3)
for mem_id, score in results:
    mem = engine.get_memory(mem_id)
    print(f"[{score:.3f}] {mem.content}")

# Query with context masking
results = engine.query("data streams", top_k=5, context="infrastructure")

engine.close()
```

### Analogical Reasoning

```python
# Define concepts
engine.define_concept("king",  {"gender": "man",   "role": "ruler"})
engine.define_concept("queen", {"gender": "woman", "role": "ruler"})
engine.define_concept("man",   {"gender": "man"})

# VSA analogy: king:man :: ?:woman → queen
result = engine.reason_by_analogy(
    src="king", val="man", tgt="woman"
)
print(result)  # [("queen", 0.934), ...]
```

### Working with the Binary HDV Layer Directly

```python
from mnemocore.core.binary_hdv import BinaryHDV, TextEncoder, majority_bundle

encoder = TextEncoder(dimension=16384)

# Encode text
python_vec  = encoder.encode("Python programming")
fastapi_vec = encoder.encode("FastAPI framework")
error_vec   = encoder.encode("error handling")

# Bind concept to role
python_in_fastapi = python_vec.xor_bind(fastapi_vec)

# Bundle multiple concepts into prototype
web_dev_prototype = majority_bundle([python_vec, fastapi_vec, error_vec])

# Similarity
print(python_vec.similarity(web_dev_prototype))  # High (part of bundle)
print(python_vec.similarity(error_vec))          # ~0.5 (unrelated)

# Batch nearest-neighbor search
from mnemocore.core.binary_hdv import batch_hamming_distance
import numpy as np

database = np.stack([v.data for v in [python_vec, fastapi_vec, error_vec]])
distances = batch_hamming_distance(python_vec, database)
```

### Reliability Feedback Loop

```python
mem_id = engine.store("Always use asyncio.Lock() in async code, not threading.Lock()")
results = engine.query("async locking")

# It works — report success
engine.provide_feedback(mem_id, outcome=True, comment="Solved deadlock issue")

# Over time, high-reliability memories get 'verified' tag
# and are ranked above unproven ones in future queries
```

### Semantic Consolidation

```python
stats = engine.trigger_consolidation()
print(f"Created {stats['abstractions_created']} semantic anchors")
print(f"Consolidated {stats['memories_consolidated']} episodic memories")

# Automatic: runs every night at 3 AM via background asyncio task
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- Redis and Qdrant are optional runtime services for legacy tiering and vector
  integrations. They are not required for local AgentMemory.
- The current 2.x wheel retains its compatibility dependencies. It does not
  connect to Redis or Qdrant merely by importing `mnemocore.agent_memory`.

### Quick Start (AgentMemory — stable local-first path)

```bash
# 1. Install. No Redis/Qdrant service is needed for AgentMemory.
pip install -e .
```

```python
# 2. Use AgentMemory with an explicit scope and local SQLite store.
from pathlib import Path
from mnemocore.agent_memory import MemoryScope, SyncAgentMemory

scope = MemoryScope(tenant_id="local", user_id="robin", agent_id="assistant")
memory = SyncAgentMemory.open(Path("agent-memory.sqlite3"), scope=scope)
memory.remember("User prefers concise answers")
print(memory.recall("concise"))
```

See "Full server setup (optional)" below for HAIM, Redis, Docker, and the REST API.

### Full server setup (optional, for Redis-backed etc.)

# To include dev tools (pytest, mypy, black, etc.):
pip install -e ".[dev]"

# Start Redis
docker run -d -p 6379:6379 redis:7.2-alpine

# Set API key (never hardcode — use env var or .env file)
# Windows PowerShell:
$env:HAIM_API_KEY = "your-secure-key-here"
# Linux / macOS:
# export HAIM_API_KEY="your-secure-key-here"

# Start the API locally
# PowerShell; use export MNEMOCORE_HOST=127.0.0.1 on Unix
$env:MNEMOCORE_HOST = "127.0.0.1"
mnemocore

The API is now live at `http://localhost:8100`. Visit `http://localhost:8100/docs` for the interactive Swagger UI.

> Do not bind MnemoCore directly to a public IP unless it is behind a trusted reverse proxy, TLS, and a strong `HAIM_API_KEY`. Diagnostics (`/stats`, `/rate-limits`, `/metrics`) require `X-API-Key` by default.

### Using the .env file

Copy the provided template and fill in your values — the API and docker-compose both pick it up automatically:

```bash
cp .env.example .env
# Edit .env and set HAIM_API_KEY, REDIS_PASSWORD, QDRANT_API_KEY, etc.
```

> **Note:** `.env` is listed in `.gitignore` and must never be committed. Only `.env.example` (with placeholder values) belongs in version control.

### Full Stack with Docker Compose

```bash
# Requires .env with HAIM_API_KEY, REDIS_PASSWORD, and QDRANT_API_KEY set.
docker compose config
docker compose up -d
```

This starts MnemoCore, Redis 7.2, and Qdrant. Only the API is published, on
`127.0.0.1:8100`; metrics are served from `/metrics/` on the same port.

### With Qdrant (Phase 4.x Scale)

```bash
# Start Qdrant alongside Redis
docker run -d -p 6333:6333 qdrant/qdrant

# Enable in config.yaml
qdrant:
  enabled: true
  host: localhost
  port: 6333
```

---

## Configuration

All configuration lives in `config.yaml`. Sensitive values can be overridden with environment variables — the config loader looks for `HAIM_`-prefixed vars and also honours per-service overrides like `HAIM_API_KEY`, `REDIS_PASSWORD`, `QDRANT_API_KEY`, `HAIM_CORS_ORIGINS`, and `SUBCONSCIOUS_AI_API_KEY`.

```yaml
haim:
  version: "5.0"
  dimensionality: 16384        # Binary vector dimensions (must be multiple of 64)

  encoding:
    mode: "binary"             # "binary" (recommended) or "float" (legacy, deprecated)
    token_method: "bundle"     # "bundle" (XOR+permute) or "hash"

  tiers:
    hot:
      max_memories: 2000       # Max nodes in RAM
      ltp_threshold_min: 0.7   # Evict below this LTP strength
      eviction_policy: "lru"
    warm:
      max_memories: 100000     # Max nodes in Redis/mmap
      ltp_threshold_min: 0.3
    cold:
      storage_backend: "filesystem"   # "filesystem" or "s3"
      compression: "gzip"

  ltp:
    initial_importance: 0.5
    decay_lambda: 0.01         # Higher = faster forgetting
    permanence_threshold: 0.95 # LTP above this is immune to decay
    half_life_days: 30.0

  hysteresis:
    promote_delta: 0.15        # LTP must exceed threshold by this much to promote
    demote_delta: 0.10

  redis:
    url: "redis://localhost:6379/0"
    stream_key: "haim:subconscious"
    max_connections: 10
    socket_timeout: 5
    # password: set via REDIS_PASSWORD env var

  qdrant:
    url: "http://localhost:6333"
    collection_hot:  "haim_hot"
    collection_warm: "haim_warm"
    enabled: false
    # api_key: set via QDRANT_API_KEY env var

  security:
    # api_key: set via HAIM_API_KEY env var — never hardcode here
    cors_origins: ["http://localhost:3000"]
    public_health_enabled: true
    public_stats_enabled: false
    public_rate_limits_enabled: false
    public_metrics_enabled: false
    require_api_key_for_export: true

  subconscious_ai:
    enabled: false
    api_url: "https://api.openai.com/v1/chat/completions"
    model: "gpt-4o-mini"
    dream_interval_seconds: 300
    batch_size: 5
    # api_key: set via SUBCONSCIOUS_AI_API_KEY env var

  observability:
    metrics_port: 8100
    log_level: "INFO"
    structured_logging: true

  paths:
    data_dir: "./data"
    memory_file: "./data/memory.jsonl"
    codebook_file: "./data/codebook.json"
    concepts_file: "./data/concepts.json"
    synapses_file: "./data/synapses.json"
    warm_mmap_dir: "./data/warm_tier"
    cold_archive_dir: "./data/cold_archive"

  mcp:
    enabled: false
    transport: "stdio"
    host: "127.0.0.1"
    port: 8110
    api_base_url: "http://localhost:8100"
```

### Security Note

MnemoCore requires an explicit API key. There is no default fallback key in production builds.

```bash
# Generate a cryptographically secure key:
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set it (never commit this value):
export HAIM_API_KEY="<generated-value>"
```

Persistent webhooks store only an opaque `secret_ref`; the signing secret is
resolved when a delivery is sent. Inline signing secrets, legacy plaintext
records, and all custom headers are rejected for persistent webhook managers.
Writes are serialized and atomically replaced, and malformed persistence loads
fail closed without partial in-memory state. Memory-only webhook managers retain
the legacy inline-secret/header behavior for compatibility.

AgentMemory also exposes a physical SQLite erase operation with exact-scope
ownership checks, optional whole-supersession-component cascade, dependent-row
cleanup, database integrity checks, and a content-free `ErasureReceipt`. All
database users must cooperate with the store's sidecar lock; raw SQLite clients
are outside the safety contract during erasure. Backups, derived external
artifacts, power-loss recovery, and exhaustive failure injection remain separate
production gates.

---

## MCP Server Integration (v2 compatibility)

MnemoCore exposes a legacy **Model Context Protocol (MCP)** server for existing
v2 integrations. It proxies the legacy HAIM REST API and is not backed by v3
AgentMemory truth. New scoped integrations should use the AgentMemory MCP
helpers in `mnemocore.integrations.mcp` (or build on `HybridMemoryRuntime`).

### Setup

```bash
# Start the legacy v2 API first
uvicorn mnemocore.api.main:app --host 0.0.0.0 --port 8100

# Configure MCP in config.yaml
haim:
  mcp:
    enabled: true
    transport: "stdio"  # or "sse" for streaming

# Run MCP server
python -m mnemocore.mcp.server
```

### Claude Desktop Configuration

Add to your Claude Desktop `config.json`:

```json
{
  "mcpServers": {
    "mnemocore": {
      "command": "python",
      "args": ["-m", "mnemocore.mcp.server"],
      "env": {
        "HAIM_API_KEY": "your-key",
        "HAIM_BASE_URL": "http://localhost:8100"
      }
    }
  }
}
```

Once connected, the agent can:
- `store_memory(content, context)` — persist learned information
- `query_memory(query, context, top_k)` — recall relevant memories
- `provide_feedback(memory_id, outcome)` — signal what worked
- `get_knowledge_gaps()` — surface what it doesn't understand

---

## Observability

MnemoCore ships with built-in Prometheus metrics and structured logging.

### Prometheus Metrics

Available at canonical `GET /metrics/` on the API listener, port 8100:

| Metric | Description |
|--------|-------------|
| `haim_api_request_count` | Total requests by endpoint and status |
| `haim_api_request_latency_seconds` | Request latency histogram |
| `haim_storage_operation_count` | Store/query/delete operations |
| `haim_hot_tier_size` | Current HOT tier memory count |
| `haim_synapse_count` | Active synaptic connections |

### Grafana Dashboard

A sample Grafana dashboard config is available at `grafana-dashboard.json` in the repository root. Import it directly into Grafana via **Dashboards → Import → Upload JSON file**.

### Structured Logging

All components use structured Python logging with contextual fields:

```
2026-02-17 20:00:00 INFO  Stored memory mem_1739821234567 (EIG: 0.7823)
2026-02-17 20:00:01 INFO  Memory mem_1739821234567 reliability updated: 0.714 (4✓ / 1✗)
2026-02-17 03:00:00 INFO  Consolidation complete: abstractions_created=12, consolidated=847
2026-02-17 04:00:00 INFO  Knowledge gap detected: asyncio ↔ FastAPI middleware (5 co-occurrences)
```

---

## Testing

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific feature tests
pytest tests/test_xor_attention.py         # Contextual masking
pytest tests/test_stability.py             # Reliability/Bayesian stability
pytest tests/test_consolidation.py         # Semantic consolidation
pytest tests/test_engine_cleanup.py        # Cleanup and decay
pytest tests/test_phase43_regressions.py   # Phase 4.3 regression guardrails
pytest tests/test_tier_manager.py          # Tier demotion / promotion logic
pytest tests/test_dream_loop.py            # Subconscious dream loop
pytest tests/test_subconscious_ai_worker.py # LLM-powered dream worker (if offline: uses mocks)
pytest tests/test_recursive_synthesizer.py  # Deep concept synthesis
pytest tests/test_batch_ops.py             # Bulk ingestion operations
pytest tests/test_mcp_server.py            # MCP server adapter

# End-to-end flow
pytest tests/test_e2e_flow.py -v
```

---

## Roadmap

### Current Release (v2.0.0)

- [x] Binary HDV core (XOR bind / bundle / permute / Hamming)
- [x] Three-tier HOT/WARM/COLD memory lifecycle
- [x] Async API + MCP integration
- [x] XOR attention masking + Bayesian reliability updates
- [x] Semantic consolidation, immunology cleanup, and gap detection/filling
- [x] Temporal recall: episodic chaining + chrono-weighted query
- [x] Phase 4.4 — Subconscious AI Worker (LLM-powered dream synthesis)
- [x] Phase 4.5 — Subconscious Daemon, persistence hardening, tier-manager demotion race fix
- [x] Dependency-injection Container pattern (replaces singleton)
- [x] HNSW in-process index for hot-tier ANN search
- [x] Batch operations for bulk ingestion
- [x] Meta-cognition layer: GoalTree + LearningJournal
- [x] Phase 5 — Cognitive Architecture: Working Memory, Episodic/Semantic/Procedural stores, Meta Memory, Self-Improvement Worker, 7-phase Pulse loop
- [x] Phase 5 Hardening — 136 additional tests, config organization
- [x] Phase 6 — Research Services: StrategyBank, KnowledgeGraph, MemoryScheduler, SAMEP, 4 new Pulse phases
- [x] 1291+ tests passing

### Next Steps

- [ ] Production hardening: distributed HOT-tier, CUDA kernels, chaos engineering
- [ ] Self-improvement Phase 1: enable writes with human-in-the-loop approval
- [ ] Multi-modal memory: image/audio embedding via CLIP/Whisper
- [ ] Ecosystem: LangChain, LlamaIndex, CrewAI adapters
- [ ] Extended observability standardization (`mnemocore_*` metric prefix)

---

## Contributing

MnemoCore is an active research project. Contributions are welcome — especially:

- **Performance**: CUDA kernels, FAISS integration, async refactoring
- **Algorithms**: Better clustering for consolidation, improved EIG formulas
- **Integrations**: New storage backends, LLM connectors
- **Tests**: Coverage for edge cases, property-based testing

### Process

```bash
# Fork and clone
git checkout -b feature/your-feature-name

# Make changes, ensure tests pass
pytest

# Commit with semantic message
git commit -m "feat(consolidation): add LLM-powered prototype labeling"

# Open PR — describe the what, why, and performance impact
```

Please follow the implementation patterns established in `docs/ARCHITECTURE.md` and `docs/ROADMAP.md` for architectural guidance, and review `CHANGELOG.md` to understand what has already landed.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture with Mermaid diagrams |
| [docs/API.md](docs/API.md) | Complete REST API reference (41 endpoints) |
| [docs/CLI.md](docs/CLI.md) | Command-line interface reference |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | All configuration options (41 dataclasses) |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker, Compose, and Kubernetes deployment |
| [docs/EVENTS.md](docs/EVENTS.md) | Event system and webhook delivery |
| [docs/FORGETTING.md](docs/FORGETTING.md) | SM-2 spaced repetition and decay system |
| [docs/LLM_INTEGRATION.md](docs/LLM_INTEGRATION.md) | Multi-provider LLM integration |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Terms and abbreviations |
| [docs/SUBCONSCIOUS_AI.md](docs/SUBCONSCIOUS_AI.md) | Subconscious worker and dream pipeline |
| [docs/RECONSTRUCTIVE_MEMORY.md](docs/RECONSTRUCTIVE_MEMORY.md) | Reconstructive recall module |
| [docs/EMBEDDING_REGISTRY.md](docs/EMBEDDING_REGISTRY.md) | Embedding model versioning |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Robin Granberg**  
📧 robin@veristatesystems.com

---

<p align="center">
  <i>Building the cognitive substrate for the next generation of autonomous AI.</i>
</p>
