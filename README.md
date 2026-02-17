# MnemoCore

### Infrastructure for Persistent Cognitive Memory

> *"Memory is not a container. It is a living process — a holographic continuum where every fragment contains the whole."*

<p align="center">
  <img src="https://img.shields.io/badge/Status-Beta%203.5.1-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Async%20Ready-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/HDV-16384--dim-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Vectors-Binary%20VSA-critical?style=for-the-badge" />
</p>

---

## What is MnemoCore?

**MnemoCore** is a research-grade cognitive memory infrastructure that gives AI agents a brain — not just a database.

Traditional vector stores retrieve. MnemoCore **thinks**. It is built on the mathematical framework of **Binary Hyperdimensional Computing (HDC)** and **Vector Symbolic Architectures (VSA)**, principles rooted in Pentti Kanerva's landmark 2009 theory of cognitive computing. Every memory is encoded as a **16,384-dimensional binary holographic vector** — a format that is simultaneously compact (2,048 bytes), noise-tolerant (Hamming geometry), and algebraically rich (XOR binding, majority bundling, circular permutation).

At its core lives the **Holographic Active Inference Memory (HAIM) Engine** — a system that does not merely answer queries, but:

- **Evaluates** the epistemic novelty of every incoming memory before deciding to store it
- **Dreams** — strengthening synaptic connections between related memories during idle cycles
- **Reasons by analogy** — if `king:man :: ?:woman`, the VSA soul computes `queen`
- **Self-organizes** into tiered storage based on biologically-inspired Long-Term Potentiation (LTP)
- **Scales** from a single process to distributed nodes targeting 1B+ memories

MnemoCore is being developed at **Veristate Systems** as the foundational memory layer for autonomous AI agent systems. Phase 4.0 introduces five cognitive enhancements — contextual query masking, reliability feedback loops, semantic consolidation, auto-associative cleanup, and proactive knowledge gap detection — transforming it from passive storage into an active reasoning substrate.

---

## Table of Contents

- [Architecture](#architecture)
- [Core Technology](#core-technology-binary-hdv--vsa)
- [The Memory Lifecycle](#the-memory-lifecycle)
- [Tiered Storage](#tiered-storage-hotwarmdcold)
- [Phase 4.0 Cognitive Enhancements](#phase-40-cognitive-enhancements)
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
| **HAIM Engine** | `src/core/engine.py` | Central cognitive coordinator — store, query, dream, delete |
| **BinaryHDV** | `src/core/binary_hdv.py` | 16384-dim binary vector math (XOR, Hamming, bundle, permute) |
| **TextEncoder** | `src/core/binary_hdv.py` | Token→HDV pipeline with positional permutation binding |
| **MemoryNode** | `src/core/node.py` | Memory unit with LTP, epistemic values, tier state |
| **TierManager** | `src/core/tier_manager.py` | HOT/WARM/COLD orchestration with LTP-driven eviction |
| **SynapticConnection** | `src/core/synapse.py` | Hebbian synapse with strength, decay, and fire tracking |
| **ConceptualMemory** | `src/core/holographic.py` | VSA soul for analogy and cross-domain symbolic reasoning |
| **AsyncRedisStorage** | `src/core/async_storage.py` | Async Redis backend (WARM tier + pub/sub) |
| **API** | `src/api/main.py` | FastAPI REST interface with async wrappers and middleware |
| **MCP Server** | `src/mcp/server.py` | Model Context Protocol adapter for agent tool integration |

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
from src.core.binary_hdv import majority_bundle

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

MnemoCore Phase 4.0 introduces five architectural enhancements that elevate the system from **data retrieval** to **cognitive reasoning**. Full implementation specifications are in [`COGNITIVE_ENHANCEMENTS.md`](COGNITIVE_ENHANCEMENTS.md).

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
engine.store("API rate limiting logic", goal_id="Veristate")
engine.store("Garden watering schedule", goal_id="HomeProject")

# Query with context mask — only Veristate memories surface
results = engine.query("API logic", top_k=5, context="Veristate")
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

## API Reference

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
  "context": "Veristate",
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
  "context": "Veristate"
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
Engine statistics — tiers, synapse count, consolidation state.

#### `GET /health`
Health check — Redis connectivity, engine readiness, degraded mode status.

#### `GET /metrics`
Prometheus metrics endpoint.

---

## Python Library Usage

### Basic Store and Query

```python
from src.core.engine import HAIMEngine

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
from src.core.binary_hdv import BinaryHDV, TextEncoder, majority_bundle

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
from src.core.binary_hdv import batch_hamming_distance
import numpy as np

# Build matrix of packed vectors
database = np.stack([v.data for v in [python_vec, fastapi_vec, error_vec]])
distances = batch_hamming_distance(python_vec, database)
```

### Reliability Feedback Loop

```python
# Store and retrieve
mem_id = engine.store("Always use asyncio.Lock() in async code, not threading.Lock()")

# Retrieve in context
results = engine.query("async locking")

# Apply solution in code — it works!
engine.provide_feedback(mem_id, outcome=True, comment="Solved deadlock issue")

# Over time, high-reliability memories get 'verified' tag
# and are ranked above unproven ones in future queries
```

### Semantic Consolidation

```python
# Manual consolidation trigger
stats = engine.trigger_consolidation()
print(f"Created {stats['abstractions_created']} semantic anchors")
print(f"Consolidated {stats['memories_consolidated']} episodic memories")

# Automatic: runs every night at 3 AM via background asyncio task
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **Redis 6+** — Required for WARM tier and async event streaming
- **Qdrant** *(optional)* — For COLD tier at billion-scale
- **Docker** *(recommended)* — For Redis and Qdrant services

### Quick Start

```bash
# 1. Clone
git clone https://github.com/RobinALG87/MnemoCore-Infrastructure-for-Persistent-Cognitive-Memory.git
cd MnemoCore-Infrastructure-for-Persistent-Cognitive-Memory

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Redis
docker run -d -p 6379:6379 redis:alpine

# 5. Set API key
export HAIM_API_KEY="your-secure-key-here"

# 6. Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8100
```

The API is now live at `http://localhost:8100`. Visit `http://localhost:8100/docs` for the interactive Swagger UI.

### With Qdrant (Phase 3.5+ Scale)

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

All configuration lives in `config.yaml`. Values can be overridden with environment variables (`HAIM_` prefix).

```yaml
haim:
  version: "3.5"
  dimensionality: 16384        # Binary vector dimensions (must be multiple of 8)

  encoding:
    mode: "binary"             # "binary" (recommended) or "float" (legacy)

  tiers:
    hot:
      max_memories: 2000       # Max nodes in RAM
      ltp_threshold: 0.3       # Evict below this LTP strength
    warm:
      max_memories: 100000     # Max nodes in Redis/mmap
    cold:
      enabled: true

  ltp:
    initial_importance: 0.5
    decay_lambda: 0.01         # Higher = faster forgetting
    permanence_threshold: 2.0  # LTP above this is considered permanent

  redis:
    url: "redis://localhost:6379/0"

  qdrant:
    enabled: false
    host: "localhost"
    port: 6333
    collection: "mnemocore_warm"

  security:
    api_key: "${HAIM_API_KEY}"  # Never hardcode — use env variable
    cors_origins: ["http://localhost:3000"]

  observability:
    metrics_enabled: true
    log_level: "INFO"

  paths:
    data_dir: "./data"
    memory_file: "./data/memory.jsonl"
    synapses_file: "./data/synapses.jsonl"
```

### Security Note

MnemoCore requires an explicit API key. There is no default fallback key in production builds.

```bash
# Required — will raise exception if not set
export HAIM_API_KEY="$(openssl rand -hex 32)"
```

---

## MCP Server Integration

MnemoCore exposes a **Model Context Protocol (MCP)** server, enabling direct integration with Claude, GPT-4, and any MCP-compatible agent framework.

### Setup

```bash
# Start API first
uvicorn src.api.main:app --host 0.0.0.0 --port 8100

# Configure MCP in config.yaml
haim:
  mcp:
    enabled: true
    transport: "stdio"  # or "sse" for streaming

# Run MCP server
python -m src.mcp.server
```

### Claude Desktop Configuration

Add to your Claude Desktop `config.json`:

```json
{
  "mcpServers": {
    "mnemocore": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
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

Available at `GET /metrics`:

| Metric | Description |
|--------|-------------|
| `haim_api_request_count` | Total requests by endpoint and status |
| `haim_api_request_latency_seconds` | Request latency histogram |
| `haim_storage_operation_count` | Store/query/delete operations |
| `haim_hot_tier_size` | Current HOT tier memory count |
| `haim_synapse_count` | Active synaptic connections |

### Grafana Dashboard

A sample Grafana dashboard config is available at `observability/grafana_dashboard.json`.

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
pytest tests/test_contextual_masking.py    # Enhancement #1
pytest tests/test_reliability.py           # Enhancement #2
pytest tests/test_consolidation.py         # Enhancement #3
pytest tests/test_cleanup.py               # Enhancement #4
pytest tests/test_gap_detection.py         # Enhancement #5

# Integration suite (requires Redis)
pytest tests/integration/ -m "integration"

# Phase 4.0 full cognitive suite
pytest tests/integration/test_cognitive_suite.py -v
```

---

## Roadmap

### Phase 3.5 *(Current — Beta)*

- [x] Binary HDV core with SHAKE-256 token encoding
- [x] Three-tier HOT/WARM/COLD architecture
- [x] LTP-driven memory eviction and promotion
- [x] Async FastAPI with Redis dual-write
- [x] Subconscious dream loop with Hebbian synapse strengthening
- [x] Conceptual reasoning layer (analogy, cross-domain inference)
- [x] Prometheus metrics + structured logging
- [x] MCP server adapter
- [ ] Qdrant binary index integration (COLD tier)
- [ ] asyncio.Lock migration (threading safety)
- [ ] UUID standardization (ID format)

### Phase 4.0 *(In Development)*

- [ ] Contextual query masking (XOR attention)
- [ ] Reliability feedback loop (Bayesian LTP)
- [ ] Nightly semantic consolidation worker
- [ ] Auto-associative cleanup loop (vector immunology)
- [ ] Knowledge gap detection (proactive curiosity)
- [ ] Autonomous gap-filling via LLM integration
- [ ] FAISS / HNSW ANN index for HOT tier
- [ ] Synapse adjacency list (O(1) lookup)

### Phase 5.0 *(Research)*

- [ ] GPU CUDA kernels for massively parallel bitwise ops
- [ ] Federated distributed memory consensus across nodes
- [ ] Temporal sequence memory (episodic ordering)
- [ ] Active goal-directed memory search (free energy minimization)
- [ ] Neuromorphic hardware deployment (Intel Loihi, IBM NorthPole)

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

Please follow the implementation patterns in [`COGNITIVE_ENHANCEMENTS.md`](COGNITIVE_ENHANCEMENTS.md) and [`CODE_REVIEW_ISSUES.md`](CODE_REVIEW_ISSUES.md) for architectural guidance.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Robin Granberg** — Founder, Veristate Systems  
📧 Robin@veristatesystems.com  
🌐 [veristatesystems.com](https://veristatesystems.com)

---

<p align="center">
  <i>MnemoCore is a research initiative by Veristate Systems.<br/>
  Building the cognitive substrate for the next generation of autonomous AI.</i>
</p>
