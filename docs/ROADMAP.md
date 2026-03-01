# MnemoCore Roadmap — Detailed Design Explorations

> **Note:** The canonical roadmap lives at [ROADMAP.md](../ROADMAP.md) in the project root.
> This document preserves detailed design explorations for future phases.

---

## Version History

| Version | Phase | Status | Key Features |
|---------|-------|--------|--------------|
| 3.x | Core Architecture | ✅ Complete | Binary HDV, 3-Tier Storage, LTP/Decay |
| 4.0 | Cognitive Enhancements | ✅ Complete | XOR Attention, Bayesian LTP, Gap Detection, Immunology |
| 4.1 | Observability | ✅ Complete | Prometheus metrics, distributed tracing, project isolation |
| 4.2 | Stability | ✅ Complete | Async lock fixes, test suite hardening |
| 4.3 | Temporal Recall | ✅ Complete | Episodic chaining, chrono-weighting, sequential context |
| 4.5 | Subconscious | ✅ Complete | Dream daemon, DI container, HNSW, batch ops, meta-cognition |
| 5.0–5.4 | Cognitive Architecture | ✅ Complete | WM, Episodic, Semantic, Procedural, Meta, SelfImprovement, Pulse |
| 6.0 | Research Services | ✅ Complete | StrategyBank, KnowledgeGraph, MemoryScheduler, SAMEP |

---

## Design Explorations for Future Phases

### 5.0 Multi-Modal Memory

**Goal:** Enable storage and retrieval of images, audio, code structures, and cross-modal associations.

```
┌─────────────────────────────────────────────────────────────────┐
│  CURRENT: Text-only encoding                                    │
│  ────────────────────────────────────────────────────────────── │
│  store("User reported bug") → BinaryHDV                         │
│                                                                 │
│  FUTURE: Multi-modal encoding                                   │
│  ────────────────────────────────────────────────────────────── │
│  store("Screenshot of error", image=bytes) → CrossModalHDV      │
│  store("Voice note", audio=bytes) → AudioHDV                    │
│  bind(text_id, image_id, relation="illustrates")                │
│                                                                 │
│  query("API error", modality="image") → screenshot.png          │
│  query(image=bytes, modality="text") → "Related conversation"   │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation Plan:**

| Component | Description | Dependencies |
|-----------|-------------|--------------|
| `MultiModalEncoder` | Abstract encoder protocol | - |
| `CLIPEncoder` | Vision encoding via CLIP | `transformers`, `torch` |
| `WhisperEncoder` | Audio encoding via Whisper | `openai-whisper` |
| `CodeEncoder` | AST-aware code encoding | `tree-sitter` |
| `CrossModalBinding` | VSA operations across modalities | BinaryHDV |

**New API Endpoints:**
```
POST /store/multi          - Store with multiple modalities
POST /query/cross-modal    - Cross-modal semantic search
POST /bind                 - Bind modalities together
GET  /memory/{id}/related  - Get cross-modal related memories
```

---

### 5.1 Emotional/Affective Layer

**Goal:** Enable emotion-weighted memory storage, retrieval, and decay - mimicking how biological memory prioritizes emotionally significant events.

```
┌─────────────────────────────────────────────────────────────────┐
│  EMOTIONAL DIMENSIONS                                           │
│  ────────────────────────────────────────────────────────────── │
│                                                                 │
│  Valence:  [-1.0 ──────────────── +1.0]                         │
│            (negative/unpleasant)  (positive/pleasant)           │
│                                                                 │
│  Arousal:  [0.0 ────────────────── 1.0]                         │
│            (calm/neutral)         (intense/urgent)              │
│                                                                 │
│  EFFECT ON MEMORY:                                              │
│  ────────────────────────────────────────────────────────────── │
│  High Arousal + Negative = "Flashbulb memory" (never forget)    │
│  High Arousal + Positive = Strong consolidation                 │
│  Low Arousal = Faster decay (forgettable)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**MemoryNode Extensions:**
```python
@dataclass
class MemoryNode:
    # ... existing fields ...

    # Phase 5.1: Emotional tagging
    emotional_valence: float = 0.0      # -1.0 (negative) to +1.0 (positive)
    emotional_arousal: float = 0.0      # 0.0 (calm) to 1.0 (intense)
    emotional_tags: List[str] = field(default_factory=list)  # ["frustration", "joy", "urgency"]

    def emotional_weight(self) -> float:
        """Calculate memory importance based on emotional factors."""
        # Arousal amplifies retention regardless of valence
        # High arousal creates "flashbulb memories"
        return abs(self.emotional_valence) * self.emotional_arousal
```

**Modified LTP Formula:**
```
S = I × log(1+A) × e^(-λT) × (1 + E)

Where E = emotional_weight() ∈ [0, 1]
```

**Use Cases:**
- B2B outreach: "Customer was almost in tears when we fixed their issue" → HIGH priority
- Support tickets: "User furious about data loss" → Never forget, prioritize retrieval
- Positive feedback: "User loved the new feature" → Moderate retention

---

### 5.2 Working Memory Layer

**Goal:** Active cognitive workspace for goal-directed reasoning, not just passive storage.

```
┌─────────────────────────────────────────────────────────────────┐
│                    COGNITIVE ARCHITECTURE                        │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │              WORKING MEMORY (Active)                     │  │
│    │              Capacity: 7 ± 2 items                       │  │
│    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │  │
│    │  │  Goal   │ │ Context │ │  Focus  │ │ Hold    │        │  │
│    │  │         │ │         │ │         │ │         │        │  │
│    │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │  │
│    └─────────────────────────────────────────────────────────┘  │
│                              ↕                                   │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │              HOT TIER (Fast Access)                      │  │
│    │              ~2,000 memories, <1ms access                │  │
│    └─────────────────────────────────────────────────────────┘  │
│                              ↕                                   │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │              WARM TIER (Qdrant/Redis)                    │  │
│    │              ~100,000 memories, <10ms access             │  │
│    └─────────────────────────────────────────────────────────┘  │
│                              ↕                                   │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │              COLD TIER (Archive)                         │  │
│    │              Unlimited, <100ms access                    │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Working Memory API:**
```python
# Create working memory instance
wm = engine.working_memory(capacity=7)

# Set active goal
wm.set_goal("Troubleshoot authentication error")

# Load relevant context
wm.focus_on(await engine.query("auth error", top_k=5))

# Hold important constraints
wm.hold("User is on deadline - prioritize speed over elegance")

# Query with working memory context
results = wm.query("related issues")
# Results are RE-RANKED based on current goal + focus + held items

# Get context summary for LLM
context = wm.context_summary()
# → "Working on: auth troubleshooting
#    Focus: Recent OAuth errors
#    Constraint: Time pressure"
```

**Implementation Components:**
| Component | Description |
|-----------|-------------|
| `WorkingMemory` | Active workspace class |
| `GoalContext` | Goal tracking and binding |
| `FocusBuffer` | Currently attended items |
| `HoldBuffer` | Constraints and important facts |
| `ContextualQuery` | Goal-directed retrieval |

---

### 5.3 Multi-Agent / Collaborative Memory

**Goal:** Enable memory sharing between agents while maintaining provenance and privacy.

```
┌─────────────────────────────────────────────────────────────────┐
│                    COLLABORATIVE MEMORY                          │
│                                                                 │
│     Agent A          Shared Memory           Agent B            │
│    ┌────────┐      ┌──────────────┐        ┌────────┐          │
│    │ Private│      │              │        │ Private│          │
│    │ Memory │◄────►│  Consensus   │◄──────►│ Memory │          │
│    └────────┘      │   Layer      │        └────────┘          │
│                    │              │                             │
│    Agent C         │  Provenance  │         Agent D             │
│    ┌────────┐      │  Tracking    │        ┌────────┐          │
│    │ Private│◄────►│              │◄──────►│ Private│          │
│    │ Memory │      │  Privacy     │        │ Memory │          │
│    └────────┘      │  Filtering   │        └────────┘          │
│                    └──────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- Memory provenance: Track which agent created/modified each memory
- Privacy levels: Private, shared-with-group, public
- Conflict resolution: When agents disagree on facts
- Collective intelligence: Aggregate insights across agents

---

### 5.4 Continual Learning

**Goal:** Enable online adaptation without catastrophic forgetting.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTINUAL LEARNING                            │
│                                                                 │
│  Traditional ML:     Train → Deploy → (forget) → Retrain        │
│                                                                 │
│  MnemoCore 5.4:      Learn → Consolidate → Adapt → Learn → ...  │
│                           ↑______________|                       │
│                                                                 │
│  KEY MECHANISMS:                                                │
│  ─────────────────────────────────────────────────────────────  │
│  • Elastic Weight Consolidation (EWC) for encoder               │
│  • Replay-based consolidation during "sleep" cycles             │
│  • Progressive neural networks for new domains                  │
│  • Meta-learning for rapid adaptation                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Priorities

### Agent Frameworks
| Framework | Priority | Use Case |
|-----------|----------|----------|
| Open Claw | ⭐⭐⭐⭐⭐ | Primary use case, deep integration |
| LangChain | ⭐⭐⭐⭐ | Memory provider plugin |
| CrewAI | ⭐⭐⭐⭐ | Shared memory between agents |
| AutoGen | ⭐⭐⭐ | Conversation memory backend |
| LlamaIndex | ⭐⭐⭐ | Vector store adapter |

### AI Platforms
| Platform | Priority | Integration Type |
|----------|----------|------------------|
| Claude (Anthropic) | ⭐⭐⭐⭐⭐ | MCP server (existing) |
| OpenAI Codex | ⭐⭐⭐⭐⭐ | API + function calling |
| Ollama | ⭐⭐⭐⭐ | Native memory backend |
| LM Studio | ⭐⭐⭐ | Plugin architecture |
| Gemini | ⭐⭐⭐ | API adapter |

---

## Research Opportunities

### Academic Collaborations
| Area | Institutions | Relevance |
|------|-------------|-----------|
| Hyperdimensional Computing | Stanford, IBM Research, Redwood Center | Core HDC/VSA theory |
| Computational Neuroscience | MIT, UCL, KTH | Biological validation |
| Cognitive Architecture | Carnegie Mellon, University of Michigan | SOAR/ACT-R comparison |
| Neuromorphic Computing | Intel Labs, ETH Zürich | Hardware acceleration |

### Publication Opportunities
1. **"Binary HDC for Long-term AI Memory"** - Novel approach to persistent memory
2. **"Episodic Chaining in Vector Memory Systems"** - Phase 4.3 temporal features
3. **"XOR Attention Masking for Memory Isolation"** - Project isolation innovation
4. **"Bayesian LTP in Artificial Memory Systems"** - Biological plausibility

---

## Status Note

The design explorations above for Multi-Modal Memory (5.0), Emotional Layer (5.1), Working Memory (5.2), Multi-Agent Memory (5.3), and Continual Learning (5.4) informed the Phase 5 and Phase 6 implementations. Many concepts were realized differently than originally envisioned here — see `IMPLEMENTATION_PROGRESS.md` for what was actually built.

---

*Last Updated: 2026-03-01*
*Current Version: 2.0.0*
