# MnemoCore Glossary

> A reference of terms, abbreviations, and concepts used throughout MnemoCore.

---

## Core Concepts

| Term | Definition |
|------|-----------|
| **HAIM** | Holographic Associative Inference Memory — the core engine architecture combining HDV encoding with tiered storage and cognitive processing. |
| **HDV** | Hyperdimensional Vector — a high-dimensional binary vector (default 16,384 bits) used to represent memories. Based on Vector Symbolic Architecture (VSA). |
| **VSA** | Vector Symbolic Architecture — a computational framework using high-dimensional vectors for symbolic reasoning. Operations include bundling (OR/majority), binding (XOR), and permutation. |
| **Memory Node** | The fundamental unit of storage. Contains content text, its HDV encoding, metadata, LTP strength, tier placement, and cognitive annotations. |
| **Tier** | One of three storage levels: HOT (in-memory, fast), WARM (Redis/mmap, medium), COLD (Qdrant/disk, large). |

## Memory Strength & Decay

| Term | Definition |
|------|-----------|
| **LTP** | Long-Term Potentiation — a strength score (0.0–1.0) representing how well-established a memory is. Inspired by neuroscience. Higher LTP means the memory is more resistant to decay and eviction. |
| **EIG** | Expected Information Gain — an epistemic value measuring how much new information a memory provides. Used for prioritization. |
| **SM-2** | SuperMemo 2 Algorithm — a spaced repetition algorithm that schedules memory reviews based on recall quality. Tracks easiness factor, interval, and repetition count. |
| **Forgetting Curve** | The exponential decay of memory retention over time, modeled by Ebbinghaus's curve. MnemoCore uses this with SM-2 to schedule reviews. |
| **Permanence Threshold** | LTP value (default 0.95) above which a memory is considered "permanent" and exempt from decay. |
| **Hysteresis** | A buffer zone around tier boundaries that prevents rapid oscillation (thrashing) between tiers during promotion/demotion. |

## Cognitive Architecture

| Term | Definition |
|------|-----------|
| **Working Memory (WM)** | Short-term active memory buffer for each agent. Limited capacity (default 20 items), items have TTL. Inspired by Miller's 7±2 model. |
| **Episodic Memory** | Temporal chains of events tied to specific episodes/tasks. Each episode has a goal, timeline, and linked memories. |
| **Semantic Memory** | General knowledge stored as concepts. Consolidated from repeated episodic experiences via CLS. |
| **Procedural Memory** | Skills and procedures — "how to" knowledge. Tracked with success/failure rates for refinement. |
| **Meta Memory** | Self-awareness about the memory system's own performance. Detects anomalies and generates improvement proposals. |

## Processing & Consolidation

| Term | Definition |
|------|-----------|
| **Dream Pipeline** | A multi-stage offline consolidation process inspired by sleep-based memory consolidation. Clusters memories, extracts patterns, resolves contradictions, and promotes important memories. |
| **Pulse Loop** | The cognitive heartbeat — a periodic tick cycle (default 30s) that runs 11 maintenance phases including WM cleanup, episodic chaining, gap detection, and strategy refinement. |
| **Consolidation** | The process of merging, compressing, or promoting memories across tiers based on LTP, access patterns, and similarity. |
| **CLS** | Complementary Learning Systems — a theory where rapid episodic learning is slowly consolidated into semantic knowledge, inspired by hippocampal-neocortical interactions. |
| **Contradiction Resolution** | Detection and resolution of conflicting memories. Can be automatic (via LLM) or manual (via API). |
| **Gap Detection** | Identification of missing knowledge — queries that return low-confidence results, indicating areas where the system lacks information. |

## Encoding & Search

| Term | Definition |
|------|-----------|
| **Binary HDV** | The default encoding mode: 16,384-bit binary vectors. Operations use XOR (binding), majority vote (bundling), and cyclic shift (permutation). |
| **Hamming Distance** | The number of differing bits between two binary vectors. Used as the similarity metric for HDV search. |
| **HNSW** | Hierarchical Navigable Small World — an approximate nearest neighbor (ANN) algorithm used for fast vector search. |
| **Hybrid Search** | Combines dense (vector) and sparse (BM25) retrieval, fused via Reciprocal Rank Fusion (RRF). |
| **RRF** | Reciprocal Rank Fusion — a method for combining ranked lists from different retrieval systems. Parameter `k` (default 60) controls the contribution curve. |
| **Query Expansion** | Automatic enrichment of queries with related terms to improve recall. |

## Subconscious & AI

| Term | Definition |
|------|-----------|
| **Subconscious AI** | An LLM-powered background worker that analyzes memories, generates insights, and performs dream synthesis. Runs autonomously at configurable intervals. |
| **RLM** | Recursive Language Model — a query strategy that decomposes complex questions into sub-queries, retrieves relevant memories for each, and synthesizes a unified answer. |
| **Ripple Context** | Context expansion that follows synaptic connections from initial results to find related memories, similar to how neural activation spreads. |
| **Synapse** | A weighted connection between two memories, forming an association graph. Synapses are created automatically when similar memories are stored, and strengthen with co-activation. |
| **Attention Masking** | XOR-based masking that emphasizes certain dimensions during query time, improving retrieval relevance. |

## Multi-Agent & Exchange

| Term | Definition |
|------|-----------|
| **SAMEP** | Secure Agent Memory Exchange Protocol — enables multiple agents to share memories with HMAC-signed transfers. |
| **Agent Profile** | Per-agent configuration including learning preferences, decay rates, and SM-2 parameters. |
| **Strategy Bank** | A repository of learned strategies that agents can discover, evaluate, and apply. Uses a 5-phase strategy lifecycle. |
| **Knowledge Graph** | A semantic graph with spreading activation for discovering relationships between concepts and memories. |

## Infrastructure

| Term | Definition |
|------|-----------|
| **Qdrant** | An open-source vector database used for WARM and COLD tier vector search. |
| **Redis** | In-memory data store used for WARM tier caching, pub/sub streams, and inter-component communication. |
| **MCP** | Model Context Protocol — a standardized protocol for AI agent tool integration. MnemoCore exposes memory operations as MCP tools. |
| **Circuit Breaker** | A fault-tolerance pattern (via `pybreaker`) that temporarily stops requests to failing backends to prevent cascade failures. |

## Observability

| Term | Definition |
|------|-----------|
| **Prometheus** | Metrics collection system. MnemoCore exposes counters, histograms, and gauges at `/metrics`. |
| **Grafana** | Dashboard visualization. A pre-built dashboard is included in `grafana-dashboard.json`. |
| **Audit Trail** | A JSONL log of all subconscious AI decisions, stored at `data/subconscious_audit.jsonl`. |
| **Dream Report** | A JSON/Markdown report generated after each dream cycle, documenting what was consolidated, patterns found, and contradictions resolved. |

## Cognitive Enhancements

| Term | Definition |
|------|-----------|
| **Reconstructive Recall** | HDV-based memory reconstruction that synthesizes answers from partial matches and memory fragments. |
| **Context Optimizer** | Token-aware context prioritization that selects the most relevant memories to fit within LLM token limits. |
| **EFT** | Episodic Future Thinking — the ability to simulate future scenarios based on past episodic memories. Used for planning and prediction. |
| **Association Network** | A graph-based system tracking relationships between memories with typed edges and strength decay. |
| **Emotional Tag** | Valence-arousal-salience annotation on memories. Emotional memories decay slower and are prioritized during recall. |
| **Anticipatory Memory** | Predictions about future events, tracked with confidence scores and verification deadlines. |

## Safety & Security

| Term | Definition |
|------|-----------|
| **Dry Run** | A safety mode where the subconscious AI and self-improvement loops analyze but do not mutate memory state. Default: `true`. |
| **Safety Mode** | Self-improvement guard levels: `strict` (all 5 validation gates), `moderate` (3 gates), `permissive` (1 gate). |
| **Validation Gates** | Five checks that self-improvement proposals must pass: schema, consistency, performance, rollback, and human review. |
| **Beta Mode** | Safety nets for beta features: extra logging, conservative defaults, and dry-run enforcement. |

---

*See [ARCHITECTURE.md](ARCHITECTURE.md) for system design. See [CONFIGURATION.md](CONFIGURATION.md) for all settings.*
