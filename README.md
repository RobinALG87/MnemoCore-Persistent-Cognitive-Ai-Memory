# MnemoCore: Infrastructure for Persistent Cognitive Memory (Phase 3.0+)

[![Veristate Systems](https://img.shields.io/badge/Veristate-Sovereign%20Intelligence-blue.svg)](https://veristatesystems.com)
[![Status](https://img.shields.io/badge/Status-Public%20Dev%20Preview-orange.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Binary%20VSA%2FHDC-green.svg)]()

> "The limitation of modern AI isn't logic, it's memory. MnemoCore is the architecture for agents that don't just process data, but evolve through experience."

**MnemoCore** is a production-grade cognitive memory engine built on Hyperdimensional Computing (HDC) and Vector Symbolic Architecture (VSA). It is designed to provide autonomous agents with a coherent, persistent, and indefinitely scalable long-term memory.

---

## 🧠 Why MnemoCore?

Traditional RAG (Retrieval-Augmented Generation) is often slow, expensive, and lacks biological plausibility. MnemoCore solves the **Scalability vs. Agency** paradox by treating memories as high-dimensional holographic bitstreams.

### Core Innovations:
*   **🚀 16,384-D Binary VSA:** High-entropy bit-vectors allow for O(1) binding and bundling operations. 100x faster similarity search using hardware-native bitwise XOR and `popcount`.
*   **🧬 Biological LTP (Long-Term Potentiation):** Memory strength isn't static. MnemoCore implements synaptic plasticity where memories are reinforced by access and decay organically over time.
*   **🧊 Tri-State Storage (Memory Tiering):**
    *   **HOT (RAM):** Zero-latency access to current working context.
    *   **WARM (Qdrant/SSD):** Sub-millisecond semantic search across millions of vectors via Binary Quantization.
    *   **COLD (Archive):** Indefinite long-term storage for deep historical analysis.
*   **🔮 Active Inference:** Predictive retrieval that uses the agent's current "state of mind" to anticipate relevant context before it's explicitly queried.

---

## 🏗️ Architecture

MnemoCore is composed of several decoupled layers that work in symbiosis:

*   **`src/core`**: The VSA engine. Handles `BinaryHDV` operations, Synaptic connections, and the Tier Manager.
*   **`src/subconscious`**: A background daemon that performs "Dreaming" (synaptic strengthening) and memory consolidation during idle cycles.
*   **`src/api`**: A high-performance FastAPI wrapper for integration with existing LLM agent frameworks.
*   **`src/nightlab`**: An orchestration layer for autonomous self-improvement and architectural audits.

---

## 🚦 Getting Started

### 1. Prerequisites
*   Python 3.10+
*   Docker (for Qdrant & Redis integration)
*   32GB RAM recommended (for large-scale HOT tiering)

### 2. Installation
```bash
git clone https://github.com/RobinALG87/MnemoCore-Infrastructure-for-Persistent-Cognitive-Memory.git
cd MnemoCore-Infrastructure-for-Persistent-Cognitive-Memory
pip install -r requirements.txt
```

### 3. Launching the Cognitive Stack
```bash
# Start the infrastructure (Qdrant, Redis, Grafana)
docker-compose up -d

# Start the Memory API
uvicorn src.api.main:app --port 8100

# Start the Subconscious Worker
python src/subconscious/daemon.py
```

---

## 📊 Monitoring & Auditing

MnemoCore includes a pre-configured **Grafana Dashboard** (`grafana-dashboard.json`) to visualize:
*   **Synaptic Density:** The growth of connections between concepts.
*   **LTP Decay Curves:** Real-time visualization of memory permanence.
*   **Tier Distribution:** Monitoring the flow between Hot, Warm, and Cold tiers.

---

## ⚖️ License

This version of MnemoCore is released under the **Business Source License 1.1 (BSL)**.
*   **For Non-Commercial/Research:** Free to use, modify, and distribute.
*   **For Commercial Use:** Requires a commercial license from Veristate Systems AB.
*   **Sunset:** Becomes Open Source (MIT) efter 3 år (2029-02-15).

*See `LICENSE` for full details.*

---

## 🤝 Join the Expansion

We are building the substrate for Sovereign Intelligence. 
*   **Vision:** [veristatesystems.com](https://veristatesystems.com)

Ω
