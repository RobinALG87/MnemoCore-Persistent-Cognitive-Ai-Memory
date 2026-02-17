# MnemoCore: Infrastructure for Persistent Cognitive Memory (Phase 3.0+)

**Holographic Active Inference Memory (HAIM) - Distributed Vector System**

![Status](https://img.shields.io/badge/Status-Beta-orange)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)

---

## 📖 Introduction

**MnemoCore** is an advanced cognitive memory infrastructure designed to provide persistent, scalable, and biologically-inspired memory for AI agents. It leverages **Binary Hyperdimensional Computing (HDC)** and **Vector Symbolic Architectures (VSA)** to create a robust, noise-tolerant, and distributed memory system.

Unlike traditional vector databases that simply store embeddings, MnemoCore implements a **Holographic Active Inference Memory (HAIM)** engine. This engine not only retrieves information based on semantic similarity but also:
*   **Predicts** future states using Active Inference.
*   **Consolidates** memories through "subconscious" processing (dreaming).
*   **Evaluates** novelty using Epistemic Information Gain (EIG).
*   **Reasons** conceptually via analogy and cross-domain inference.

MnemoCore is designed for **infinite scalability** (Phase 3.5+), targeting 1B+ memories with sub-10ms latency using a tiered architecture (HOT/WARM/COLD) and distributed consensus.

---

## 🚀 Key Features

*   **Binary HDV/VSA Core**:
    *   Utilizes **16,384-dimensional binary vectors** for high efficiency and noise resilience.
    *   Fast bitwise operations (XOR binding, Hamming distance) optimized for modern hardware (CPU/GPU).
*   **Tiered Memory Architecture**:
    *   **🔥 HOT Tier**: In-memory (RAM) for ultra-low latency access to recent and high-salience memories.
    *   **sqr WARM Tier**: Redis/mmap-backed storage for medium-term retention and rapid retrieval.
    *   **❄️ COLD Tier**: Disk/S3 archival for massive long-term storage (Phase 3.5).
*   **Active Inference & EIG**:
    *   Calculates **Epistemic Information Gain (EIG)** to prioritize storing novel and surprising information.
    *   Drives curiosity and exploration in autonomous agents.
*   **Conceptual Reasoning ("The Soul")**:
    *   A dedicated conceptual layer that enables **analogical reasoning** (A is to B as C is to ?).
    *   Supports cross-domain inference and abstraction.
*   **Subconscious Processing**:
    *   Background "dreaming" processes that strengthen synaptic connections between related memories.
    *   Automatic consolidation and cleanup of low-value memories.
*   **Async & Distributed**:
    *   Fully asynchronous API built with **FastAPI** and **Redis Streams**.
    *   Designed for distributed deployment with horizontal scaling (Phase 3.5).

---

## 🏗️ Architecture

MnemoCore is built on a modular architecture:

1.  **API Gateway (FastAPI)**: Handles REST requests, authentication, and rate limiting.
2.  **HAIM Engine**: The core cognitive engine managing memory lifecycle, encoding, and retrieval.
3.  **Tier Manager**: Orchestrates data movement between HOT, WARM, and COLD tiers based on access frequency and importance (LTP - Long-Term Potentiation).
4.  **Vector Core**: Handles HDV operations (encoding, binding, bundling, superposition).
5.  **Storage Layer**:
    *   **Redis**: Metadata, WARM tier index, and Pub/Sub events.
    *   **Qdrant** (Optional/Phase 3.5): Vector storage for massive scale.
    *   **FileSystem/S3**: Persistence for COLD tier and backups.

---

## 🛠️ Installation

### Prerequisites

*   **Python 3.10+**
*   **Redis** (Required for WARM tier and async features)
*   **Qdrant** (Optional, for scalable vector search in Phase 3.5)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/veristatesystems/mnemocore.git
    cd mnemocore
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables (Optional):**
    Copy `config.yaml` and modify as needed. You can override settings via environment variables (e.g., `HAIM_REDIS_URL`).

---

## ⚙️ Configuration

MnemoCore is configured via `config.yaml`. Key sections include:

*   **`haim.dimensionality`**: Vector dimension (default: 16384).
*   **`haim.tiers`**: Configuration for HOT, WARM, and COLD tiers (max size, LTP thresholds).
*   **`haim.redis`**: Redis connection details.
*   **`haim.qdrant`**: Qdrant connection details (Phase 3.5).
*   **`haim.security`**: API Key configuration.

Example `config.yaml` snippet:
```yaml
haim:
  version: "3.0"
  dimensionality: 16384
  tiers:
    hot:
      max_memories: 2000
    warm:
      max_memories: 100000
  redis:
    url: "redis://localhost:6379/0"
```

---

## 💻 Usage

### Python Library

You can use the `HAIMEngine` directly in your Python applications:

```python
from src.core.engine import HAIMEngine

# Initialize the engine
engine = HAIMEngine()

# Store a memory
memory_id = engine.store(
    content="The quick brown fox jumps over the lazy dog.",
    metadata={"source": "book", "author": "unknown"}
)
print(f"Stored memory: {memory_id}")

# Query memories
results = engine.query("What did the fox do?", top_k=3)
for mem_id, score in results:
    memory = engine.get_memory(mem_id)
    print(f"Match ({score:.4f}): {memory.content}")

# Close the engine
engine.close()
```

### REST API

Start the API server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8100 --reload
```

**Store a Memory:**
```bash
curl -X POST "http://localhost:8100/store" \
     -H "X-API-Key: mnemocore-beta-key" \
     -H "Content-Type: application/json" \
     -d '{"content": "Artificial Intelligence is evolving.", "metadata": {"tag": "AI"}}'
```

**Query Memories:**
```bash
curl -X POST "http://localhost:8100/query" \
     -H "X-API-Key: mnemocore-beta-key" \
     -H "Content-Type: application/json" \
     -d '{"query": "AI evolution", "top_k": 5}'
```

**Solve Analogy:**
```bash
curl -X POST "http://localhost:8100/analogy" \
     -H "X-API-Key: mnemocore-beta-key" \
     -H "Content-Type: application/json" \
     -d '{"source_concept": "king", "source_value": "man", "target_concept": "woman"}'
# Expected result: "queen" (if concepts are learned)
```

---

## 🤖 Model Context Protocol (MCP)

MnemoCore provides an MCP server implementation, allowing seamless integration with AI agents like **Claude Desktop**, **Gemini**, and others.

### Features
*   **Tools**: Store and query memories, perform analogical reasoning, and delete memories.
*   **Resources**: Access recent memories (`mnemocore://memories/recent`) and engine stats (`mnemocore://stats`).
*   **Prompts**: Built-in prompts for recalling information and "dreaming" (subconscious processing).

### Running the MCP Server

You can run the MCP server directly using the launcher script:

```bash
python run_mcp.py
```

### Claude Desktop Configuration

To use MnemoCore as a memory tool for Claude Desktop, add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mnemocore": {
      "command": "python",
      "args": [
        "/absolute/path/to/mnemocore/run_mcp.py"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/mnemocore"
      }
    }
  }
}
```

Make sure to replace `/absolute/path/to/mnemocore` with the actual path to your cloned repository.

---

## 🗺️ Roadmap (Phase 3.5+)

*   **[Phase 3.5] Distributed Vector Database**: Full integration with **Qdrant** for billion-scale memory support.
*   **[Phase 3.5] GPU Acceleration**: Implement CUDA kernels for massive parallel bitwise operations (XOR/Popcount).
*   **[Phase 3.5] Distributed Consensus**: Federated holographic state across multiple nodes.
*   **[Phase 4.0] Neural-Symbolic Interface**: Deeper integration with LLMs for improved semantic understanding.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add some amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

Please ensure all tests pass (`pytest`) before submitting.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For inquiries, research collaborations, or support, please reach out to:

**Robin Granberg**
Email: Robin@veristatesystems.com

---
*MnemoCore is a research project by Veristate Systems, pushing the boundaries of cognitive AI memory.*
