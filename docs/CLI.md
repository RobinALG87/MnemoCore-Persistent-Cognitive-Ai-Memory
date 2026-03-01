# MnemoCore CLI Reference

> **Version**: 5.1.0 &nbsp;|&nbsp; **Entry Point**: `mnemocore` or `python -m mnemocore.cli.main`

The MnemoCore CLI provides command-line access to all core memory operations — storing, recalling, dreaming, exporting, and managing associative memories.

---

## Table of Contents

- [Installation](#installation)
- [Global Options](#global-options)
- [Core Commands](#core-commands)
  - [store](#store)
  - [recall](#recall)
  - [get](#get)
  - [delete](#delete)
  - [dream](#dream)
  - [stats](#stats)
  - [health](#health)
  - [export](#export)
- [Association Commands](#association-commands)
  - [bind](#bind)
  - [associative](#associative)
  - [associations find](#associations-find)
- [Concept Commands](#concept-commands)
  - [concepts define](#concepts-define)
  - [concepts inspect](#concepts-inspect)
- [Batch Commands](#batch-commands)
  - [batch store](#batch-store)
  - [batch export](#batch-export)
- [Output Formats](#output-formats)
- [Examples](#examples)

---

## Installation

The CLI is included with the MnemoCore package. After installation:

```bash
# Using the CLI entry point
mnemocore --help

# Or via Python module
python -m mnemocore.cli.main --help
```

---

## Global Options

| Flag | Short | Description |
|------|-------|-------------|
| `--config PATH` | `-c` | Path to `config.yaml` file |
| `--verbose` | `-v` | Enable verbose/debug output |
| `--data-dir PATH` | `-d` | Data directory path (default: `./data`) |
| `--help` | | Show help message |

```bash
mnemocore -c /path/to/config.yaml -v store "Hello, world"
```

---

## Core Commands

### `store`

Store a new memory.

```bash
mnemocore store "Birds can migrate thousands of miles"
```

| Option | Type | Description |
|--------|------|-------------|
| `--metadata TEXT` | JSON string | Metadata JSON: `'{"topic": "biology"}'` |
| `--tags TEXT` | Comma-separated | Tags: `"biology,migration"` |
| `--importance FLOAT` | 0.0–1.0 | Initial importance weight |
| `--category TEXT` | string | Memory category |
| `--json` | flag | Output result as JSON |

**Examples**:

```bash
# Store with metadata
mnemocore store "Python uses dynamic typing" \
  --metadata '{"language": "python"}' \
  --tags "programming,python" \
  --importance 0.8

# JSON output for scripting
mnemocore store "Test memory" --json
```

---

### `recall`

Search and recall memories using semantic similarity.

```bash
mnemocore recall "What do birds do?"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k INT` | integer | 5 | Number of results |
| `--min-score FLOAT` | float | 0.0 | Minimum similarity score |
| `--json` | flag | — | Output as JSON |
| `--show-content` | flag | — | Show full content (not truncated) |

**Examples**:

```bash
# Top 10 results with minimum score
mnemocore recall "machine learning" --top-k 10 --min-score 0.5

# Full content output
mnemocore recall "Python syntax" --show-content
```

---

### `get`

Retrieve a specific memory by ID.

```bash
mnemocore get mem_abc123
```

| Option | Type | Description |
|--------|------|-------------|
| `--json` | flag | Output as JSON |

---

### `delete`

Delete a memory by ID.

```bash
mnemocore delete mem_abc123
```

| Option | Type | Description |
|--------|------|-------------|
| `--force` | flag | Skip confirmation prompt |

---

### `dream`

Trigger a dream consolidation session. Dreams cluster related memories, extract patterns, resolve contradictions, and promote important memories.

```bash
mnemocore dream --now
```

| Option | Type | Description |
|--------|------|-------------|
| `--now` | flag | Run dream immediately |
| `--report-path PATH` | path | Save dream report to file |
| `--json` | flag | Output result as JSON |

**Example**:

```bash
mnemocore dream --now --report-path ./dream_report.json --json
```

---

### `stats`

Show system statistics.

```bash
mnemocore stats
```

| Option | Type | Description |
|--------|------|-------------|
| `--json` | flag | Output as JSON |
| `--tier TEXT` | string | Show stats for specific tier (`hot`/`warm`/`cold`) |

---

### `health`

Check system health and connectivity.

```bash
mnemocore health
```

---

### `export`

Export memories to a file.

```bash
mnemocore export --format json > backup.json
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format TEXT` | `json`/`jsonl` | `json` | Export format |
| `--output PATH` | path | stdout | Output file path |
| `--collection TEXT` | string | — | Filter by collection |
| `--limit INT` | integer | — | Max memories to export |
| `--include-vectors` | flag | — | Include vector data |

---

## Association Commands

### `bind`

Create a synaptic connection between two memories.

```bash
mnemocore bind mem_abc123 mem_def456 --strength 0.8
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--strength FLOAT` | 0.0–1.0 | 1.0 | Connection strength |
| `--json` | flag | — | Output as JSON |

---

### `associative`

Perform an associative query following synaptic connections.

```bash
mnemocore associative "related concept"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k INT` | integer | 5 | Number of results |
| `--json` | flag | — | Output as JSON |

---

### `associations find`

Find associated memories following synaptic connections from a root memory.

```bash
mnemocore associations find mem_abc123
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--depth INT` | integer | 2 | Max traversal depth |
| `--limit INT` | integer | 10 | Max results |
| `--json` | flag | — | Output as JSON |

---

## Concept Commands

### `concepts define`

Define a conceptual symbol.

```bash
mnemocore concepts define "bird" --examples "sparrow,eagle,penguin"
```

| Option | Type | Description |
|--------|------|-------------|
| `--examples TEXT` | Comma-separated | Example instances |
| `--json` | flag | Output as JSON |

---

### `concepts inspect`

Inspect a concept or find related concepts.

```bash
mnemocore concepts inspect "bird"
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k INT` | integer | 5 | Related concepts |
| `--json` | flag | — | Output as JSON |

---

## Batch Commands

### `batch store`

Batch store memories from a file.

```bash
mnemocore batch store memories.json --format json
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format TEXT` | `json`/`jsonl` | auto-detect | Input file format |
| `--json` | flag | — | Output summary as JSON |

**Input format** (JSON array):

```json
[
  {"content": "First memory", "metadata": {"source": "import"}},
  {"content": "Second memory", "tags": ["test"]}
]
```

---

### `batch export`

Batch export memories matching a query.

```bash
mnemocore batch export "programming" --format jsonl --top-k 100
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format TEXT` | `json`/`jsonl` | `json` | Output format |
| `--top-k INT` | integer | 50 | Max memories |

---

## Output Formats

All commands support `--json` for machine-readable output. By default, the CLI uses formatted table output with colors.

### Table Output (default)

```
┌──────────────────────────────────────────────────────────┐
│  MnemoCore Query Results                                  │
├──────────┬──────────────────────────────┬───────┬────────┤
│ ID       │ Content                      │ Score │ Tier   │
├──────────┼──────────────────────────────┼───────┼────────┤
│ mem_a123 │ Birds can migrate thousan... │ 0.87  │ hot    │
│ mem_b456 │ Salmon swim upstream to...  │ 0.72  │ warm   │
└──────────┴──────────────────────────────┴───────┴────────┘
```

### JSON Output (`--json`)

```json
{
  "ok": true,
  "results": [
    {"id": "mem_a123", "content": "Birds can migrate...", "score": 0.87, "tier": "hot"}
  ]
}
```

---

## Examples

### Quick Start

```bash
# Store some memories
mnemocore store "Python is a dynamically typed language"
mnemocore store "Rust uses a borrow checker for memory safety"
mnemocore store "Go has built-in concurrency with goroutines"

# Query
mnemocore recall "memory safe languages" --top-k 3

# Trigger dream to consolidate
mnemocore dream --now

# Check stats
mnemocore stats
```

### Scripting Workflow

```bash
# Store and capture ID
MEMORY_ID=$(mnemocore store "Important fact" --json | python -c "import sys,json; print(json.load(sys.stdin)['memory_id'])")

# Retrieve by ID
mnemocore get $MEMORY_ID --json

# Export all memories as backup
mnemocore export --format jsonl --output backup.jsonl

# Batch import
mnemocore batch store backup.jsonl --format jsonl
```

### Multi-Agent Use

```bash
# Store with agent context
mnemocore store "Agent-specific insight" --metadata '{"agent_id": "agent-01"}'

# Query within agent scope
mnemocore recall "insight" --top-k 5
```

---

*See [API.md](API.md) for the full REST API reference. See [CONFIGURATION.md](CONFIGURATION.md) for all config options.*
