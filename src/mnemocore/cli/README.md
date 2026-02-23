# MnemoCore CLI

Command-line interface for MnemoCore persistent cognitive memory system.

## Installation

After installing MnemoCore, the CLI is available as `mnemocore-cli`:

```bash
pip install mnemocore
mnemocore-cli --help
```

## Basic Commands

### Store a Memory

Store text memories with optional metadata and tags:

```bash
# Simple storage
mnemocore-cli store "Robin gillar Python programmering"

# With tags
mnemocore-cli store "Robin gillar Python" -t programming -t python

# With category and importance
mnemocore-cli store "Important meeting tomorrow" -c reminders -i 0.9

# With custom metadata (JSON)
mnemocore-cli store "Meeting notes" -m '{"attendees": ["Alice", "Bob"]}'

# Output as JSON
mnemocore-cli store "Test memory" --json
```

### Recall/Search Memories

Search stored memories by semantic similarity:

```bash
# Basic search
mnemocore-cli recall "vad gillar Robin?"

# Get more results
mnemocore-cli recall "Python" -k 10

# Filter by minimum score
mnemocore-cli recall "programming" -s 0.5

# Show full content
mnemocore-cli recall "meeting" -C

# JSON output
mnemocore-cli recall "Python" --json
```

### Dream Session

Trigger a dream session for memory consolidation:

```bash
# Run dream session immediately
mnemocore-cli dream --now

# Save dream report
mnemocore-cli dream --now --report-path dream_report.json

# JSON output
mnemocore-cli dream --now --json
```

Dream sessions perform:
- Episodic clustering
- Pattern extraction
- Recursive synthesis
- Contradiction resolution
- Semantic promotion

### Statistics

View system statistics:

```bash
# Show all stats
mnemocore-cli stats

# Filter by tier
mnemocore-cli stats -t hot
mnemocore-cli stats -t warm

# JSON output
mnemocore-cli stats --json
```

### Export Memories

Export memories to file:

```bash
# Export all to JSON
mnemocore-cli export -f json -o backup.json

# Export to JSONL
mnemocore-cli export -f jsonl -o backup.jsonl

# Export specific tier
mnemocore-cli export -f json -o hot_backup.json -c hot

# Limit export
mnemocore-cli export -f json -o sample.json -l 100

# Include vector embeddings
mnemocore-cli export -f json -o full_backup.json --include-vectors
```

### Health Check

Check system health:

```bash
mnemocore-cli health
```

### Memory Operations

Get, delete, and bind memories:

```bash
# Get a memory by ID
mnemocore-cli get mem_abc123

# Delete a memory (with confirmation)
mnemocore-cli delete mem_abc123

# Force delete without confirmation
mnemocore-cli delete mem_abc123 --force

# Bind two memories together
mnemocore-cli bind mem_abc123 mem_def456 --strength 0.8
```

## Advanced Commands

### Associations

Find related memories through synaptic connections:

```bash
# Find associations
mnemocore-cli associations find mem_abc123 --depth 2

# Limit results
mnemocore-cli associations find mem_abc123 -d 2 -l 10
```

### Concepts

Define and inspect conceptual symbols:

```bash
# Define a concept
mnemocore-cli concepts define "programming" "Writing code to solve problems"

# Define with examples
mnemocore-cli concepts define "Python" "A programming language" -e "fast" -e "dynamic"

# Inspect a concept
mnemocore-cli concepts inspect "programming"
```

### Batch Operations

Perform batch operations on memories:

```bash
# Batch store from file (JSONL)
mnemocore-cli batch store memories.jsonl

# Batch export matching a query
mnemocore-cli batch export "Python" python_memories.jsonl
```

### Associative Query

Search with associative memory:

```bash
mnemocore-cli associative "Python programming tools"
```

## Options

### Global Options

- `--config, -c`: Path to config.yaml file
- `--verbose, -v`: Enable verbose output
- `--data-dir, -d`: Data directory path (default: ./data)
- `--help`: Show help message

### Output Formats

Most commands support `--json` flag for JSON output instead of formatted text.

## Examples

### Complete Workflow

```bash
# Store some memories
mnemocore-cli store "I love programming in Python" -t programming
mnemocore-cli store "Python is great for data science" -t datascience

# Search for them
mnemocore-cli recall "Python programming"

# Bind related memories
mnemocore-cli bind mem_abc123 mem_def456

# Check system health
mnemocore-cli health

# Run consolidation
mnemocore-cli dream --now

# Export backup
mnemocore-cli export -f jsonl -o backup.jsonl
```

## File Formats

### JSONL Import Format

For batch import, use JSONL format (one JSON object per line):

```jsonl
{"content": "First memory", "metadata": {"tags": ["important"]}}
{"content": "Second memory", "metadata": {"category": "work"}}
```

### Export Format

Exported JSON/JSONL files include:

```json
{
  "id": "mem_abc123",
  "content": "Memory content here",
  "created_at": "2024-01-01T12:00:00+00:00",
  "tier": "hot",
  "ltp_strength": 0.75,
  "access_count": 5,
  "metadata": {
    "tags": ["tag1", "tag2"]
  }
}
```
