# MnemoCore — Persistent Cognitive Memory

You have access to a persistent memory system via the MnemoCore REST API at `$MNEMOCORE_URL` (default: `http://localhost:8100`).

## Querying memory

To recall relevant context, call the API at the start of a task:

```bash
curl -s -X POST "$MNEMOCORE_URL/query" \
  -H "X-API-Key: $HAIM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "DESCRIBE_TASK_HERE", "top_k": 5}'
```

## Storing memory

After completing significant work, store a memory:

```bash
curl -s -X POST "$MNEMOCORE_URL/store" \
  -H "X-API-Key: $HAIM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "WHAT_WAS_DONE_AND_WHY",
    "metadata": {"source": "gemini-cli", "tags": ["relevant", "tags"]}
  }'
```

## Guidelines

- **Query before starting** any non-trivial task on a known codebase
- **Store after completing** important changes, bug fixes, or design decisions
- **Do NOT store** trivial or ephemeral information
- Include relevant tags: language, component, type (bugfix/feature/refactor)
