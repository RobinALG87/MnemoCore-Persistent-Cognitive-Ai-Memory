# MnemoCore — Persistent Cognitive Memory

You have access to a persistent memory system via MCP tools:
- `memory_query` — search for relevant memories before starting any task
- `memory_store` — save important decisions, findings, and bug fixes after completing work
- `memory_stats` / `memory_health` — check system status

## When to use memory

**At session start:** Call `memory_query` with the user's first message to retrieve relevant past context.

**After completing a task:** Call `memory_store` to record:
- What was changed and why (key architectural decisions)
- Bug fixes and root causes
- Non-obvious patterns discovered in the codebase
- User preferences and project conventions

**When you find something unexpected:** Store it immediately with relevant tags.

## Storing memories

Include useful metadata:
```json
{
  "content": "Fixed async race condition in tier_manager.py by adding asyncio.Lock around promotion logic",
  "metadata": {
    "source": "claude-code",
    "tags": ["bugfix", "async", "tier_manager"],
    "project": "mnemocore"
  }
}
```

## Rules
- Do NOT store trivial information (e.g., "the user asked me to open a file")
- DO store non-obvious insights, decisions with reasoning, and recurring patterns
- Query memory BEFORE reading files when working on a known codebase
- Store memory AFTER completing non-trivial changes
