# MnemoCore API Reference (Beta)

## Beta Notice

API contracts may change during beta without backward compatibility guarantees.
Use pinned commits if you need reproducibility.

## Base URL

Default local API URL:
- `http://localhost:8100`

## Endpoints

### `GET /`
Basic service status.

### `GET /health`
Returns health status, Redis connectivity, and engine stats.

### `POST /store`
Store a memory.

Request body:
```json
{
  "content": "string",
  "metadata": {"key": "value"},
  "agent_id": "optional-string",
  "ttl": 3600
}
```

### `POST /query`
Query semantic memory.

Request body:
```json
{
  "query": "string",
  "top_k": 5,
  "agent_id": "optional-string"
}
```

### `GET /memory/{memory_id}`
Fetch a memory by ID (Redis-first, engine fallback).

### `DELETE /memory/{memory_id}`
Delete a memory by ID.

### `POST /concept`
Define a concept for conceptual memory operations.

### `POST /analogy`
Run analogy inference.

### `GET /stats`
Return engine statistics.

### `GET /metrics`
Prometheus metrics endpoint.

## Example Requests

Store:
```bash
curl -X POST http://localhost:8100/store \
  -H "Content-Type: application/json" \
  -d '{"content":"Birds can migrate long distances"}'
```

Query:
```bash
curl -X POST http://localhost:8100/query \
  -H "Content-Type: application/json" \
  -d '{"query":"animal migration","top_k":3}'
```

## Error Behavior

- `404` for missing memory IDs.
- In degraded infrastructure modes, API may still return successful core operations while external storage writes fail.

## Compatibility Guidance

During beta, treat responses as evolving contracts:
- Parse defensively.
- Avoid rigid coupling to optional fields.
- Revalidate after version upgrades.
