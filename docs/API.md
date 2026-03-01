# MnemoCore API Reference

> **Version**: 2.0.0 &nbsp;|&nbsp; **Base URL**: `http://localhost:8100` &nbsp;|&nbsp; **Phase**: Beta

---

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Security Headers](#security-headers)
- [Health & Stats](#1-health--stats)
- [Memory Operations](#2-memory-operations)
- [Episodic Memory](#3-episodic-memory)
- [Working Memory](#4-working-memory--observations)
- [Dream Loop](#5-dream-loop)
- [Export](#6-export)
- [Procedural Memory](#7-procedural-memory)
- [Predictions](#8-predictions)
- [Concepts & Analogy](#9-concepts--analogy)
- [Meta Memory & Self-Improvement](#10-meta-memory--self-improvement)
- [Maintenance](#11-maintenance)
- [Subtle Thoughts](#12-subtle-thoughts)
- [Recursive Synthesis (RLM)](#13-recursive-synthesis-rlm)
- [Trust & Provenance](#14-trust--provenance)
- [Proactive Recall](#15-proactive-recall)
- [Contradictions](#16-contradictions)
- [Emotional Tags](#17-emotional-tags)
- [Knowledge Gaps](#18-knowledge-gaps)
- [Association Network](#19-association-network)
- [Prometheus Metrics](#prometheus-metrics)

---

## Authentication

All endpoints except Health & Stats require an API key via the `X-API-Key` header.

```bash
curl -H "X-API-Key: YOUR_KEY" http://localhost:8100/store ...
```

Set the key via environment variable:

```bash
export HAIM_API_KEY="your-secret-key"
```

If no key is configured, authentication is disabled (development mode only — not recommended for production).

---

## Rate Limiting

Rate limits are applied per-client based on endpoint category:

| Category | Requests / Minute |
|----------|-------------------|
| Store    | 100               |
| Query    | 500               |
| Dream    | 5                 |
| Concept  | 100               |

Exceeding the limit returns `429 Too Many Requests`. Inspect current limits via `GET /rate-limits`.

---

## Error Handling

All error responses follow a consistent structure:

```json
{
  "ok": false,
  "error": "Human-readable error description",
  "detail": "Optional technical detail"
}
```

| Status Code | Meaning                          |
|-------------|----------------------------------|
| `400`       | Invalid request (validation)     |
| `401`       | Missing or invalid API key       |
| `404`       | Resource not found               |
| `429`       | Rate limit exceeded              |
| `500`       | Internal server error            |

---

## Security Headers

Every response includes hardened security headers:

| Header                          | Value                                   |
|---------------------------------|-----------------------------------------|
| `X-Content-Type-Options`        | `nosniff`                               |
| `X-Frame-Options`               | `DENY`                                  |
| `X-XSS-Protection`             | `1; mode=block`                         |
| `Content-Security-Policy`       | `default-src 'self'`                    |
| `Referrer-Policy`               | `strict-origin-when-cross-origin`       |
| `Strict-Transport-Security`     | `max-age=31536000; includeSubDomains`   |
| `X-Trace-ID`                    | Unique trace identifier per request     |

---

## Endpoints

### 1. Health & Stats

These endpoints do **not** require authentication.

#### `GET /`

Returns basic service information.

```json
{
  "status": "ok",
  "service": "MnemoCore",
  "version": "2.0.0",
  "phase": "Phase 6",
  "timestamp": "2026-02-28T12:00:00Z"
}
```

#### `GET /health`

Returns system health including backend connectivity.

```json
{
  "status": "healthy",
  "redis_connected": true,
  "storage_circuit_breaker": "closed",
  "qdrant_circuit_breaker": "closed",
  "engine_ready": true,
  "timestamp": "2026-02-28T12:00:00Z"
}
```

Status values: `healthy`, `degraded`.

#### `GET /stats`

Returns engine aggregate statistics (memory counts per tier, LTP distributions, etc.).

#### `GET /rate-limits`

Returns the current rate limit configuration.

```json
{
  "limits": {
    "Store": { "requests": 100, "window_seconds": 60, "requests_per_minute": 100 },
    "Query": { "requests": 500, "window_seconds": 60, "requests_per_minute": 500 },
    "Dream": { "requests": 5, "window_seconds": 60, "requests_per_minute": 5 }
  }
}
```

---

### 2. Memory Operations

#### `POST /store`

Store a new memory.

**Request Body** (`StoreRequest`):

| Field      | Type     | Required | Constraints          | Description                          |
|------------|----------|----------|----------------------|--------------------------------------|
| `content`  | `string` | Yes      | max 100,000 chars    | Memory text content                  |
| `metadata` | `object` | No       | max 50 keys          | Arbitrary key-value metadata         |
| `agent_id` | `string` | No       | max 256 chars        | Agent identifier for multi-agent use |
| `ttl`      | `int`    | No       | 1–31,536,000 seconds | Time-to-live (auto-expires)          |

**Response** (`StoreResponse`):

```json
{
  "ok": true,
  "memory_id": "mem_abc123",
  "message": "Memory stored successfully"
}
```

**Example**:

```bash
curl -X POST http://localhost:8100/store \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "content": "Birds can migrate thousands of miles using magnetic fields",
    "metadata": {"topic": "biology", "source": "textbook"},
    "agent_id": "agent-01"
  }'
```

#### `POST /query`

Semantic search across all memory tiers.

**Request Body** (`QueryRequest`):

| Field      | Type     | Required | Default | Constraints | Description              |
|------------|----------|----------|---------|-------------|--------------------------|
| `query`    | `string` | Yes      | —       | max 10,000  | Natural language query   |
| `top_k`    | `int`    | No       | 5       | 1–100       | Number of results        |
| `agent_id` | `string` | No       | —       | max 256     | Filter by agent          |

**Response** (`QueryResponse`):

```json
{
  "ok": true,
  "query": "animal migration",
  "results": [
    {
      "id": "mem_abc123",
      "content": "Birds can migrate thousands of miles...",
      "score": 0.87,
      "metadata": {"topic": "biology"},
      "tier": "hot"
    }
  ]
}
```

**Example**:

```bash
curl -X POST http://localhost:8100/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"query": "animal migration patterns", "top_k": 3}'
```

#### `GET /memory/{memory_id}`

Retrieve a specific memory by ID.

**Response**:

```json
{
  "source": "hot",
  "id": "mem_abc123",
  "content": "Birds can migrate thousands of miles...",
  "metadata": {"topic": "biology"},
  "created_at": "2026-02-28T12:00:00Z",
  "epistemic_value": 0.75,
  "ltp_strength": 0.85,
  "tier": "hot"
}
```

#### `DELETE /memory/{memory_id}`

Delete a memory by ID.

**Response**: `{"ok": true, "deleted": true}`

---

### 3. Episodic Memory

#### `POST /episodes/start`

Start a new episodic memory chain for an agent.

**Request Body** (`EpisodeStartRequest`):

| Field      | Type     | Required | Description              |
|------------|----------|----------|--------------------------|
| `agent_id` | `string` | Yes      | Agent identifier          |
| `goal`     | `string` | Yes      | Episode goal/objective    |
| `context`  | `string` | No       | Additional context        |

**Response**: `{"ok": true, "episode_id": "ep_xyz789"}`

---

### 4. Working Memory / Observations

#### `POST /wm/observe`

Add an observation to an agent's working memory.

**Request Body** (`ObserveRequest`):

| Field        | Type       | Required | Default         | Description           |
|-------------|------------|----------|-----------------|------------------------|
| `agent_id`  | `string`   | Yes      | —               | Agent identifier        |
| `content`   | `string`   | Yes      | —               | Observation content     |
| `kind`      | `string`   | No       | `"observation"` | Observation type        |
| `importance`| `float`    | No       | `0.5`           | Importance weight (0–1) |
| `tags`      | `string[]` | No       | `[]`            | Tags for categorization |

**Response**: `{"ok": true, "item_id": "wm_item_001"}`

#### `GET /wm/context/{agent_id}`

Get the current working memory context for an agent.

**Query Parameters**: `limit` (int, default 16)

**Response**:

```json
{
  "ok": true,
  "items": [
    {"id": "wm_item_001", "content": "User asked about...", "kind": "observation", "importance": 0.8}
  ]
}
```

---

### 5. Dream Loop

#### `POST /dream`

Trigger a consolidation dream cycle. Dreams cluster, extract patterns, resolve contradictions, and promote important memories.

**Request Body** (`DreamRequest`):

| Field           | Type   | Required | Default | Constraints | Description                |
|----------------|--------|----------|---------|-------------|----------------------------|
| `max_cycles`   | `int`  | No       | 1       | 1–10        | Number of dream cycles     |
| `force_insight`| `bool` | No       | `false` | —           | Force insight generation   |

**Response** (`DreamResponse`):

```json
{
  "ok": true,
  "cycles_completed": 1,
  "insights_generated": 3,
  "concepts_extracted": 2,
  "parallels_found": 1,
  "memories_processed": 47,
  "message": "Dream cycle completed"
}
```

---

### 6. Export

#### `GET /export`

Export memories in JSON or JSONL format.

**Query Parameters**:

| Parameter          | Type     | Default  | Description                              |
|-------------------|----------|----------|------------------------------------------|
| `agent_id`        | `string` | —        | Filter by agent                           |
| `tier`            | `string` | —        | Filter: `hot`, `warm`, `cold`, `soul`    |
| `limit`           | `int`    | 100      | Max memories to export                    |
| `include_metadata`| `bool`   | `true`   | Include metadata in output                |
| `format`          | `string` | `"json"` | Output format: `json` or `jsonl`         |

**Response** (`ExportResponse`):

```json
{
  "ok": true,
  "count": 42,
  "format": "json",
  "memories": [
    {
      "id": "mem_abc123",
      "content": "...",
      "created_at": "2026-02-28T12:00:00Z",
      "ltp_strength": 0.85,
      "tier": "hot",
      "metadata": {}
    }
  ]
}
```

---

### 7. Procedural Memory

#### `GET /procedures/search`

Search the procedural skill library.

| Parameter  | Type     | Default | Description       |
|-----------|----------|---------|-------------------|
| `query`   | `string` | —       | Search query (req) |
| `agent_id`| `string` | —       | Filter by agent    |
| `top_k`   | `int`    | 5       | Number of results  |

**Response**: `{"ok": true, "procedures": [...]}`

#### `POST /procedures/{proc_id}/feedback`

Report success/failure for a procedure execution.

**Request Body**: `{"success": true}`

**Response**: `{"ok": true, "procedure_id": "proc_001", "success_recorded": true}`

---

### 8. Predictions

#### `POST /predictions`

Create a new prediction (anticipatory memory).

| Field               | Type       | Required | Default | Description                 |
|--------------------|------------|----------|---------|------------------------------|
| `content`          | `string`   | Yes      | —       | Prediction text              |
| `confidence`       | `float`    | No       | `0.5`   | Confidence level (0–1)       |
| `deadline_days`    | `float`    | No       | —       | Days until verification      |
| `related_memory_ids`| `string[]`| No       | `[]`    | Related memory IDs           |
| `tags`             | `string[]` | No       | `[]`    | Classification tags          |

**Response**: `{"ok": true, "prediction": {...}}`

#### `GET /predictions`

List predictions. Optional filter: `?status=pending`

#### `POST /predictions/{pred_id}/verify`

Verify a prediction outcome.

**Request Body**: `{"success": true, "notes": "Prediction was accurate"}`

---

### 9. Concepts & Analogy

#### `POST /concept`

Define a conceptual symbol for analogy and reasoning operations.

| Field        | Type             | Required | Constraints | Description          |
|-------------|------------------|----------|-------------|----------------------|
| `name`      | `string`         | Yes      | max 256     | Concept name          |
| `attributes`| `dict[str, str]` | Yes      | 1–50 keys   | Attribute definitions |

**Example**:

```bash
curl -X POST http://localhost:8100/concept \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"name": "bird", "attributes": {"can_fly": "true", "has_feathers": "true"}}'
```

#### `POST /analogy`

Solve an analogy: "A is to X as B is to ?"

| Field            | Type     | Required | Description           |
|-----------------|----------|----------|-----------------------|
| `source_concept`| `string` | Yes      | Source concept name    |
| `source_value`  | `string` | Yes      | Source attribute value |
| `target_concept`| `string` | Yes      | Target concept name   |

**Response**: `{"ok": true, "analogy": "...", "results": [{"value": "...", "score": 0.92}]}`

---

### 10. Meta Memory & Self-Improvement

#### `GET /meta/proposals`

List self-improvement proposals generated by the MetaMemory module.

**Query Parameters**: `status` (optional: `"pending"`, `"accepted"`, `"rejected"`)

**Response**:

```json
{
  "ok": true,
  "count": 3,
  "proposals": [
    {
      "proposal_id": "prop_001",
      "type": "consolidation",
      "description": "Merge 5 similar memories about Python syntax",
      "status": "pending",
      "created_at": "2026-02-28T10:00:00Z"
    }
  ]
}
```

#### `POST /meta/proposals/{proposal_id}/status`

Update proposal status. Valid values: `accepted`, `rejected`, `implemented`.

**Request Body**: `{"status": "accepted"}`

---

### 11. Maintenance

#### `POST /maintenance/cleanup`

Remove low-quality memories below a threshold.

**Query Parameters**: `threshold` (float, default 0.1)

#### `POST /maintenance/consolidate`

Trigger a consolidation cycle across all tiers.

#### `POST /maintenance/sweep`

Run a full sweep of expired/stale memories.

All maintenance endpoints return `{"ok": true, "stats": {...}}`.

---

### 12. Subtle Thoughts

#### `GET /agents/{agent_id}/subtle-thoughts`

Retrieve associative "subtle thoughts" — background associations triggered by recent agent activity.

**Query Parameters**: `limit` (int, default 5)

**Response**: `{"ok": true, "associations": [...]}`

---

### 13. Recursive Synthesis (RLM)

#### `POST /rlm/query`

Execute a recursive language model query that decomposes complex questions into sub-queries, retrieves relevant memories for each, and synthesizes a unified answer.

**Request Body** (`RLMQueryRequest`):

| Field             | Type     | Required | Default | Constraints  | Description                |
|------------------|----------|----------|---------|--------------|----------------------------|
| `query`          | `string` | Yes      | —       | 1–4096 chars | The complex query          |
| `context_text`   | `string` | No       | —       | max 500k     | Additional context document |
| `project_id`     | `string` | No       | —       | max 128      | Project scope filter       |
| `max_depth`      | `int`    | No       | 2       | 0–5          | Max recursive depth        |
| `max_sub_queries`| `int`    | No       | 3       | 1–10         | Sub-queries per level      |
| `top_k`          | `int`    | No       | 5       | 1–50         | Results per sub-query      |

**Response** (`RLMQueryResponse`):

```json
{
  "ok": true,
  "query": "How does memory consolidation relate to learning?",
  "sub_queries": ["What is memory consolidation?", "What mechanisms underlie learning?"],
  "results": [...],
  "synthesis": "Memory consolidation and learning are deeply interconnected...",
  "max_depth_hit": 2,
  "elapsed_ms": 342.5,
  "ripple_snippets": ["Related: sleep plays a role in consolidation..."],
  "stats": {"total_queries": 7, "cache_hits": 2}
}
```

---

### 14. Trust & Provenance

#### `GET /memory/{memory_id}/lineage`

Get the provenance chain for a memory.

```json
{
  "ok": true,
  "memory_id": "mem_abc123",
  "provenance": {
    "origin": "user_input",
    "created_at": "2026-02-28T12:00:00Z",
    "transformations": ["encoded", "consolidated"],
    "parent_ids": []
  }
}
```

#### `GET /memory/{memory_id}/confidence`

Get the Bayesian confidence envelope for a memory.

```json
{
  "ok": true,
  "memory_id": "mem_abc123",
  "confidence": {
    "overall": 0.85,
    "source_reliability": 0.9,
    "temporal_decay": 0.88,
    "contradiction_free": true
  }
}
```

---

### 15. Proactive Recall

#### `GET /proactive`

Get proactively recommended memories based on context and patterns.

| Parameter  | Type     | Default | Description     |
|-----------|----------|---------|-----------------|
| `agent_id`| `string` | —       | Filter by agent |
| `limit`   | `int`    | 10      | Max results     |

**Response**:

```json
{
  "ok": true,
  "proactive_results": [
    {"id": "mem_abc123", "content": "...", "ltp_strength": 0.9, "confidence": 0.85, "tier": "hot"}
  ],
  "count": 3
}
```

---

### 16. Contradictions

#### `GET /contradictions`

List detected contradictions. Query: `?unresolved_only=true` (default).

```json
{
  "ok": true,
  "count": 2,
  "contradictions": [
    {
      "group_id": "cg_001",
      "memory_a_id": "mem_abc123",
      "memory_b_id": "mem_def456",
      "similarity_score": 0.72,
      "resolved": false
    }
  ]
}
```

#### `POST /contradictions/{group_id}/resolve`

**Request Body**: `{"note": "Memory A is more recent and accurate"}`

**Response**: `{"ok": true, "resolved_group_id": "cg_001"}`

---

### 17. Emotional Tags

#### `GET /memory/{memory_id}/emotional-tag`

Get the emotional profile of a memory.

```json
{
  "ok": true,
  "memory_id": "mem_abc123",
  "emotional_tag": {"valence": 0.6, "arousal": 0.3, "salience": 0.7}
}
```

#### `PATCH /memory/{memory_id}/emotional-tag`

Update emotional valence and arousal.

**Request Body**: `{"valence": 0.8, "arousal": 0.4}`

---

### 18. Knowledge Gaps

#### `GET /gaps`

List detected knowledge gaps.

```json
{
  "ok": true,
  "gaps": [
    {"gap_id": "gap_001", "query": "quantum entanglement", "type": "missing_info", "severity": 0.7}
  ],
  "count": 1
}
```

---

### 19. Association Network

Phase 6.0 graph-based memory associations.

#### `GET /associations/{node_id}`

Get memories associated with a given node.

| Parameter         | Type   | Default | Description             |
|------------------|--------|---------|--------------------------|
| `max_results`    | `int`  | 10      | Max associations         |
| `min_strength`   | `float`| 0.1     | Minimum edge strength    |
| `include_content`| `bool` | `true`  | Include memory content   |

```json
{
  "ok": true,
  "node_id": "mem_abc123",
  "associations": [
    {
      "id": "mem_def456",
      "content": "Related memory...",
      "strength": 0.82,
      "association_type": "semantic",
      "fire_count": 5,
      "metadata": {}
    }
  ]
}
```

#### `POST /associations/path`

Find paths between two memories in the association graph.

| Field         | Type     | Required | Default | Constraints | Description       |
|--------------|----------|----------|---------|-------------|-------------------|
| `from_id`    | `string` | Yes      | —       | —           | Source memory ID   |
| `to_id`      | `string` | Yes      | —       | —           | Target memory ID   |
| `max_hops`   | `int`    | No       | 3       | 1–10        | Max path length    |
| `min_strength`| `float` | No       | 0.1     | 0–1         | Min edge strength  |

#### `GET /associations/{node_id}/clusters`

Get clusters containing a specific node. Query: `?min_cluster_size=3`

#### `GET /associations/metrics`

Get graph-level topology metrics.

```json
{
  "ok": true,
  "metrics": {
    "node_count": 1234,
    "edge_count": 5678,
    "avg_degree": 9.2,
    "density": 0.015,
    "avg_clustering": 0.42,
    "connected_components": 3,
    "largest_component_size": 1200
  }
}
```

#### `POST /associations/reinforce`

Manually reinforce an association between two memories.

**Request Body**: `{"node_a": "mem_abc123", "node_b": "mem_def456", "association_type": "co_occurrence"}`

#### `GET /associations/visualize`

Interactive HTML visualization of the association graph.

| Parameter      | Type     | Default    | Description        |
|---------------|----------|------------|--------------------|
| `max_nodes`   | `int`    | 100        | Max nodes          |
| `min_strength`| `float`  | 0.1        | Min edge strength  |
| `layout`      | `string` | `"spring"` | Graph layout       |

Returns `text/html` content type.

---

## Prometheus Metrics

Available at `GET /metrics` (no authentication required).

| Metric                            | Type      | Description               |
|-----------------------------------|-----------|---------------------------|
| `haim_store_total`                | Counter   | Total store operations    |
| `haim_query_total`                | Counter   | Total query operations    |
| `haim_store_latency_seconds`      | Histogram | Store latency             |
| `haim_query_latency_seconds`      | Histogram | Query latency             |
| `haim_tier_count`                 | Gauge     | Memory count per tier     |
| `haim_dream_cycles_total`         | Counter   | Dream cycles completed    |
| `haim_consolidation_total`        | Counter   | Consolidation events      |

---

*This document covers all 41 API endpoints as of v2.0.0. For deployment, see [DEPLOYMENT.md](DEPLOYMENT.md). For configuration, see [CONFIGURATION.md](CONFIGURATION.md).*

