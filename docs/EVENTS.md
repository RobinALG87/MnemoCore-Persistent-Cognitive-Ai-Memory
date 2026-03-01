# MnemoCore Event System

> **Version**: 5.1.0 &nbsp;|&nbsp; **Source**: `src/mnemocore/events/`

MnemoCore includes a built-in publish-subscribe event system with webhook delivery. Events are emitted during memory operations, consolidation, dreams, and other cognitive processes.

---

## Table of Contents

- [Architecture](#architecture)
- [Event Types](#event-types)
- [EventBus API](#eventbus-api)
- [Webhook Manager](#webhook-manager)
- [Event Schemas](#event-schemas)
- [Integration Helpers](#integration-helpers)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)

---

## Architecture

```
┌──────────────┐     publish()     ┌──────────────┐     send_webhooks()     ┌──────────────┐
│ HAIM Engine  │ ─────────────────>│  Event Bus   │ ──────────────────────>│   Webhook    │
│ (Store/Query │                   │              │                        │   Manager    │
│  Dream/etc.) │                   │  - Queue     │                        │              │
└──────────────┘                   │  - History   │                        │  - HMAC sig  │
                                   │  - Routing   │                        │  - Retry     │
                                   └──────┬───────┘                        │  - Tracking  │
                                          │                                └──────┬───────┘
                                    subscribe()                                   │
                                          │                              HTTP POST (signed)
                                   ┌──────▼───────┐                        ┌──────▼───────┐
                                   │  Internal    │                        │  External    │
                                   │  Handlers    │                        │  Endpoints   │
                                   └──────────────┘                        └──────────────┘
```

---

## Event Types

Events follow a hierarchical dot-notation naming scheme:

| Pattern | Events | Description |
|---------|--------|-------------|
| `memory.*` | `memory.created`, `memory.accessed`, `memory.deleted`, `memory.consolidated` | Memory lifecycle events |
| `consolidation.*` | `consolidation.started`, `consolidation.completed` | Tier consolidation events |
| `contradiction.*` | `contradiction.detected`, `contradiction.resolved` | Contradiction detection/resolution |
| `dream.*` | `dream.started`, `dream.completed`, `dream.failed` | Dream pipeline events |
| `synapse.*` | `synapse.formed`, `synapse.fired` | Synaptic connection events |
| `gap.*` | `gap.detected`, `gap.filled` | Knowledge gap events |

### Event Object

```python
@dataclass
class Event:
    id: str              # "evt_{uuid}"
    type: str            # e.g., "memory.created"
    data: Dict[str, Any] # Event-specific payload
    metadata: Dict[str, Any]  # Optional context
    published_at: datetime
```

---

## EventBus API

The `EventBus` is a singleton that manages subscriptions, event routing, and delivery.

### Getting the EventBus

```python
from mnemocore.events.event_bus import get_event_bus, reset_event_bus

# Get or create the singleton
bus = get_event_bus()

# Reset (mainly for testing)
reset_event_bus()
```

### Subscribing to Events

```python
async def on_memory_created(event):
    print(f"Memory created: {event.data['memory_id']}")

# Subscribe to a specific event type
sub_id = bus.subscribe("memory.created", on_memory_created)

# Subscribe with a wildcard pattern
sub_id = bus.subscribe("memory.*", handle_all_memory_events)

# Subscribe with a filter
sub_id = bus.subscribe(
    "memory.created",
    handler=on_memory_created,
    event_filter=lambda e: e.data.get("tier") == "hot"
)
```

### Unsubscribing

```python
# Unsubscribe by ID
bus.unsubscribe(sub_id)

# Unsubscribe all handlers for a pattern
count = bus.unsubscribe_pattern("memory.*")

# Enable/disable without removing
bus.disable_subscription(sub_id)
bus.enable_subscription(sub_id)
```

### Publishing Events

```python
# Publish a single event
event = await bus.publish(
    event_type="memory.created",
    data={"memory_id": "mem_123", "content": "...", "tier": "hot"},
    metadata={"agent_id": "agent-01"}
)

# Publish a batch of events
events = await bus.publish_batch([
    {"type": "memory.created", "data": {...}},
    {"type": "synapse.formed", "data": {...}},
])
```

### Lifecycle

```python
# Start the event bus (begins processing queue)
await bus.start()

# Stop gracefully (drains queue)
await bus.stop()

# Full shutdown
await bus.shutdown()
```

### History and Metrics

```python
# Get recent event history
history = await bus.get_history(event_type="memory.created", limit=50)

# Get delivery metrics
metrics = bus.get_metrics()
# Returns: {
#   "total_published": 1234,
#   "total_delivered": 1200,
#   "total_errors": 34,
#   "subscriptions_active": 5,
#   "queue_size": 0
# }

# Detailed stats
stats = await bus.get_stats()
```

---

## Webhook Manager

The `WebhookManager` handles external HTTP webhook delivery with HMAC signatures and retry logic.

### Webhook Registration

```python
from mnemocore.events.webhook_manager import get_webhook_manager

wm = get_webhook_manager()

# Register a webhook
webhook = await wm.register_webhook(
    url="https://example.com/webhook",
    events=["memory.created", "dream.completed"],
    secret="your-hmac-secret",
    headers={"X-Custom": "value"},
    enabled=True,
    retry_config=RetryConfig(
        max_attempts=5,
        base_delay_seconds=1.0,
        max_delay_seconds=300.0,
        timeout_seconds=30.0
    )
)

# Update a webhook
await wm.update_webhook(webhook.id, events=["memory.*"])

# Delete a webhook
await wm.delete_webhook(webhook.id)

# List webhooks
webhooks = wm.list_webhooks(enabled_only=True)
```

### Signature Verification

Webhooks are signed using HMAC-SHA256. The signature is sent in the `X-MnemoCore-Signature` header:

```
X-MnemoCore-Signature: t=1709150400,v1=5257a869e7ecebeda32affa62cdca3fa51cad7e77a0e56ff536d0ce8e108d8bd
```

Verification on the receiving side:

```python
from mnemocore.events.webhook_manager import WebhookSignature

is_valid = WebhookSignature.verify(
    payload=request.body,
    secret="your-hmac-secret",
    signature_header=request.headers["X-MnemoCore-Signature"]
)
```

### Retry Logic

Failed deliveries are retried with exponential backoff:

| Attempt | Delay |
|---------|-------|
| 1st retry | 1s |
| 2nd retry | 2s |
| 3rd retry | 4s |
| 4th retry | 8s |
| 5th retry (max) | 16s (capped at `max_delay_seconds`) |

### Delivery Tracking

```python
# Get delivery history for a webhook
history = await wm.get_delivery_history(
    webhook_id=webhook.id,
    limit=100,
    status="failed"  # Optional: filter by status
)

# Get webhook status summary
status = await wm.get_webhook_status(webhook.id)
# Returns delivery counts, success rate, last delivery time

# Get aggregate metrics
metrics = wm.get_metrics()
```

---

## Event Schemas

Every event type has a JSON Schema for validation. The schema registry ensures event payloads conform to expected structures.

### Available Schemas

| Event Type | Required Fields | Optional Fields |
|------------|----------------|-----------------|
| `memory.created` | `memory_id`, `content`, `tier` | `ltp_strength`, `eig`, `tags`, `category`, `agent_id`, `episode_id` |
| `memory.accessed` | `memory_id`, `query` | `similarity_score`, `rank`, `tier` |
| `memory.deleted` | `memory_id` | `tier`, `reason` |
| `memory.consolidated` | `memory_id`, `from_tier`, `to_tier` | `consolidation_type`, `ltp_strength` |
| `consolidation.completed` | `cycle_id`, `duration_seconds` | `memories_processed`, `hot_to_warm`, `warm_to_cold` |
| `contradiction.detected` | `group_id`, `memory_a_id`, `memory_b_id` | `similarity_score`, `llm_confirmed` |
| `contradiction.resolved` | `group_id` | `resolution_type`, `resolution_note` |
| `dream.started` | `session_id`, `trigger` | `schedule_name`, `max_memories`, `stages_enabled` |
| `dream.completed` | `session_id`, `duration_seconds` | `memories_processed`, `clusters_found`, `patterns_extracted` |
| `dream.failed` | `session_id`, `error` | `stage`, `duration_seconds` |
| `synapse.formed` | `synapse_id`, `neuron_a_id`, `neuron_b_id` | `initial_strength`, `formation_reason` |
| `synapse.fired` | `synapse_id`, `source_id`, `target_id` | `strength`, `was_successful` |
| `gap.detected` | `gap_id`, `query` | `gap_type`, `related_memory_ids`, `severity` |
| `gap.filled` | `gap_id`, `memory_id` | `fill_method`, `confidence` |

### Validation

```python
from mnemocore.events.schemas import validate_event, list_event_types

# Validate event data
is_valid, errors = validate_event("memory.created", {
    "memory_id": "mem_123",
    "content": "Test memory",
    "tier": "hot"
})

# List all registered event types
types = list_event_types()
# ["memory.created", "memory.accessed", "memory.deleted", ...]

# Export as OpenAPI spec
spec = export_openapi_spec()
```

---

## Integration Helpers

The `integration` module provides convenient functions for emitting events from within MnemoCore components. These handle schema construction and error handling.

```python
from mnemocore.events.integration import (
    emit_memory_created,
    emit_memory_accessed,
    emit_memory_deleted,
    emit_memory_consolidated,
    emit_dream_started,
    emit_dream_completed,
    emit_dream_failed,
    emit_contradiction_detected,
    emit_contradiction_resolved,
    emit_synapse_formed,
    emit_synapse_fired,
    emit_gap_detected,
    emit_gap_filled,
    emit_consolidation_completed,
)

# Example: emit a memory.created event
await emit_memory_created(
    event_bus=bus,
    memory_id="mem_123",
    content="New knowledge",
    tier="hot",
    ltp_strength=0.5,
    tags=["test"],
    agent_id="agent-01"
)

# Example: emit a dream.completed event
await emit_dream_completed(
    event_bus=bus,
    session_id="dream_001",
    duration_seconds=12.5,
    memories_processed=47,
    clusters_found=5,
    patterns_extracted=3,
    dream_report_path="./data/dream_reports/report_001.json"
)
```

---

## Configuration

Events and webhooks are configured in `config.yaml`:

```yaml
events:
  enabled: true
  max_queue_size: 10000
  delivery_timeout: 30.0

webhooks:
  enabled: true
  persistence_path: "./data/webhooks.json"
```

See [CONFIGURATION.md](CONFIGURATION.md) for the complete reference.

---

## Usage Examples

### Monitor All Memory Operations

```python
from mnemocore.events.event_bus import get_event_bus

bus = get_event_bus()

async def log_memory_event(event):
    print(f"[{event.type}] {event.data.get('memory_id')} at {event.published_at}")

bus.subscribe("memory.*", log_memory_event)
await bus.start()
```

### Dream Completion Notification

```python
async def on_dream_completed(event):
    report = {
        "session": event.data["session_id"],
        "duration": event.data["duration_seconds"],
        "processed": event.data.get("memories_processed", 0),
    }
    await send_notification(f"Dream completed: {report}")

bus.subscribe("dream.completed", on_dream_completed)
```

### External Webhook Integration

```python
from mnemocore.events.webhook_manager import get_webhook_manager

wm = get_webhook_manager()

# Register Slack-style webhook
await wm.register_webhook(
    url="https://hooks.slack.com/services/T.../B.../xxx",
    events=["dream.completed", "contradiction.detected"],
    secret="webhook-secret-123"
)
```

---

*See [API.md](API.md) for the REST API. See [ARCHITECTURE.md](ARCHITECTURE.md) for how events fit into the system.*
