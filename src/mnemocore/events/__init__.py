"""
MnemoCore Events & Webhook System
==================================

This module provides:
- EventBus: Internal publish-subscribe for memory events
- WebhookManager: External webhook delivery with retry logic
- Event schemas: JSON Schema validation for events
- Signature verification: HMAC-based webhook authentication

Event Types:
    - memory.created: New memory stored
    - memory.accessed: Memory retrieved via query
    - memory.deleted: Memory removed from system
    - memory.consolidated: Memory moved between tiers
    - consolidation.completed: Consolidation cycle finished
    - contradiction.detected: Contradiction between memories found
    - contradiction.resolved: Contradiction marked as resolved
    - dream.started: Dream session initiated
    - dream.completed: Dream session finished
    - dream.failed: Dream session encountered error
    - synapse.formed: New synaptic connection created
    - synapse.fired: Existing synapse activated
    - gap.detected: Knowledge gap identified
    - gap.filled: Knowledge gap filled with new content

Usage:
    ```python
    from mnemocore.events import get_event_bus, get_webhook_manager

    # Subscribe to events
    event_bus = get_event_bus()
    event_bus.subscribe("memory.created", my_handler)

    # Publish events
    await event_bus.publish("memory.created", {"memory_id": "..."})

    # Webhooks are automatically triggered for subscribed events
    webhook_manager = get_webhook_manager()
    await webhook_manager.register_webhook(
        url="https://example.com/webhook",
        events=["memory.created", "consolidation.completed"],
        secret="my_secret"
    )
    ```
"""

from .event_bus import EventBus, get_event_bus, Event, EventHandler, EventFilter
from .webhook_manager import (
    WebhookManager,
    get_webhook_manager,
    WebhookConfig,
    WebhookDelivery,
    WebhookSignature,
    RetryConfig as WebhookRetryConfig,
)
from .schemas import (
    EventSchema,
    MemoryCreatedSchema,
    MemoryAccessedSchema,
    MemoryDeletedSchema,
    MemoryConsolidatedSchema,
    ConsolidationCompletedSchema,
    ContradictionDetectedSchema,
    ContradictionResolvedSchema,
    DreamStartedSchema,
    DreamCompletedSchema,
    DreamFailedSchema,
    SynapseFormedSchema,
    SynapseFiredSchema,
    GapDetectedSchema,
    GapFilledSchema,
    validate_event,
    get_schema_for_event_type,
)
from . import integration
from .integration import (
    emit_event,
    emit_memory_created,
    emit_memory_accessed,
    emit_memory_deleted,
    emit_memory_consolidated,
    emit_consolidation_completed,
    emit_contradiction_detected,
    emit_contradiction_resolved,
    emit_dream_started,
    emit_dream_completed,
    emit_dream_failed,
    emit_synapse_formed,
    emit_synapse_fired,
    emit_gap_detected,
    emit_gap_filled,
)

__all__ = [
    # EventBus
    "EventBus",
    "get_event_bus",
    "Event",
    "EventHandler",
    "EventFilter",
    # WebhookManager
    "WebhookManager",
    "get_webhook_manager",
    "WebhookConfig",
    "WebhookDelivery",
    "WebhookSignature",
    # Schemas
    "EventSchema",
    "MemoryCreatedSchema",
    "MemoryAccessedSchema",
    "MemoryDeletedSchema",
    "MemoryConsolidatedSchema",
    "ConsolidationCompletedSchema",
    "ContradictionDetectedSchema",
    "ContradictionResolvedSchema",
    "DreamStartedSchema",
    "DreamCompletedSchema",
    "DreamFailedSchema",
    "SynapseFormedSchema",
    "SynapseFiredSchema",
    "GapDetectedSchema",
    "GapFilledSchema",
    "validate_event",
    "get_schema_for_event_type",
    # Integration helpers
    "emit_event",
    "emit_memory_created",
    "emit_memory_accessed",
    "emit_memory_deleted",
    "emit_memory_consolidated",
    "emit_consolidation_completed",
    "emit_contradiction_detected",
    "emit_contradiction_resolved",
    "emit_dream_started",
    "emit_dream_completed",
    "emit_dream_failed",
    "emit_synapse_formed",
    "emit_synapse_fired",
    "emit_gap_detected",
    "emit_gap_filled",
]

# Version info
__version__ = "1.0.0"
