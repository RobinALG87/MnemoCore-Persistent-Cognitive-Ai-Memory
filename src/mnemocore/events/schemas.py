"""
Event Schemas - JSON Schema Validation for MnemoCore Events
============================================================

Defines JSON Schema for all MnemoCore event types.

Provides validation, documentation, and type safety for events
flowing through the EventBus and webhook deliveries.

Available Schemas:
    - MemoryCreatedSchema: New memory stored
    - MemoryAccessedSchema: Memory retrieved via query
    - MemoryDeletedSchema: Memory removed from system
    - MemoryConsolidatedSchema: Memory moved between tiers
    - ConsolidationCompletedSchema: Consolidation cycle finished
    - ContradictionDetectedSchema: Contradiction found
    - ContradictionResolvedSchema: Contradiction resolved
    - DreamStartedSchema: Dream session started
    - DreamCompletedSchema: Dream session finished
    - DreamFailedSchema: Dream session failed
    - SynapseFormedSchema: New synapse created
    - SynapseFiredSchema: Synapse activated
    - GapDetectedSchema: Knowledge gap found
    - GapFilledSchema: Gap filled with new content

Example:
    ```python
    from mnemocore.events import validate_event

    # Validate event data
    is_valid, errors = validate_event("memory.created", {
        "memory_id": "abc123",
        "content": "Hello world",
        "tier": "hot"
    })

    if not is_valid:
        print(f"Validation errors: {errors}")
    ```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


# =============================================================================
# Base Event Schema
# =============================================================================

@dataclass
class EventSchema:
    """
    Base class for event schemas.

    Each event type has a corresponding schema class that:
    - Defines the JSON schema for validation
    - Provides documentation
    - Supports event data construction
    """

    event_type: str
    description: str
    schema: Dict[str, Any] = field(default_factory=dict)

    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate event data against the schema.

        Args:
            data: Event data to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        required = self.schema.get("required", [])
        for field_name in required:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")

        # Check field types
        properties = self.schema.get("properties", {})
        for field_name, value in data.items():
            if field_name not in properties:
                errors.append(f"Unknown field: {field_name}")
                continue

            field_schema = properties[field_name]
            field_type = field_schema.get("type")

            if not self._check_type(value, field_type, field_schema):
                expected = field_type
                actual = type(value).__name__
                errors.append(
                    f"Field '{field_name}' has wrong type: "
                    f"expected {expected}, got {actual}"
                )

        # Check enum values
        for field_name, value in data.items():
            if field_name in properties:
                enum_values = properties[field_name].get("enum")
                if enum_values and value not in enum_values:
                    errors.append(
                        f"Field '{field_name}' has invalid value: "
                        f"{value} not in {enum_values}"
                    )

        return len(errors) == 0, errors

    def _check_type(
        self,
        value: Any,
        expected_type: str,
        field_schema: Dict[str, Any],
    ) -> bool:
        """Check if a value matches the expected type."""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        elif expected_type == "null":
            return value is None
        return True


# =============================================================================
# Memory Event Schemas
# =============================================================================

class MemoryCreatedSchema(EventSchema):
    """Schema for memory.created events."""

    def __init__(self):
        super().__init__(
            event_type="memory.created",
            description="Emitted when a new memory is stored in the system",
            schema={
                "type": "object",
                "title": "MemoryCreatedEvent",
                "description": "Event emitted when a new memory is stored",
                "required": ["memory_id", "content", "tier"],
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Unique identifier for the memory",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content of the memory",
                    },
                    "content_preview": {
                        "type": "string",
                        "description": "Truncated content preview (max 200 chars)",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["hot", "warm", "cold"],
                        "description": "Storage tier where memory was placed",
                    },
                    "ltp_strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Long-term potentization strength",
                    },
                    "eig": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Expected information gain (novelty)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags associated with the memory",
                    },
                    "category": {
                        "type": "string",
                        "description": "Memory category",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "ID of the agent that created the memory",
                    },
                    "episode_id": {
                        "type": "string",
                        "description": "ID of the episode this memory belongs to",
                    },
                    "previous_id": {
                        "type": "string",
                        "description": "ID of previous memory in episodic chain",
                    },
                },
            },
        )


class MemoryAccessedSchema(EventSchema):
    """Schema for memory.accessed events."""

    def __init__(self):
        super().__init__(
            event_type="memory.accessed",
            description="Emitted when a memory is retrieved via query",
            schema={
                "type": "object",
                "title": "MemoryAccessedEvent",
                "description": "Event emitted when a memory is accessed",
                "required": ["memory_id", "query"],
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Unique identifier for the memory",
                    },
                    "query": {
                        "type": "string",
                        "description": "Query text that retrieved the memory",
                    },
                    "similarity_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Similarity score for retrieval",
                    },
                    "rank": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Rank in query results",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["hot", "warm", "cold"],
                        "description": "Storage tier where memory was found",
                    },
                },
            },
        )


class MemoryDeletedSchema(EventSchema):
    """Schema for memory.deleted events."""

    def __init__(self):
        super().__init__(
            event_type="memory.deleted",
            description="Emitted when a memory is removed from the system",
            schema={
                "type": "object",
                "title": "MemoryDeletedEvent",
                "description": "Event emitted when a memory is deleted",
                "required": ["memory_id"],
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Unique identifier for the deleted memory",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["hot", "warm", "cold"],
                        "description": "Storage tier the memory was in before deletion",
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["manual", "eviction", "decay", "contradiction"],
                        "description": "Reason for deletion",
                    },
                },
            },
        )


class MemoryConsolidatedSchema(EventSchema):
    """Schema for memory.consolidated events."""

    def __init__(self):
        super().__init__(
            event_type="memory.consolidated",
            description="Emitted when a memory moves between storage tiers",
            schema={
                "type": "object",
                "title": "MemoryConsolidatedEvent",
                "description": "Event emitted when a memory is consolidated",
                "required": ["memory_id", "from_tier", "to_tier"],
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Unique identifier for the memory",
                    },
                    "from_tier": {
                        "type": "string",
                        "enum": ["hot", "warm", "cold"],
                        "description": "Source storage tier",
                    },
                    "to_tier": {
                        "type": "string",
                        "enum": ["hot", "warm", "cold"],
                        "description": "Destination storage tier",
                    },
                    "consolidation_type": {
                        "type": "string",
                        "enum": ["promotion", "demotion", "archival"],
                        "description": "Type of consolidation",
                    },
                    "ltp_strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "LTP strength at time of consolidation",
                    },
                },
            },
        )


# =============================================================================
# Consolidation Event Schemas
# =============================================================================

class ConsolidationCompletedSchema(EventSchema):
    """Schema for consolidation.completed events."""

    def __init__(self):
        super().__init__(
            event_type="consolidation.completed",
            description="Emitted when a consolidation cycle finishes",
            schema={
                "type": "object",
                "title": "ConsolidationCompletedEvent",
                "description": "Event emitted when consolidation cycle completes",
                "required": ["cycle_id", "duration_seconds"],
                "properties": {
                    "cycle_id": {
                        "type": "string",
                        "description": "Unique identifier for the consolidation cycle",
                    },
                    "duration_seconds": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Duration of the consolidation cycle",
                    },
                    "memories_processed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of memories processed",
                    },
                    "memories_consolidated": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of memories moved between tiers",
                    },
                    "hot_to_warm": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of memories moved from hot to warm",
                    },
                    "warm_to_cold": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of memories moved from warm to cold",
                    },
                },
            },
        )


# =============================================================================
# Contradiction Event Schemas
# =============================================================================

class ContradictionDetectedSchema(EventSchema):
    """Schema for contradiction.detected events."""

    def __init__(self):
        super().__init__(
            event_type="contradiction.detected",
            description="Emitted when a contradiction between memories is found",
            schema={
                "type": "object",
                "title": "ContradictionDetectedEvent",
                "description": "Event emitted when contradiction is detected",
                "required": ["group_id", "memory_a_id", "memory_b_id"],
                "properties": {
                    "group_id": {
                        "type": "string",
                        "description": "Unique identifier for the contradiction group",
                    },
                    "memory_a_id": {
                        "type": "string",
                        "description": "ID of the first memory in the contradiction",
                    },
                    "memory_b_id": {
                        "type": "string",
                        "description": "ID of the second memory in the contradiction",
                    },
                    "similarity_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Similarity score between the memories",
                    },
                    "llm_confirmed": {
                        "type": "boolean",
                        "description": "Whether the contradiction was confirmed by LLM",
                    },
                    "llm_confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "LLM confidence in the contradiction",
                    },
                },
            },
        )


class ContradictionResolvedSchema(EventSchema):
    """Schema for contradiction.resolved events."""

    def __init__(self):
        super().__init__(
            event_type="contradiction.resolved",
            description="Emitted when a contradiction is marked as resolved",
            schema={
                "type": "object",
                "title": "ContradictionResolvedEvent",
                "description": "Event emitted when contradiction is resolved",
                "required": ["group_id"],
                "properties": {
                    "group_id": {
                        "type": "string",
                        "description": "Unique identifier for the contradiction group",
                    },
                    "resolution_type": {
                        "type": "string",
                        "enum": ["manual", "auto", "merge", "delete_a", "delete_b"],
                        "description": "How the contradiction was resolved",
                    },
                    "resolution_note": {
                        "type": "string",
                        "description": "Notes about the resolution",
                    },
                    "resolved_by": {
                        "type": "string",
                        "description": "Who/what resolved the contradiction",
                    },
                },
            },
        )


# =============================================================================
# Dream Event Schemas
# =============================================================================

class DreamStartedSchema(EventSchema):
    """Schema for dream.started events."""

    def __init__(self):
        super().__init__(
            event_type="dream.started",
            description="Emitted when a dream session is initiated",
            schema={
                "type": "object",
                "title": "DreamStartedEvent",
                "description": "Event emitted when dream session starts",
                "required": ["session_id", "trigger"],
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique identifier for the dream session",
                    },
                    "trigger": {
                        "type": "string",
                        "enum": ["idle", "scheduled", "manual"],
                        "description": "What triggered the dream session",
                    },
                    "schedule_name": {
                        "type": "string",
                        "description": "Name of the schedule (if scheduled)",
                    },
                    "max_memories": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Maximum memories to process",
                    },
                    "stages_enabled": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "episodic_clustering",
                                "pattern_extraction",
                                "recursive_synthesis",
                                "contradiction_resolution",
                                "semantic_promotion",
                                "dream_report",
                            ],
                        },
                        "description": "Pipeline stages enabled for this session",
                    },
                },
            },
        )


class DreamCompletedSchema(EventSchema):
    """Schema for dream.completed events."""

    def __init__(self):
        super().__init__(
            event_type="dream.completed",
            description="Emitted when a dream session finishes successfully",
            schema={
                "type": "object",
                "title": "DreamCompletedEvent",
                "description": "Event emitted when dream session completes",
                "required": ["session_id", "duration_seconds"],
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique identifier for the dream session",
                    },
                    "duration_seconds": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Duration of the dream session",
                    },
                    "memories_processed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of memories processed",
                    },
                    "clusters_found": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of episodic clusters found",
                    },
                    "patterns_extracted": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of patterns extracted",
                    },
                    "contradictions_resolved": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of contradictions resolved",
                    },
                    "memories_promoted": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of memories promoted to warm tier",
                    },
                    "dream_report_path": {
                        "type": "string",
                        "description": "Path to the generated dream report",
                    },
                },
            },
        )


class DreamFailedSchema(EventSchema):
    """Schema for dream.failed events."""

    def __init__(self):
        super().__init__(
            event_type="dream.failed",
            description="Emitted when a dream session encounters an error",
            schema={
                "type": "object",
                "title": "DreamFailedEvent",
                "description": "Event emitted when dream session fails",
                "required": ["session_id", "error"],
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique identifier for the dream session",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message",
                    },
                    "stage": {
                        "type": "string",
                        "description": "Pipeline stage where error occurred",
                    },
                    "duration_seconds": {
                        "type": "number",
                        "minimum": 0.0,
                        "description": "Duration before failure",
                    },
                    "memories_processed": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of memories processed before failure",
                    },
                },
            },
        )


# =============================================================================
# Synapse Event Schemas
# =============================================================================

class SynapseFormedSchema(EventSchema):
    """Schema for synapse.formed events."""

    def __init__(self):
        super().__init__(
            event_type="synapse.formed",
            description="Emitted when a new synaptic connection is created",
            schema={
                "type": "object",
                "title": "SynapseFormedEvent",
                "description": "Event emitted when synapse is formed",
                "required": ["synapse_id", "neuron_a_id", "neuron_b_id"],
                "properties": {
                    "synapse_id": {
                        "type": "string",
                        "description": "Unique identifier for the synapse",
                    },
                    "neuron_a_id": {
                        "type": "string",
                        "description": "ID of the first memory (neuron)",
                    },
                    "neuron_b_id": {
                        "type": "string",
                        "description": "ID of the second memory (neuron)",
                    },
                    "initial_strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Initial synaptic strength",
                    },
                    "formation_reason": {
                        "type": "string",
                        "enum": ["auto_bind", "associative", "manual", "dream"],
                        "description": "Why the synapse was formed",
                    },
                    "similarity": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Similarity between the connected memories",
                    },
                },
            },
        )


class SynapseFiredSchema(EventSchema):
    """Schema for synapse.fired events."""

    def __init__(self):
        super().__init__(
            event_type="synapse.fired",
            description="Emitted when an existing synapse is activated",
            schema={
                "type": "object",
                "title": "SynapseFiredEvent",
                "description": "Event emitted when synapse fires",
                "required": ["synapse_id", "source_id", "target_id"],
                "properties": {
                    "synapse_id": {
                        "type": "string",
                        "description": "Unique identifier for the synapse",
                    },
                    "source_id": {
                        "type": "string",
                        "description": "ID of the source memory",
                    },
                    "target_id": {
                        "type": "string",
                        "description": "ID of the target memory",
                    },
                    "strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Synaptic strength after firing",
                    },
                    "was_successful": {
                        "type": "boolean",
                        "description": "Whether the activation was successful",
                    },
                    "context": {
                        "type": "string",
                        "description": "Context in which the synapse fired",
                    },
                },
            },
        )


# =============================================================================
# Gap Event Schemas
# =============================================================================

class GapDetectedSchema(EventSchema):
    """Schema for gap.detected events."""

    def __init__(self):
        super().__init__(
            event_type="gap.detected",
            description="Emitted when a knowledge gap is identified",
            schema={
                "type": "object",
                "title": "GapDetectedEvent",
                "description": "Event emitted when knowledge gap is detected",
                "required": ["gap_id", "query"],
                "properties": {
                    "gap_id": {
                        "type": "string",
                        "description": "Unique identifier for the gap",
                    },
                    "query": {
                        "type": "string",
                        "description": "Query that revealed the gap",
                    },
                    "gap_type": {
                        "type": "string",
                        "enum": ["missing_info", "low_confidence", "contradiction"],
                        "description": "Type of knowledge gap",
                    },
                    "related_memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of related memories",
                    },
                    "severity": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Severity of the gap",
                    },
                },
            },
        )


class GapFilledSchema(EventSchema):
    """Schema for gap.filled events."""

    def __init__(self):
        super().__init__(
            event_type="gap.filled",
            description="Emitted when a knowledge gap is filled",
            schema={
                "type": "object",
                "title": "GapFilledEvent",
                "description": "Event emitted when knowledge gap is filled",
                "required": ["gap_id", "memory_id"],
                "properties": {
                    "gap_id": {
                        "type": "string",
                        "description": "Unique identifier for the gap",
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory that filled the gap",
                    },
                    "fill_method": {
                        "type": "string",
                        "enum": ["llm_generation", "user_input", "external_source"],
                        "description": "How the gap was filled",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in the filled information",
                    },
                },
            },
        )


# =============================================================================
# Schema Registry
# =============================================================================

_SCHEMAS: Dict[str, EventSchema] = {}


def _register_schema(schema: EventSchema) -> None:
    """Register an event schema."""
    _SCHEMAS[schema.event_type] = schema


# Register all schemas
def _initialize_schemas() -> None:
    """Initialize all event schemas."""
    schemas = [
        MemoryCreatedSchema(),
        MemoryAccessedSchema(),
        MemoryDeletedSchema(),
        MemoryConsolidatedSchema(),
        ConsolidationCompletedSchema(),
        ContradictionDetectedSchema(),
        ContradictionResolvedSchema(),
        DreamStartedSchema(),
        DreamCompletedSchema(),
        DreamFailedSchema(),
        SynapseFormedSchema(),
        SynapseFiredSchema(),
        GapDetectedSchema(),
        GapFilledSchema(),
    ]

    for schema in schemas:
        _register_schema(schema)


# Auto-initialize on module load
_initialize_schemas()


# =============================================================================
# Validation Functions
# =============================================================================

def validate_event(event_type: str, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate event data against its schema.

    Args:
        event_type: Type of event (e.g., "memory.created")
        data: Event data to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    schema = _SCHEMAS.get(event_type)

    if schema is None:
        # Unknown event type - warn but don't fail
        logger.warning(f"[EventSchema] Unknown event type: {event_type}")
        return True, []

    return schema.validate(data)


def get_schema_for_event_type(event_type: str) -> Optional[EventSchema]:
    """Get the schema for a specific event type."""
    return _SCHEMAS.get(event_type)


def list_event_types() -> List[str]:
    """List all registered event types."""
    return list(_SCHEMAS.keys())


def get_json_schema(event_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get JSON Schema format for an event type or all events.

    Args:
        event_type: Specific event type, or None for all

    Returns:
        JSON Schema dictionary
    """
    if event_type:
        schema = _SCHEMAS.get(event_type)
        if schema:
            return schema.schema
        return {}

    # Return all schemas as a mapping
    return {
        event_type: schema.schema
        for event_type, schema in _SCHEMAS.items()
    }


def export_openapi_spec() -> Dict[str, Any]:
    """
    Export event schemas as OpenAPI-compatible specification.

    Useful for API documentation and webhook recipient integration.
    """
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "MnemoCore Events API",
            "version": "1.0.0",
            "description": "Webhook event schemas for MnemoCore",
        },
        "components": {
            "schemas": {}
        },
    }

    for event_type, schema in _SCHEMAS.items():
        component_name = event_type.replace(".", "_")
        spec["components"]["schemas"][component_name] = schema.schema

    return spec
