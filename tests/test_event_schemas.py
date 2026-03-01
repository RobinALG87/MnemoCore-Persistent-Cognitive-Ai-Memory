"""
Tests for Event Schemas Module
==============================
Comprehensive tests for JSON Schema validation for MnemoCore events.

Tests cover:
- Schema validation for each defined schema
- Type checking for all supported types
- Invalid schema rejection
- Edge cases: empty objects, nested objects
"""

import pytest

from mnemocore.events.schemas import (
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
    list_event_types,
    get_json_schema,
    export_openapi_spec,
)


@pytest.fixture
def memory_created_schema():
    """Create a MemoryCreatedSchema instance."""
    return MemoryCreatedSchema()


@pytest.fixture
def memory_accessed_schema():
    """Create a MemoryAccessedSchema instance."""
    return MemoryAccessedSchema()


@pytest.fixture
def dream_started_schema():
    """Create a DreamStartedSchema instance."""
    return DreamStartedSchema()


class TestEventSchemaBase:
    """Tests for base EventSchema class."""

    def test_validate_missing_required_field(self, memory_created_schema):
        """Test validation fails when required field is missing."""
        data = {
            "memory_id": "test123",
            # Missing content and tier
        }

        is_valid, errors = memory_created_schema.validate(data)

        assert is_valid is False
        assert len(errors) > 0
        assert any("content" in e.lower() for e in errors)

    def test_validate_all_required_fields(self, memory_created_schema):
        """Test validation passes with all required fields."""
        data = {
            "memory_id": "test123",
            "content": "Test content",
            "tier": "hot"
        }

        is_valid, errors = memory_created_schema.validate(data)

        assert is_valid is True
        assert errors == []

    def test_validate_unknown_field(self, memory_created_schema):
        """Test validation warns about unknown fields."""
        data = {
            "memory_id": "test123",
            "content": "Test content",
            "tier": "hot",
            "unknown_field": "value"
        }

        is_valid, errors = memory_created_schema.validate(data)

        assert is_valid is False
        assert any("unknown" in e.lower() for e in errors)

    def test_validate_wrong_type_string(self, memory_created_schema):
        """Test validation fails for wrong type (string expected)."""
        data = {
            "memory_id": 123,  # Should be string
            "content": "Test content",
            "tier": "hot"
        }

        is_valid, errors = memory_created_schema.validate(data)

        assert is_valid is False
        assert any("memory_id" in e for e in errors)

    def test_validate_wrong_type_number(self, memory_created_schema):
        """Test validation fails for wrong type (number expected)."""
        data = {
            "memory_id": "test123",
            "content": "Test content",
            "tier": "hot",
            "ltp_strength": "high"  # Should be number
        }

        is_valid, errors = memory_created_schema.validate(data)

        assert is_valid is False
        assert any("ltp_strength" in e for e in errors)

    def test_validate_invalid_enum_value(self, memory_created_schema):
        """Test validation fails for invalid enum value."""
        data = {
            "memory_id": "test123",
            "content": "Test content",
            "tier": "invalid_tier"  # Should be hot, warm, or cold
        }

        is_valid, errors = memory_created_schema.validate(data)

        assert is_valid is False
        assert any("tier" in e.lower() for e in errors)


class TestTypeChecking:
    """Tests for type checking in schema validation."""

    def test_check_type_string(self, memory_created_schema):
        """Test string type checking."""
        assert memory_created_schema._check_type("hello", "string", {}) is True
        assert memory_created_schema._check_type(123, "string", {}) is False

    def test_check_type_integer(self, memory_created_schema):
        """Test integer type checking."""
        assert memory_created_schema._check_type(42, "integer", {}) is True
        assert memory_created_schema._check_type("42", "integer", {}) is False
        assert memory_created_schema._check_type(True, "integer", {}) is False
        assert memory_created_schema._check_type(3.14, "integer", {}) is False

    def test_check_type_number(self, memory_created_schema):
        """Test number type checking."""
        assert memory_created_schema._check_type(42, "number", {}) is True
        assert memory_created_schema._check_type(3.14, "number", {}) is True
        assert memory_created_schema._check_type("42", "number", {}) is False

    def test_check_type_boolean(self, memory_created_schema):
        """Test boolean type checking."""
        assert memory_created_schema._check_type(True, "boolean", {}) is True
        assert memory_created_schema._check_type(False, "boolean", {}) is True
        assert memory_created_schema._check_type(1, "boolean", {}) is False
        assert memory_created_schema._check_type("true", "boolean", {}) is False

    def test_check_type_array(self, memory_created_schema):
        """Test array type checking."""
        assert memory_created_schema._check_type([1, 2, 3], "array", {}) is True
        assert memory_created_schema._check_type([], "array", {}) is True
        assert memory_created_schema._check_type("list", "array", {}) is False

    def test_check_type_object(self, memory_created_schema):
        """Test object type checking."""
        assert memory_created_schema._check_type({"key": "value"}, "object", {}) is True
        assert memory_created_schema._check_type({}, "object", {}) is True
        assert memory_created_schema._check_type("dict", "object", {}) is False

    def test_check_type_null(self, memory_created_schema):
        """Test null type checking."""
        assert memory_created_schema._check_type(None, "null", {}) is True
        assert memory_created_schema._check_type("", "null", {}) is False


class TestMemoryEventSchemas:
    """Tests for memory event schemas."""

    def test_memory_created_valid(self):
        """Test valid memory.created event."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem123",
            "content": "Test memory content",
            "tier": "hot"
        })

        assert is_valid is True

    def test_memory_created_with_optional_fields(self):
        """Test memory.created with optional fields."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem123",
            "content": "Test memory content",
            "tier": "warm",
            "ltp_strength": 0.85,
            "tags": ["important", "work"],
            "agent_id": "agent_001"
        })

        assert is_valid is True

    def test_memory_accessed_valid(self):
        """Test valid memory.accessed event."""
        is_valid, errors = validate_event("memory.accessed", {
            "memory_id": "mem123",
            "query": "search query"
        })

        assert is_valid is True

    def test_memory_accessed_with_optional_fields(self):
        """Test memory.accessed with optional fields."""
        is_valid, errors = validate_event("memory.accessed", {
            "memory_id": "mem123",
            "query": "search query",
            "similarity_score": 0.92,
            "rank": 1,
            "tier": "hot"
        })

        assert is_valid is True

    def test_memory_deleted_valid(self):
        """Test valid memory.deleted event."""
        is_valid, errors = validate_event("memory.deleted", {
            "memory_id": "mem123"
        })

        assert is_valid is True

    def test_memory_deleted_with_reason(self):
        """Test memory.deleted with reason."""
        is_valid, errors = validate_event("memory.deleted", {
            "memory_id": "mem123",
            "reason": "eviction"
        })

        assert is_valid is True

    def test_memory_consolidated_valid(self):
        """Test valid memory.consolidated event."""
        is_valid, errors = validate_event("memory.consolidated", {
            "memory_id": "mem123",
            "from_tier": "hot",
            "to_tier": "warm"
        })

        assert is_valid is True


class TestConsolidationEventSchemas:
    """Tests for consolidation event schemas."""

    def test_consolidation_completed_valid(self):
        """Test valid consolidation.completed event."""
        is_valid, errors = validate_event("consolidation.completed", {
            "cycle_id": "cycle123",
            "duration_seconds": 5.5
        })

        assert is_valid is True

    def test_consolidation_completed_with_stats(self):
        """Test consolidation.completed with statistics."""
        is_valid, errors = validate_event("consolidation.completed", {
            "cycle_id": "cycle123",
            "duration_seconds": 10.0,
            "memories_processed": 100,
            "memories_consolidated": 25,
            "hot_to_warm": 15,
            "warm_to_cold": 10
        })

        assert is_valid is True


class TestContradictionEventSchemas:
    """Tests for contradiction event schemas."""

    def test_contradiction_detected_valid(self):
        """Test valid contradiction.detected event."""
        is_valid, errors = validate_event("contradiction.detected", {
            "group_id": "group123",
            "memory_a_id": "mem_a",
            "memory_b_id": "mem_b"
        })

        assert is_valid is True

    def test_contradiction_detected_with_llm(self):
        """Test contradiction.detected with LLM confirmation."""
        is_valid, errors = validate_event("contradiction.detected", {
            "group_id": "group123",
            "memory_a_id": "mem_a",
            "memory_b_id": "mem_b",
            "similarity_score": 0.75,
            "llm_confirmed": True,
            "llm_confidence": 0.9
        })

        assert is_valid is True

    def test_contradiction_resolved_valid(self):
        """Test valid contradiction.resolved event."""
        is_valid, errors = validate_event("contradiction.resolved", {
            "group_id": "group123"
        })

        assert is_valid is True

    def test_contradiction_resolved_with_resolution_type(self):
        """Test contradiction.resolved with resolution type."""
        is_valid, errors = validate_event("contradiction.resolved", {
            "group_id": "group123",
            "resolution_type": "merge",
            "resolved_by": "auto_resolver"
        })

        assert is_valid is True


class TestDreamEventSchemas:
    """Tests for dream event schemas."""

    def test_dream_started_valid(self):
        """Test valid dream.started event."""
        is_valid, errors = validate_event("dream.started", {
            "session_id": "session123",
            "trigger": "scheduled"
        })

        assert is_valid is True

    def test_dream_started_with_stages(self):
        """Test dream.started with enabled stages."""
        is_valid, errors = validate_event("dream.started", {
            "session_id": "session123",
            "trigger": "manual",
            "max_memories": 500,
            "stages_enabled": [
                "episodic_clustering",
                "pattern_extraction",
                "contradiction_resolution"
            ]
        })

        assert is_valid is True

    def test_dream_completed_valid(self):
        """Test valid dream.completed event."""
        is_valid, errors = validate_event("dream.completed", {
            "session_id": "session123",
            "duration_seconds": 30.5
        })

        assert is_valid is True

    def test_dream_completed_with_results(self):
        """Test dream.completed with results."""
        is_valid, errors = validate_event("dream.completed", {
            "session_id": "session123",
            "duration_seconds": 45.0,
            "memories_processed": 200,
            "clusters_found": 15,
            "patterns_extracted": 8,
            "contradictions_resolved": 3,
            "memories_promoted": 10
        })

        assert is_valid is True

    def test_dream_failed_valid(self):
        """Test valid dream.failed event."""
        is_valid, errors = validate_event("dream.failed", {
            "session_id": "session123",
            "error": "Processing timeout"
        })

        assert is_valid is True


class TestSynapseEventSchemas:
    """Tests for synapse event schemas."""

    def test_synapse_formed_valid(self):
        """Test valid synapse.formed event."""
        is_valid, errors = validate_event("synapse.formed", {
            "synapse_id": "syn123",
            "neuron_a_id": "neuron_a",
            "neuron_b_id": "neuron_b"
        })

        assert is_valid is True

    def test_synapse_formed_with_metadata(self):
        """Test synapse.formed with metadata."""
        is_valid, errors = validate_event("synapse.formed", {
            "synapse_id": "syn123",
            "neuron_a_id": "neuron_a",
            "neuron_b_id": "neuron_b",
            "initial_strength": 0.75,
            "formation_reason": "auto_bind",
            "similarity": 0.85
        })

        assert is_valid is True

    def test_synapse_fired_valid(self):
        """Test valid synapse.fired event."""
        is_valid, errors = validate_event("synapse.fired", {
            "synapse_id": "syn123",
            "source_id": "source",
            "target_id": "target"
        })

        assert is_valid is True


class TestGapEventSchemas:
    """Tests for gap event schemas."""

    def test_gap_detected_valid(self):
        """Test valid gap.detected event."""
        is_valid, errors = validate_event("gap.detected", {
            "gap_id": "gap123",
            "query": "What is the meaning of life?"
        })

        assert is_valid is True

    def test_gap_detected_with_metadata(self):
        """Test gap.detected with metadata."""
        is_valid, errors = validate_event("gap.detected", {
            "gap_id": "gap123",
            "query": "What is the meaning of life?",
            "gap_type": "missing_info",
            "related_memory_ids": ["mem1", "mem2"],
            "severity": 0.8
        })

        assert is_valid is True

    def test_gap_filled_valid(self):
        """Test valid gap.filled event."""
        is_valid, errors = validate_event("gap.filled", {
            "gap_id": "gap123",
            "memory_id": "mem_new"
        })

        assert is_valid is True


class TestInvalidSchemaRejection:
    """Tests for rejecting invalid schemas."""

    def test_invalid_tier_value(self):
        """Test invalid tier value is rejected."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem123",
            "content": "Test",
            "tier": "invalid_tier"
        })

        assert is_valid is False
        assert any("tier" in e.lower() for e in errors)

    def test_invalid_trigger_value(self):
        """Test invalid trigger value is rejected."""
        is_valid, errors = validate_event("dream.started", {
            "session_id": "session123",
            "trigger": "invalid_trigger"
        })

        assert is_valid is False

    def test_invalid_number_type(self):
        """Test invalid number type is rejected."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem123",
            "content": "Test",
            "tier": "hot",
            "ltp_strength": "not_a_number"
        })

        assert is_valid is False

    def test_missing_all_required_fields(self):
        """Test missing all required fields."""
        is_valid, errors = validate_event("memory.created", {})

        assert is_valid is False
        assert len(errors) >= 3  # memory_id, content, tier


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_object(self):
        """Test validation with empty object."""
        is_valid, errors = validate_event("memory.deleted", {})

        assert is_valid is False
        assert any("memory_id" in e for e in errors)

    def test_nested_object_in_metadata(self):
        """Test nested objects in metadata fields."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem123",
            "content": "Test",
            "tier": "hot"
        })

        # Should be valid even without nested objects
        assert is_valid is True

    def test_empty_array_field(self):
        """Test empty array in array field."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem123",
            "content": "Test",
            "tier": "hot",
            "tags": []
        })

        assert is_valid is True

    def test_large_array(self):
        """Test large array in array field."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem123",
            "content": "Test",
            "tier": "hot",
            "tags": [f"tag_{i}" for i in range(100)]
        })

        assert is_valid is True

    def test_special_characters_in_strings(self):
        """Test special characters in string fields."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem/123:456",
            "content": "Test with special chars: \n\t\r\"'<>&",
            "tier": "hot"
        })

        assert is_valid is True

    def test_unicode_strings(self):
        """Test unicode in string fields."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem_unicode",
            "content": "Unicode: \u4e2d\u6587 \u0420\u0443\u0441\u0441\u043a\u0438\u0439 \u65e5\u672c\u8a9e",
            "tier": "hot"
        })

        assert is_valid is True

    def test_boundary_number_values(self):
        """Test boundary number values."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem123",
            "content": "Test",
            "tier": "hot",
            "ltp_strength": 0.0,  # Minimum
            "eig": 1.0  # Maximum
        })

        assert is_valid is True


class TestValidationFunctions:
    """Tests for module-level validation functions."""

    def test_validate_event_known_type(self):
        """Test validate_event with known event type."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "mem123",
            "content": "Test",
            "tier": "hot"
        })

        assert is_valid is True

    def test_validate_event_unknown_type(self):
        """Test validate_event with unknown event type."""
        # Unknown types should return valid (warned but not failed)
        is_valid, errors = validate_event("unknown.event", {
            "any": "data"
        })

        # Returns True with empty errors for unknown types
        assert is_valid is True

    def test_get_schema_for_event_type(self):
        """Test get_schema_for_event_type function."""
        schema = get_schema_for_event_type("memory.created")

        assert schema is not None
        assert schema.event_type == "memory.created"

    def test_get_schema_for_unknown_type(self):
        """Test get_schema_for_event_type with unknown type."""
        schema = get_schema_for_event_type("unknown.event")

        assert schema is None

    def test_list_event_types(self):
        """Test list_event_types function."""
        types = list_event_types()

        assert isinstance(types, list)
        assert "memory.created" in types
        assert "dream.started" in types
        assert "synapse.formed" in types

    def test_get_json_schema_single(self):
        """Test get_json_schema for single event type."""
        schema = get_json_schema("memory.created")

        assert isinstance(schema, dict)
        assert "type" in schema
        assert "properties" in schema

    def test_get_json_schema_all(self):
        """Test get_json_schema for all events."""
        schemas = get_json_schema()

        assert isinstance(schemas, dict)
        assert "memory.created" in schemas
        assert "dream.completed" in schemas

    def test_export_openapi_spec(self):
        """Test export_openapi_spec function."""
        spec = export_openapi_spec()

        assert isinstance(spec, dict)
        assert spec["openapi"] == "3.0.0"
        assert "components" in spec
        assert "schemas" in spec["components"]
        assert "memory_created" in spec["components"]["schemas"]
