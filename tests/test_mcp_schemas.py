"""
Tests for MCP Schemas Module
============================
Comprehensive tests for Pydantic model validation for MCP tool inputs.

Tests cover:
- Pydantic model validation for each input schema
- Field constraints (min/max values, string patterns)
- Required vs optional fields
"""

import pytest
from pydantic import ValidationError

from mnemocore.mcp.schemas import (
    StoreToolInput,
    QueryToolInput,
    MemoryIdInput,
    ObserveToolInput,
    ContextToolInput,
    EpisodeToolInput,
    SynthesizeToolInput,
    DreamToolInput,
    ExportToolInput,
    ToolResult,
)
from mnemocore.core.exceptions import ValidationError as MnemoValidationError


class TestStoreToolInput:
    """Tests for StoreToolInput schema."""

    def test_valid_input(self):
        """Test valid StoreToolInput creation."""
        inp = StoreToolInput(
            content="This is valid content"
        )

        assert inp.content == "This is valid content"
        assert inp.metadata is None
        assert inp.agent_id is None
        assert inp.ttl is None

    def test_valid_input_with_all_fields(self):
        """Test valid StoreToolInput with all optional fields."""
        inp = StoreToolInput(
            content="Test content",
            metadata={"key": "value"},
            agent_id="agent_123",
            ttl=3600
        )

        assert inp.content == "Test content"
        assert inp.metadata == {"key": "value"}
        assert inp.agent_id == "agent_123"
        assert inp.ttl == 3600

    def test_content_min_length_constraint(self):
        """Test content must have at least 1 character."""
        with pytest.raises(ValidationError) as exc_info:
            StoreToolInput(content="")

        errors = exc_info.value.errors()
        assert any("min_length" in str(e).lower() or "at least" in str(e).lower()
                   for e in errors)

    def test_content_max_length_constraint(self):
        """Test content cannot exceed 100,000 characters."""
        long_content = "x" * 100001

        with pytest.raises(ValidationError) as exc_info:
            StoreToolInput(content=long_content)

        errors = exc_info.value.errors()
        assert any("max_length" in str(e).lower() or "at most" in str(e).lower()
                   for e in errors)

    def test_content_at_max_length(self):
        """Test content at exactly max length is valid."""
        content = "x" * 100000

        inp = StoreToolInput(content=content)
        assert len(inp.content) == 100000

    def test_agent_id_max_length(self):
        """Test agent_id max length constraint."""
        long_id = "x" * 257

        with pytest.raises(ValidationError):
            StoreToolInput(content="test", agent_id=long_id)

    def test_ttl_must_be_positive(self):
        """Test TTL must be greater than 0."""
        with pytest.raises(ValidationError):
            StoreToolInput(content="test", ttl=0)

        with pytest.raises(ValidationError):
            StoreToolInput(content="test", ttl=-1)

    def test_ttl_positive_value(self):
        """Test valid positive TTL."""
        inp = StoreToolInput(content="test", ttl=1)
        assert inp.ttl == 1

        inp = StoreToolInput(content="test", ttl=86400)
        assert inp.ttl == 86400


class TestQueryToolInput:
    """Tests for QueryToolInput schema."""

    def test_valid_input(self):
        """Test valid QueryToolInput creation."""
        inp = QueryToolInput(query="search query")

        assert inp.query == "search query"
        assert inp.top_k == 5  # Default
        assert inp.agent_id is None

    def test_valid_input_with_all_fields(self):
        """Test valid QueryToolInput with all optional fields."""
        inp = QueryToolInput(
            query="search query",
            top_k=20,
            agent_id="agent_123"
        )

        assert inp.query == "search query"
        assert inp.top_k == 20
        assert inp.agent_id == "agent_123"

    def test_query_min_length(self):
        """Test query must have at least 1 character."""
        with pytest.raises(ValidationError):
            QueryToolInput(query="")

    def test_query_max_length(self):
        """Test query cannot exceed 10,000 characters."""
        long_query = "x" * 10001

        with pytest.raises(ValidationError):
            QueryToolInput(query=long_query)

    def test_top_k_min_value(self):
        """Test top_k must be at least 1."""
        with pytest.raises(ValidationError):
            QueryToolInput(query="test", top_k=0)

    def test_top_k_max_value(self):
        """Test top_k cannot exceed 100."""
        with pytest.raises(ValidationError):
            QueryToolInput(query="test", top_k=101)

    def test_top_k_boundary_values(self):
        """Test top_k at boundary values."""
        inp = QueryToolInput(query="test", top_k=1)
        assert inp.top_k == 1

        inp = QueryToolInput(query="test", top_k=100)
        assert inp.top_k == 100


class TestMemoryIdInput:
    """Tests for MemoryIdInput schema."""

    def test_valid_input(self):
        """Test valid MemoryIdInput creation."""
        inp = MemoryIdInput(memory_id="mem_123456")

        assert inp.memory_id == "mem_123456"

    def test_memory_id_min_length(self):
        """Test memory_id must have at least 1 character."""
        with pytest.raises(ValidationError):
            MemoryIdInput(memory_id="")

    def test_memory_id_max_length(self):
        """Test memory_id cannot exceed 256 characters."""
        long_id = "x" * 257

        with pytest.raises(ValidationError):
            MemoryIdInput(memory_id=long_id)

    def test_memory_id_at_max_length(self):
        """Test memory_id at exactly max length is valid."""
        memory_id = "x" * 256

        inp = MemoryIdInput(memory_id=memory_id)
        assert len(inp.memory_id) == 256


class TestObserveToolInput:
    """Tests for ObserveToolInput schema (Phase 5)."""

    def test_valid_input(self):
        """Test valid ObserveToolInput creation."""
        inp = ObserveToolInput(
            agent_id="agent_123",
            content="Observation content"
        )

        assert inp.agent_id == "agent_123"
        assert inp.content == "Observation content"
        assert inp.kind == "observation"  # Default
        assert inp.importance == 0.5  # Default
        assert inp.tags is None

    def test_valid_input_with_all_fields(self):
        """Test valid ObserveToolInput with all fields."""
        inp = ObserveToolInput(
            agent_id="agent_123",
            content="Observation",
            kind="event",
            importance=0.9,
            tags=["important", "work"]
        )

        assert inp.kind == "event"
        assert inp.importance == 0.9
        assert inp.tags == ["important", "work"]

    def test_agent_id_required(self):
        """Test agent_id is required."""
        with pytest.raises(ValidationError):
            ObserveToolInput(content="test")

    def test_content_required(self):
        """Test content is required."""
        with pytest.raises(ValidationError):
            ObserveToolInput(agent_id="agent_123")

    def test_importance_min_value(self):
        """Test importance must be >= 0.0."""
        with pytest.raises(ValidationError):
            ObserveToolInput(
                agent_id="agent_123",
                content="test",
                importance=-0.1
            )

    def test_importance_max_value(self):
        """Test importance must be <= 1.0."""
        with pytest.raises(ValidationError):
            ObserveToolInput(
                agent_id="agent_123",
                content="test",
                importance=1.1
            )

    def test_importance_boundary_values(self):
        """Test importance at boundary values."""
        inp = ObserveToolInput(
            agent_id="agent_123",
            content="test",
            importance=0.0
        )
        assert inp.importance == 0.0

        inp = ObserveToolInput(
            agent_id="agent_123",
            content="test",
            importance=1.0
        )
        assert inp.importance == 1.0

    def test_kind_max_length(self):
        """Test kind max length constraint."""
        long_kind = "x" * 65

        with pytest.raises(ValidationError):
            ObserveToolInput(
                agent_id="agent_123",
                content="test",
                kind=long_kind
            )


class TestContextToolInput:
    """Tests for ContextToolInput schema (Phase 5)."""

    def test_valid_input(self):
        """Test valid ContextToolInput creation."""
        inp = ContextToolInput(agent_id="agent_123")

        assert inp.agent_id == "agent_123"
        assert inp.limit == 16  # Default

    def test_valid_input_with_limit(self):
        """Test valid ContextToolInput with custom limit."""
        inp = ContextToolInput(agent_id="agent_123", limit=50)

        assert inp.limit == 50

    def test_limit_min_value(self):
        """Test limit must be at least 1."""
        with pytest.raises(ValidationError):
            ContextToolInput(agent_id="agent_123", limit=0)

    def test_limit_max_value(self):
        """Test limit cannot exceed 100."""
        with pytest.raises(ValidationError):
            ContextToolInput(agent_id="agent_123", limit=101)

    def test_limit_boundary_values(self):
        """Test limit at boundary values."""
        inp = ContextToolInput(agent_id="agent_123", limit=1)
        assert inp.limit == 1

        inp = ContextToolInput(agent_id="agent_123", limit=100)
        assert inp.limit == 100


class TestEpisodeToolInput:
    """Tests for EpisodeToolInput schema (Phase 5)."""

    def test_valid_input(self):
        """Test valid EpisodeToolInput creation."""
        inp = EpisodeToolInput(
            agent_id="agent_123",
            goal="Complete the task"
        )

        assert inp.agent_id == "agent_123"
        assert inp.goal == "Complete the task"
        assert inp.context is None

    def test_valid_input_with_context(self):
        """Test valid EpisodeToolInput with context."""
        inp = EpisodeToolInput(
            agent_id="agent_123",
            goal="Complete the task",
            context="Additional context"
        )

        assert inp.context == "Additional context"

    def test_goal_min_length(self):
        """Test goal must have at least 1 character."""
        with pytest.raises(ValidationError):
            EpisodeToolInput(agent_id="agent_123", goal="")

    def test_goal_max_length(self):
        """Test goal cannot exceed 10,000 characters."""
        long_goal = "x" * 10001

        with pytest.raises(ValidationError):
            EpisodeToolInput(agent_id="agent_123", goal=long_goal)


class TestSynthesizeToolInput:
    """Tests for SynthesizeToolInput schema (Phase 4.5)."""

    def test_valid_input(self):
        """Test valid SynthesizeToolInput creation."""
        inp = SynthesizeToolInput(query="complex query")

        assert inp.query == "complex query"
        assert inp.top_k == 10  # Default
        assert inp.max_depth == 3  # Default
        assert inp.context_text is None
        assert inp.project_id is None

    def test_valid_input_with_all_fields(self):
        """Test valid SynthesizeToolInput with all fields."""
        inp = SynthesizeToolInput(
            query="complex query",
            top_k=25,
            max_depth=5,
            context_text="Additional context",
            project_id="project_123"
        )

        assert inp.top_k == 25
        assert inp.max_depth == 5
        assert inp.context_text == "Additional context"
        assert inp.project_id == "project_123"

    def test_query_min_length(self):
        """Test query must have at least 3 characters."""
        with pytest.raises(ValidationError):
            SynthesizeToolInput(query="ab")

    def test_query_max_length(self):
        """Test query cannot exceed 4,096 characters."""
        long_query = "x" * 4097

        with pytest.raises(ValidationError):
            SynthesizeToolInput(query=long_query)

    def test_top_k_constraints(self):
        """Test top_k constraints."""
        # Min value
        with pytest.raises(ValidationError):
            SynthesizeToolInput(query="test", top_k=0)

        # Max value
        with pytest.raises(ValidationError):
            SynthesizeToolInput(query="test", top_k=51)

        # Valid boundaries
        inp = SynthesizeToolInput(query="test", top_k=1)
        assert inp.top_k == 1

        inp = SynthesizeToolInput(query="test", top_k=50)
        assert inp.top_k == 50

    def test_max_depth_constraints(self):
        """Test max_depth constraints."""
        # Min value
        with pytest.raises(ValidationError):
            SynthesizeToolInput(query="test", max_depth=-1)

        # Max value
        with pytest.raises(ValidationError):
            SynthesizeToolInput(query="test", max_depth=6)

        # Valid boundaries
        inp = SynthesizeToolInput(query="test", max_depth=0)
        assert inp.max_depth == 0

        inp = SynthesizeToolInput(query="test", max_depth=5)
        assert inp.max_depth == 5


class TestDreamToolInput:
    """Tests for DreamToolInput schema."""

    def test_valid_input(self):
        """Test valid DreamToolInput creation."""
        inp = DreamToolInput()

        assert inp.max_cycles == 1  # Default
        assert inp.force_insight is False  # Default

    def test_valid_input_with_fields(self):
        """Test valid DreamToolInput with fields."""
        inp = DreamToolInput(max_cycles=5, force_insight=True)

        assert inp.max_cycles == 5
        assert inp.force_insight is True

    def test_max_cycles_constraints(self):
        """Test max_cycles constraints."""
        # Min value
        with pytest.raises(ValidationError):
            DreamToolInput(max_cycles=0)

        # Max value
        with pytest.raises(ValidationError):
            DreamToolInput(max_cycles=11)

        # Valid boundaries
        inp = DreamToolInput(max_cycles=1)
        assert inp.max_cycles == 1

        inp = DreamToolInput(max_cycles=10)
        assert inp.max_cycles == 10


class TestExportToolInput:
    """Tests for ExportToolInput schema."""

    def test_valid_input(self):
        """Test valid ExportToolInput creation."""
        inp = ExportToolInput()

        assert inp.agent_id is None
        assert inp.tier is None
        assert inp.limit == 100  # Default
        assert inp.include_metadata is True  # Default
        assert inp.format == "json"  # Default

    def test_valid_input_with_all_fields(self):
        """Test valid ExportToolInput with all fields."""
        inp = ExportToolInput(
            agent_id="agent_123",
            tier="warm",
            limit=500,
            include_metadata=False,
            format="jsonl"
        )

        assert inp.agent_id == "agent_123"
        assert inp.tier == "warm"
        assert inp.limit == 500
        assert inp.include_metadata is False
        assert inp.format == "jsonl"

    def test_tier_pattern_constraint(self):
        """Test tier must match pattern."""
        # Valid values
        inp = ExportToolInput(tier="hot")
        assert inp.tier == "hot"

        inp = ExportToolInput(tier="warm")
        assert inp.tier == "warm"

        inp = ExportToolInput(tier="cold")
        assert inp.tier == "cold"

        inp = ExportToolInput(tier="soul")
        assert inp.tier == "soul"

        # Invalid value
        with pytest.raises(ValidationError):
            ExportToolInput(tier="invalid")

    def test_format_pattern_constraint(self):
        """Test format must match pattern."""
        # Valid values
        inp = ExportToolInput(format="json")
        assert inp.format == "json"

        inp = ExportToolInput(format="jsonl")
        assert inp.format == "jsonl"

        # Invalid value
        with pytest.raises(ValidationError):
            ExportToolInput(format="xml")

    def test_limit_constraints(self):
        """Test limit constraints."""
        # Min value
        with pytest.raises(ValidationError):
            ExportToolInput(limit=0)

        # Max value
        with pytest.raises(ValidationError):
            ExportToolInput(limit=1001)

        # Valid boundaries
        inp = ExportToolInput(limit=1)
        assert inp.limit == 1

        inp = ExportToolInput(limit=1000)
        assert inp.limit == 1000


class TestToolResult:
    """Tests for ToolResult schema."""

    def test_success_result(self):
        """Test successful ToolResult creation."""
        result = ToolResult(ok=True, data={"key": "value"})

        assert result.ok is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        """Test error ToolResult creation."""
        result = ToolResult(ok=False, error="Something went wrong")

        assert result.ok is False
        assert result.error == "Something went wrong"
        assert result.data is None

    def test_ok_with_error_invalid(self):
        """Test that ok=True with error is invalid."""
        with pytest.raises((ValidationError, MnemoValidationError)):
            ToolResult(ok=True, error="Error message")

    def test_minimal_success(self):
        """Test minimal successful result."""
        result = ToolResult(ok=True)

        assert result.ok is True
        assert result.data is None
        assert result.error is None

    def test_minimal_error(self):
        """Test minimal error result."""
        result = ToolResult(ok=False)

        assert result.ok is False
        assert result.error is None
        assert result.data is None


class TestRequiredVsOptionalFields:
    """Tests for required vs optional field behavior."""

    def test_store_required_fields(self):
        """Test StoreToolInput required fields."""
        # Missing content should fail
        with pytest.raises(ValidationError):
            StoreToolInput()

    def test_query_required_fields(self):
        """Test QueryToolInput required fields."""
        # Missing query should fail
        with pytest.raises(ValidationError):
            QueryToolInput()

    def test_memory_id_required_fields(self):
        """Test MemoryIdInput required fields."""
        # Missing memory_id should fail
        with pytest.raises(ValidationError):
            MemoryIdInput()

    def test_observe_required_fields(self):
        """Test ObserveToolInput required fields."""
        # Missing agent_id
        with pytest.raises(ValidationError):
            ObserveToolInput(content="test")

        # Missing content
        with pytest.raises(ValidationError):
            ObserveToolInput(agent_id="agent_123")

    def test_context_required_fields(self):
        """Test ContextToolInput required fields."""
        # Missing agent_id
        with pytest.raises(ValidationError):
            ContextToolInput()

    def test_episode_required_fields(self):
        """Test EpisodeToolInput required fields."""
        # Missing agent_id
        with pytest.raises(ValidationError):
            EpisodeToolInput(goal="test")

        # Missing goal
        with pytest.raises(ValidationError):
            EpisodeToolInput(agent_id="agent_123")

    def test_synthesize_required_fields(self):
        """Test SynthesizeToolInput required fields."""
        # Missing query
        with pytest.raises(ValidationError):
            SynthesizeToolInput()

    def test_dream_all_optional(self):
        """Test DreamToolInput has no required fields."""
        # Should work with no arguments
        inp = DreamToolInput()
        assert inp.max_cycles == 1
        assert inp.force_insight is False

    def test_export_all_optional(self):
        """Test ExportToolInput has no required fields."""
        # Should work with no arguments
        inp = ExportToolInput()
        assert inp.limit == 100
        assert inp.format == "json"


class TestFieldConstraints:
    """Additional tests for field constraints."""

    def test_string_pattern_validation(self):
        """Test string pattern validation."""
        # tier must match ^(hot|warm|cold|soul)$
        valid_tiers = ["hot", "warm", "cold", "soul"]
        for tier in valid_tiers:
            inp = ExportToolInput(tier=tier)
            assert inp.tier == tier

    def test_numeric_range_validation(self):
        """Test numeric range validation."""
        # importance must be 0.0 to 1.0
        valid_values = [0.0, 0.5, 1.0]
        for val in valid_values:
            inp = ObserveToolInput(
                agent_id="agent_123",
                content="test",
                importance=val
            )
            assert inp.importance == val

    def test_integer_validation(self):
        """Test integer validation."""
        # top_k must be int
        inp = QueryToolInput(query="test", top_k=50)
        assert isinstance(inp.top_k, int)

    def test_boolean_validation(self):
        """Test boolean validation."""
        # force_insight must be bool
        inp = DreamToolInput(force_insight=True)
        assert inp.force_insight is True

        inp = DreamToolInput(force_insight=False)
        assert inp.force_insight is False

    def test_list_validation(self):
        """Test list validation."""
        # tags must be list
        inp = ObserveToolInput(
            agent_id="agent_123",
            content="test",
            tags=["a", "b", "c"]
        )
        assert inp.tags == ["a", "b", "c"]

    def test_dict_validation(self):
        """Test dict validation."""
        # metadata must be dict
        inp = StoreToolInput(
            content="test",
            metadata={"key": "value", "number": 42}
        )
        assert inp.metadata == {"key": "value", "number": 42}
