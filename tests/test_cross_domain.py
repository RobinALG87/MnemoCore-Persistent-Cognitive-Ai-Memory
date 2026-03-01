"""
Comprehensive Tests for Cross-Domain Synapse Builder
=====================================================

Tests automatic cross-domain association creation.

Coverage:
- Domain inference from content keywords
- Cross-domain synapse creation when overlap detected
- Buffer management and trim logic
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from mnemocore.core.cross_domain import (
    CrossDomainSynapseBuilder,
    infer_domain,
    DOMAINS,
    DEFAULT_DOMAIN,
    DOMAIN_KEYWORDS,
    CROSS_DOMAIN_WEIGHT,
    COOCCURRENCE_WINDOW_HOURS,
)
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.node import MemoryNode


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_engine():
    """Create a mock HAIMEngine for testing."""
    engine = MagicMock()
    engine.synapse_index = MagicMock()
    engine.synapse_index.add_or_strengthen = MagicMock()
    return engine


@pytest.fixture
def cross_domain_builder(mock_engine):
    """Create a CrossDomainSynapseBuilder instance."""
    return CrossDomainSynapseBuilder(
        engine=mock_engine,
        window_hours=2.0,
        cross_domain_weight=0.2,
    )


def create_test_node(node_id: str, content: str, metadata: dict = None, dimension: int = 1024) -> MemoryNode:
    """Helper to create test memory nodes."""
    node = MemoryNode(
        id=node_id,
        content=content,
        hdv=BinaryHDV.random(dimension),
    )
    if metadata:
        node.metadata = metadata
    return node


# =============================================================================
# Domain Inference Tests
# =============================================================================

class TestInferDomain:
    """Test domain inference from content and metadata."""

    def test_infer_from_metadata(self):
        """Should use domain from metadata if present."""
        metadata = {"domain": "strategic"}
        result = infer_domain("any content", metadata)

        assert result == "strategic"

    def test_infer_from_metadata_case_insensitive(self):
        """Should handle case-insensitive domain in metadata."""
        metadata = {"domain": "STRATEGIC"}
        result = infer_domain("any content", metadata)

        assert result == "strategic"

    def test_infer_invalid_domain_defaults(self):
        """Should default for invalid domain in metadata."""
        metadata = {"domain": "invalid_domain"}
        result = infer_domain("any content", metadata)

        assert result == DEFAULT_DOMAIN

    def test_infer_strategic_keywords(self):
        """Should infer strategic domain from keywords."""
        strategic_texts = [
            "We need to set a clear goal for Q4",
            "Our strategy focuses on market expansion",
            "The roadmap includes three major milestones",
            "Key decision: pivot to enterprise customers",
        ]

        for text in strategic_texts:
            result = infer_domain(text)
            assert result == "strategic", f"Failed for: {text}"

    def test_infer_personal_keywords(self):
        """Should infer personal domain from keywords."""
        personal_texts = [
            "I prefer dark mode in all applications",
            "My habit is to check emails first thing",
            "I feel this approach is right",
            "John is a trusted colleague",
        ]

        for text in personal_texts:
            result = infer_domain(text)
            assert result == "personal", f"Failed for: {text}"

    def test_infer_operational_keywords(self):
        """Should infer operational domain from keywords."""
        operational_texts = [
            "The code needs refactoring",
            "Fix this bug in the API",
            "Implement the new feature",
            "Deploy to production",
            "Write tests for this module",
        ]

        for text in operational_texts:
            result = infer_domain(text)
            assert result == "operational", f"Failed for: {text}"

    def test_infer_multiple_keyword_matches(self):
        """Should pick domain with most keyword matches."""
        # This text has 2 operational keywords, 1 strategic
        text = "We need to fix the code and deploy it as part of our goal"
        result = infer_domain(text)

        # Operational should win (fix, deploy, code vs goal)
        assert result == "operational"

    def test_infer_no_matches_defaults(self):
        """Should default to operational when no keywords match."""
        result = infer_domain("Lorem ipsum dolor sit amet")

        assert result == DEFAULT_DOMAIN

    def test_infer_empty_content(self):
        """Should handle empty content."""
        result = infer_domain("")

        assert result == DEFAULT_DOMAIN


# =============================================================================
# CrossDomainSynapseBuilder Initialization Tests
# =============================================================================

class TestCrossDomainSynapseBuilderInit:
    """Test builder initialization."""

    def test_init_with_defaults(self, mock_engine):
        """Should initialize with default values."""
        builder = CrossDomainSynapseBuilder(engine=mock_engine)

        assert builder.window == timedelta(hours=COOCCURRENCE_WINDOW_HOURS)
        assert builder.weight == CROSS_DOMAIN_WEIGHT
        assert builder._buffer == []

    def test_init_with_custom_values(self, mock_engine):
        """Should accept custom window and weight."""
        builder = CrossDomainSynapseBuilder(
            engine=mock_engine,
            window_hours=4.0,
            cross_domain_weight=0.3,
        )

        assert builder.window == timedelta(hours=4.0)
        assert builder.weight == 0.3

    def test_init_without_engine(self):
        """Should work without engine (for testing)."""
        builder = CrossDomainSynapseBuilder(engine=None)

        assert builder.engine is None


# =============================================================================
# tag_domain Tests
# =============================================================================

class TestTagDomain:
    """Test domain tagging."""

    def test_tag_domain_adds_to_metadata(self, cross_domain_builder):
        """Should add domain to node metadata."""
        node = create_test_node("test-1", "Fix the bug in the code")

        domain = cross_domain_builder.tag_domain(node)

        assert domain == "operational"
        assert node.metadata["domain"] == "operational"

    def test_tag_domain_respects_existing(self, cross_domain_builder):
        """Should respect existing domain in metadata."""
        node = create_test_node("test-1", "Fix the bug", {"domain": "strategic"})

        domain = cross_domain_builder.tag_domain(node)

        assert domain == "strategic"


# =============================================================================
# Synapse Creation Tests
# =============================================================================

class TestCreateSynapse:
    """Test synapse creation."""

    @pytest.mark.asyncio
    async def test_create_synapse_with_engine(self, cross_domain_builder, mock_engine):
        """Should create synapse via engine's synapse_index."""
        await cross_domain_builder._create_synapse("node-a", "node-b")

        mock_engine.synapse_index.add_or_strengthen.assert_called_once_with(
            "node-a", "node-b", delta=0.2
        )

    @pytest.mark.asyncio
    async def test_create_synapse_without_engine(self, mock_engine):
        """Should handle missing engine gracefully."""
        builder = CrossDomainSynapseBuilder(engine=None)

        # Should not raise
        await builder._create_synapse("node-a", "node-b")

    @pytest.mark.asyncio
    async def test_create_synapse_exception_handled(self, cross_domain_builder, mock_engine):
        """Should handle synapse creation exceptions."""
        mock_engine.synapse_index.add_or_strengthen.side_effect = Exception("Synapse error")

        # Should not raise
        await cross_domain_builder._create_synapse("node-a", "node-b")


# =============================================================================
# process_new_memory Tests
# =============================================================================

class TestProcessNewMemory:
    """Test processing new memories."""

    @pytest.mark.asyncio
    async def test_process_new_memory_tags_domain(self, cross_domain_builder):
        """Should tag domain on new memory."""
        node = create_test_node("test-1", "Fix the bug")

        await cross_domain_builder.process_new_memory(node)

        assert node.metadata.get("domain") == "operational"

    @pytest.mark.asyncio
    async def test_process_new_memory_adds_to_buffer(self, cross_domain_builder):
        """Should add new memory to buffer."""
        node = create_test_node("test-1", "Fix the bug")

        await cross_domain_builder.process_new_memory(node)

        assert len(cross_domain_builder._buffer) == 1
        assert cross_domain_builder._buffer[0][0] == "test-1"

    @pytest.mark.asyncio
    async def test_process_new_memory_creates_cross_domain_synapse(self, cross_domain_builder, mock_engine):
        """Should create synapse between different domains."""
        # Add a strategic node first
        strategic_node = create_test_node("strategic-1", "Our goal is to expand")
        await cross_domain_builder.process_new_memory(strategic_node)

        # Then add an operational node
        operational_node = create_test_node("operational-1", "Fix the code bug")
        await cross_domain_builder.process_new_memory(operational_node)

        # Should have created cross-domain synapse
        mock_engine.synapse_index.add_or_strengthen.assert_called()

    @pytest.mark.asyncio
    async def test_process_new_memory_no_same_domain_synapse(self, cross_domain_builder, mock_engine):
        """Should not create synapse between same domain nodes."""
        # Add two operational nodes
        node1 = create_test_node("op-1", "Fix the bug in the code")
        node2 = create_test_node("op-2", "Deploy the new API feature")

        await cross_domain_builder.process_new_memory(node1)
        await cross_domain_builder.process_new_memory(node2)

        # Should not have created synapse (same domain)
        mock_engine.synapse_index.add_or_strengthen.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_returns_pairs(self, cross_domain_builder):
        """Should return list of created pairs."""
        # Add strategic node
        strategic_node = create_test_node("strategic-1", "Our goal is to expand")
        await cross_domain_builder.process_new_memory(strategic_node)

        # Add operational node
        operational_node = create_test_node("operational-1", "Fix the code")
        pairs = await cross_domain_builder.process_new_memory(operational_node)

        assert len(pairs) == 1
        assert pairs[0] == ("operational-1", "strategic-1")


# =============================================================================
# Buffer Management Tests
# =============================================================================

class TestBufferManagement:
    """Test buffer management and trimming."""

    @pytest.mark.asyncio
    async def test_buffer_trims_stale_entries(self):
        """Should trim entries older than window."""
        # Create builder with short window
        builder = CrossDomainSynapseBuilder(engine=None, window_hours=0.001)  # ~3.6 seconds

        # Add a node
        node = create_test_node("test-1", "Fix the bug")
        await builder.process_new_memory(node)

        # Manually age the buffer entry
        builder._buffer[0] = (
            "test-1",
            "operational",
            datetime.now(timezone.utc) - timedelta(hours=1)
        )

        # Add another node - should trigger trim
        node2 = create_test_node("test-2", "Another task")
        await builder.process_new_memory(node2)

        # Old entry should be removed
        assert len(builder._buffer) == 1
        assert builder._buffer[0][0] == "test-2"

    @pytest.mark.asyncio
    async def test_buffer_keeps_recent_entries(self, cross_domain_builder):
        """Should keep entries within window."""
        # Add multiple nodes quickly
        for i in range(5):
            node = create_test_node(f"test-{i}", f"Task {i}")
            await cross_domain_builder.process_new_memory(node)

        # All should be in buffer
        assert len(cross_domain_builder._buffer) == 5

    def test_clear_buffer(self, cross_domain_builder):
        """clear_buffer should remove all entries."""
        cross_domain_builder._buffer = [("id1", "domain1", datetime.now(timezone.utc))]

        cross_domain_builder.clear_buffer()

        assert cross_domain_builder._buffer == []


# =============================================================================
# scan_recent Tests
# =============================================================================

class TestScanRecent:
    """Test scanning for recent cross-domain pairs."""

    @pytest.mark.asyncio
    async def test_scan_recent_finds_pairs(self, cross_domain_builder, mock_engine):
        """Should find cross-domain pairs in buffer."""
        # Manually populate buffer with cross-domain entries
        now = datetime.now(timezone.utc)
        cross_domain_builder._buffer = [
            ("strategic-1", "strategic", now),
            ("operational-1", "operational", now),
            ("personal-1", "personal", now),
        ]

        pairs = await cross_domain_builder.scan_recent(hours=1.0)

        # Should have created pairs between different domains
        assert len(pairs) >= 1

    @pytest.mark.asyncio
    async def test_scan_recent_respects_hours(self, cross_domain_builder):
        """Should only include entries within specified hours."""
        now = datetime.now(timezone.utc)
        cross_domain_builder._buffer = [
            ("recent", "strategic", now),
            ("old", "operational", now - timedelta(hours=3)),
        ]

        # Scan only last hour
        pairs = await cross_domain_builder.scan_recent(hours=1.0)

        # Only recent entry in buffer for this scan
        # No pairs possible with just one entry
        assert len(pairs) == 0

    @pytest.mark.asyncio
    async def test_scan_recent_empty_buffer(self, cross_domain_builder):
        """Should handle empty buffer."""
        pairs = await cross_domain_builder.scan_recent(hours=1.0)

        assert pairs == []


# =============================================================================
# Integration Tests
# =============================================================================

class TestCrossDomainIntegration:
    """Integration tests for cross-domain synapse creation."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_engine):
        """Test complete workflow from memory storage to synapse creation."""
        builder = CrossDomainSynapseBuilder(
            engine=mock_engine,
            window_hours=2.0,
            cross_domain_weight=0.2,
        )

        # Store strategic memory
        strategic = create_test_node("str-1", "Our goal is to increase revenue")
        pairs1 = await builder.process_new_memory(strategic)
        assert len(pairs1) == 0  # No other memories yet

        # Store personal memory
        personal = create_test_node("per-1", "I prefer concise reports")
        pairs2 = await builder.process_new_memory(personal)
        assert len(pairs2) == 1  # Cross-domain with strategic

        # Store operational memory
        operational = create_test_node("op-1", "Fix the API bug")
        pairs3 = await builder.process_new_memory(operational)
        assert len(pairs3) == 2  # Cross-domain with both strategic and personal

        # Verify all domains tagged correctly
        assert strategic.metadata["domain"] == "strategic"
        assert personal.metadata["domain"] == "personal"
        assert operational.metadata["domain"] == "operational"

        # Verify buffer contains all
        assert len(builder._buffer) == 3

    @pytest.mark.asyncio
    async def test_multiple_same_domain_no_synapse(self, mock_engine):
        """Multiple same-domain memories should not create synapses."""
        builder = CrossDomainSynapseBuilder(engine=mock_engine)

        # Store multiple operational memories
        for i in range(5):
            node = create_test_node(f"op-{i}", f"Task {i}: fix bug {i}")
            await builder.process_new_memory(node)

        # No cross-domain synapses should have been created
        mock_engine.synapse_index.add_or_strengthen.assert_not_called()

        # But buffer should have all
        assert len(builder._buffer) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
