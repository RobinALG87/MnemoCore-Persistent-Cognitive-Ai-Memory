"""
Test suite for Reconstructive Memory Module (Phase 6.0)
=======================================================
Tests for memory reconstruction, fragment synthesis, and confidence scoring.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.mnemocore.cognitive.memory_reconstructor import (
    ReconstructiveRecall,
    ReconstructedMemory,
    ReconstructionConfig,
    ReconstructionResult,
    MemoryFragment,
    is_reconstructed_memory,
    get_reconstruction_metadata,
)


@pytest.fixture
def mock_engine():
    """Create a mock HAIMEngine for testing."""
    engine = MagicMock()
    engine.query = AsyncMock(return_value=[
        ("node_1", 0.75),
        ("node_2", 0.55),
        ("node_3", 0.35),
    ])
    engine.get_memory = AsyncMock()
    engine.store = AsyncMock(return_value="new_node_id")
    engine.record_retrieval_feedback = AsyncMock()
    return engine


@pytest.fixture
def mock_gap_detector():
    """Create a mock GapDetector for testing."""
    detector = MagicMock()
    detector.assess_query = AsyncMock(return_value=[])
    return detector


@pytest.fixture
def mock_nodes():
    """Create mock MemoryNode objects."""
    nodes = []

    for i, (content, tier) in enumerate([
        ("The meeting discussed project timeline and deliverables.", "HOT"),
        ("Project milestones were reviewed in the session.", "WARM"),
        ("Team members raised concerns about deadlines.", "COLD"),
    ]):
        node = MagicMock()
        node.id = f"node_{i+1}"
        node.content = content
        node.tier = tier
        node.metadata = {"tags": ["meeting", "project"]}
        nodes.append(node)

    # Configure get_memory to return appropriate nodes
    async def get_memory(node_id):
        idx = int(node_id.split("_")[1]) - 1
        if 0 <= idx < len(nodes):
            return nodes[idx]
        return None

    return nodes, get_memory


class TestMemoryFragment:
    """Tests for MemoryFragment dataclass."""

    def test_fragment_creation(self):
        """Test creating a memory fragment."""
        fragment = MemoryFragment(
            node_id="test_1",
            content="Test content",
            similarity=0.75,
            source_tier="HOT",
        )

        assert fragment.node_id == "test_1"
        assert fragment.content == "Test content"
        assert fragment.similarity == 0.75
        assert fragment.source_tier == "HOT"
        assert fragment.is_reconstructed_source is False

    def test_fragment_to_dict(self):
        """Test serializing fragment to dictionary."""
        fragment = MemoryFragment(
            node_id="test_1",
            content="Test content",
            similarity=0.75,
            metadata={"key": "value"},
        )

        data = fragment.to_dict()

        assert data["node_id"] == "test_1"
        assert data["content"] == "Test content"
        assert data["similarity"] == 0.75
        assert data["metadata"] == {"key": "value"}


class TestReconstructedMemory:
    """Tests for ReconstructedMemory dataclass."""

    def test_reconstructed_memory_creation(self):
        """Test creating a reconstructed memory."""
        fragments = [
            MemoryFragment(node_id="f1", content="Fragment 1", similarity=0.8),
            MemoryFragment(node_id="f2", content="Fragment 2", similarity=0.6),
        ]

        memory = ReconstructedMemory(
            content="Synthesized content",
            fragments=fragments,
            confidence=0.75,
            query_id="query_123",
        )

        assert memory.content == "Synthesized content"
        assert len(memory.fragments) == 2
        assert memory.confidence == 0.75
        assert memory.is_reconstructed is True
        assert memory.query_id == "query_123"

    def test_reconstructed_memory_to_dict(self):
        """Test serializing reconstructed memory to dictionary."""
        fragments = [
            MemoryFragment(node_id="f1", content="Fragment 1", similarity=0.8),
        ]

        memory = ReconstructedMemory(
            content="Synthesized",
            fragments=fragments,
            confidence=0.8,
            reconstruction_method="synthesis",
        )

        data = memory.to_dict()

        assert data["content"] == "Synthesized"
        assert data["confidence"] == 0.8
        assert data["reconstruction_method"] == "synthesis"
        assert data["is_reconstructed"] is True
        assert data["fragment_count"] == 1


class TestReconstructionResult:
    """Tests for ReconstructionResult dataclass."""

    def test_result_with_reconstruction(self):
        """Test result containing a reconstruction."""
        reconstructed = ReconstructedMemory(
            content="Synthesized answer",
            fragments=[],
            confidence=0.7,
        )

        result = ReconstructionResult(
            reconstructed=reconstructed,
            direct_matches=[("node_1", 0.6)],
            fragments=[],
            confidence_breakdown={"overall_confidence": 0.7},
            is_reconstructed=True,
        )

        assert result.is_reconstructed is True
        assert result.get_primary_content() == "Synthesized answer"

    def test_result_without_reconstruction(self):
        """Test result without reconstruction (direct match)."""
        result = ReconstructionResult(
            reconstructed=None,
            direct_matches=[("node_1", 0.9)],
            fragments=[],
            confidence_breakdown={"overall_confidence": 0.9},
            is_reconstructed=False,
        )

        assert result.is_reconstructed is False
        assert "Direct match" in result.get_primary_content()


class TestReconstructiveRecall:
    """Tests for ReconstructiveRecall engine."""

    @pytest.mark.asyncio
    async def test_recall_with_direct_match(self, mock_engine, mock_nodes):
        """Test recall when direct match is available."""
        nodes, get_memory_func = mock_nodes
        mock_engine.get_memory = get_memory_func

        # Lower threshold to trigger reconstruction
        config = ReconstructionConfig(
            synthesis_threshold=0.5,  # Below avg similarity
        )
        reconstructor = ReconstructiveRecall(mock_engine, config)

        result = await reconstructor.recall("What was discussed in the meeting?")

        assert result is not None
        assert isinstance(result, ReconstructionResult)

    @pytest.mark.asyncio
    async def test_recall_triggers_reconstruction(self, mock_engine, mock_nodes):
        """Test that low similarity triggers reconstruction."""
        nodes, get_memory_func = mock_nodes
        mock_engine.get_memory = get_memory_func

        # Set high threshold to force reconstruction
        config = ReconstructionConfig(
            synthesis_threshold=0.8,  # Above avg similarity of 0.55
        )
        reconstructor = ReconstructiveRecall(mock_engine, config)

        result = await reconstructor.recall("meeting discussion")

        assert result.is_reconstructed is True
        assert result.reconstructed is not None
        assert result.reconstructed.confidence > 0

    @pytest.mark.asyncio
    async def test_recall_with_gap_detection(self, mock_engine, mock_nodes, mock_gap_detector):
        """Test recall with gap detection enabled."""
        nodes, get_memory_func = mock_nodes
        mock_engine.get_memory = get_memory_func

        # Create a minimal gap record without importing from gap_detector
        # to avoid Prometheus metrics registration issues
        class MockGapRecord:
            def __init__(self):
                self.gap_id = "gap_1"
                self.query_text = "meeting discussion"
                self.detected_at = datetime.now(timezone.utc)
                self.last_seen = datetime.now(timezone.utc)
                self.signal = "low_confidence"
                self.confidence = 0.3

        gap = MockGapRecord()
        mock_gap_detector.assess_query = AsyncMock(return_value=[gap])

        config = ReconstructionConfig(
            synthesis_threshold=0.8,
            enable_gap_detection=True,
        )
        reconstructor = ReconstructiveRecall(mock_engine, config, mock_gap_detector)

        result = await reconstructor.recall("meeting discussion")

        assert len(result.gap_records) > 0

    @pytest.mark.asyncio
    async def test_fragment_retrieval(self, mock_engine, mock_nodes):
        """Test fragment retrieval from engine."""
        nodes, get_memory_func = mock_nodes
        mock_engine.get_memory = get_memory_func

        reconstructor = ReconstructiveRecall(mock_engine)

        fragments = await reconstructor._retrieve_fragments(
            "meeting discussion",
            top_k=5,
        )

        assert len(fragments) > 0
        assert all(isinstance(f, MemoryFragment) for f in fragments)
        assert fragments[0].similarity >= fragments[-1].similarity  # Sorted

    @pytest.mark.asyncio
    async def test_synthesis_methods(self, mock_engine):
        """Test different synthesis methods."""
        reconstructor = ReconstructiveRecall(mock_engine)

        fragments = [
            MemoryFragment(node_id="f1", content="First fragment", similarity=0.8),
            MemoryFragment(node_id="f2", content="Second fragment", similarity=0.6),
        ]

        # Test extraction (single high-quality fragment)
        method = reconstructor._determine_synthesis_method([fragments[0]])
        assert method == "extraction"

        # Test interpolation (two fragments with first > 0.75 but second significantly lower)
        # Actually, with first at 0.8, extraction is still chosen
        method = reconstructor._determine_synthesis_method(fragments)
        # First fragment is high quality (>0.75), so extraction is used
        assert method == "extraction"

        # Test true synthesis (multiple fragments without a single dominant one)
        many_frags = [
            MemoryFragment(node_id="f1", content="First", similarity=0.6),
            MemoryFragment(node_id="f2", content="Second", similarity=0.5),
            MemoryFragment(node_id="f3", content="Third", similarity=0.45),
            MemoryFragment(node_id="f4", content="Fourth", similarity=0.4),
        ]
        method = reconstructor._determine_synthesis_method(many_frags)
        assert method == "synthesis"

        # Test interpolation (exactly two fragments, both below 0.75)
        two_frags = [
            MemoryFragment(node_id="f1", content="First", similarity=0.6),
            MemoryFragment(node_id="f2", content="Second", similarity=0.5),
        ]
        method = reconstructor._determine_synthesis_method(two_frags)
        assert method == "interpolation"

    def test_confidence_calculation(self, mock_engine):
        """Test confidence score calculation."""
        reconstructor = ReconstructiveRecall(mock_engine)

        fragments = [
            MemoryFragment(node_id="f1", content="Content", similarity=0.8),
            MemoryFragment(node_id="f2", content="More", similarity=0.6),
        ]

        confidence = reconstructor._calculate_reconstruction_confidence(
            fragments,
            "Synthesized content from fragments",
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be decent with these fragments

    def test_confidence_breakdown(self, mock_engine):
        """Test confidence breakdown calculation."""
        reconstructor = ReconstructiveRecall(mock_engine)

        fragments = [
            MemoryFragment(node_id="f1", content="Content", similarity=0.8),
            MemoryFragment(node_id="f2", content="More", similarity=0.6),
        ]

        breakdown = reconstructor._calculate_confidence_breakdown(fragments, None)

        assert "avg_fragment_similarity" in breakdown
        assert "max_fragment_similarity" in breakdown
        assert "fragment_count" in breakdown
        assert breakdown["avg_fragment_similarity"] == 0.7
        assert breakdown["max_fragment_similarity"] == 0.8
        assert breakdown["fragment_count"] == 2

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, mock_engine, mock_nodes):
        """Test that statistics are properly tracked."""
        nodes, get_memory_func = mock_nodes
        mock_engine.get_memory = get_memory_func

        config = ReconstructionConfig(synthesis_threshold=0.8)
        reconstructor = ReconstructiveRecall(mock_engine, config)

        # Perform a few recalls
        await reconstructor.recall("query 1")
        await reconstructor.recall("query 2")
        await reconstructor.recall("query 3")

        stats = reconstructor.stats
        assert stats["total_recalls"] == 3
        assert stats["reconstructed_count"] == 3  # All trigger reconstruction


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_is_reconstructed_memory(self):
        """Test checking if a node is reconstructed."""
        node = MagicMock()
        node.metadata = None

        assert is_reconstructed_memory(node) is False

        node.metadata = {}
        assert is_reconstructed_memory(node) is False

        node.metadata = {"is_reconstructed": True}
        assert is_reconstructed_memory(node) is True

    def test_get_reconstruction_metadata(self):
        """Test extracting reconstruction metadata."""
        node = MagicMock()
        node.metadata = None

        meta = get_reconstruction_metadata(node)
        assert meta == {}

        node.metadata = {
            "is_reconstructed": True,
            "reconstruction_method": "synthesis",
            "query_id": "q123",
            "fragment_count": 3,
        }

        meta = get_reconstruction_metadata(node)
        assert meta["method"] == "synthesis"
        assert meta["query_id"] == "q123"
        assert meta["fragment_count"] == 3


@pytest.mark.asyncio
async def test_full_reconstruction_flow(mock_engine, mock_nodes, mock_gap_detector):
    """Integration test of full reconstruction flow."""
    nodes, get_memory_func = mock_nodes
    mock_engine.get_memory = get_memory_func

    config = ReconstructionConfig(
        synthesis_threshold=0.7,
        enable_gap_detection=True,
        enable_persistent_storage=False,
    )

    reconstructor = ReconstructiveRecall(mock_engine, config, mock_gap_detector)

    result = await reconstructor.recall("What did we decide about the timeline?")

    # Verify result structure
    assert isinstance(result, ReconstructionResult)
    assert result.fragments is not None
    assert result.confidence_breakdown is not None

    # Verify we got some fragments
    assert len(result.fragments) > 0

    # Verify confidence breakdown has expected fields
    assert "avg_fragment_similarity" in result.confidence_breakdown
    assert "overall_confidence" in result.confidence_breakdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
