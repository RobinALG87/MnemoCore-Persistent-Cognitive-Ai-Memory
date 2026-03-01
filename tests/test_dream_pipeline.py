"""
Comprehensive tests for Dream Pipeline Module.
==============================================
Tests for:
  - EpisodicClusterer (temporal grouping)
  - PatternExtractor (keyword, temporal, metadata patterns)
  - DreamSynthesizer (synthesis from patterns)
  - ContradictionResolver (detection and resolution)
  - SemanticPromoter (promotion scoring)
  - DreamPipeline (full pipeline orchestration)
  - Pipeline with disabled stages
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest

from mnemocore.subconscious.dream.clusterer import EpisodicClusterer, EpisodicCluster
from mnemocore.subconscious.dream.patterns import PatternExtractor
from mnemocore.subconscious.dream.synthesizer import DreamSynthesizer, SynthesisResult
from mnemocore.subconscious.dream.contradictions import (
    ContradictionResolver,
    ContradictionScanResult,
)
from mnemocore.subconscious.dream.promoter import SemanticPromoter, PromotionResult
from mnemocore.subconscious.dream.pipeline import (
    DreamPipeline,
    DreamPipelineConfig,
    DreamPipelineResult,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_haim_engine():
    """Create a mock HAIM engine for testing."""
    engine = MagicMock()
    engine.dimension = 10000

    # Mock tier manager
    tier_manager = MagicMock()
    tier_manager.hot = {}
    tier_manager.warm = {}
    tier_manager.get_memory = MagicMock(return_value=None)
    tier_manager.promote_to_warm = MagicMock()
    tier_manager.get_all_hot = AsyncMock(return_value=[])
    engine.tier_manager = tier_manager

    # Mock store
    engine.store = MagicMock(return_value="test_node_id")

    return engine


@pytest.fixture
def mock_memory_nodes():
    """Create a list of mock memory nodes for testing."""
    nodes = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=10)

    for i in range(10):
        node = MagicMock()
        node.id = f"node_{i}"
        node.content = f"Memory content about topic {i % 3} with keywords like python and testing"
        node.metadata = {
            "category": f"cat_{i % 2}",
            "tags": ["tag1", "tag2"] if i % 2 else ["tag3"],
            "importance": "high" if i < 5 else "low",
        }
        node.created_at = base_time + timedelta(hours=i)
        node.access_count = i + 1
        node.ltp_strength = 0.5 + (i * 0.05)
        node.tier = "hot"
        node.hdv = MagicMock()
        node.hdv.similarity = MagicMock(return_value=0.8)

        nodes.append(node)

    return nodes


@pytest.fixture
def dream_pipeline_config():
    """Create a standard dream pipeline configuration."""
    return DreamPipelineConfig(
        enable_episodic_clustering=True,
        cluster_time_window_hours=24.0,
        min_cluster_size=3,
        enable_pattern_extraction=True,
        pattern_min_frequency=2,
        enable_recursive_synthesis=False,  # Disable for simpler tests
        enable_contradiction_resolution=True,
        enable_semantic_promotion=True,
        enable_dream_report=True,
    )


# =====================================================================
# EpisodicClusterer Tests
# =====================================================================

class TestEpisodicClusterer:
    """Tests for the EpisodicClusterer component."""

    def test_init_defaults(self):
        """Test default initialization."""
        clusterer = EpisodicClusterer()
        assert clusterer.time_window.total_seconds() == 24 * 3600  # 24 hours
        assert clusterer.min_cluster_size == 3

    def test_init_custom_params(self):
        """Test custom initialization."""
        clusterer = EpisodicClusterer(
            time_window_hours=12.0,
            min_cluster_size=5
        )
        assert clusterer.time_window.total_seconds() == 12 * 3600
        assert clusterer.min_cluster_size == 5

    @pytest.mark.asyncio
    async def test_cluster_empty_memories(self):
        """Test clustering with empty memory list."""
        clusterer = EpisodicClusterer()
        result = await clusterer.cluster([])
        assert result == []

    @pytest.mark.asyncio
    async def test_cluster_single_memory(self):
        """Test clustering with single memory (no cluster formed)."""
        clusterer = EpisodicClusterer(min_cluster_size=2)
        node = MagicMock()
        node.created_at = datetime.now(timezone.utc)
        node.content = "Single memory"
        node.metadata = {}
        node.ltp_strength = 0.5

        result = await clusterer.cluster([node])
        # Should not form a cluster with just one memory
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_cluster_with_mocked_engine(self, mock_memory_nodes):
        """Test clustering with mocked engine returns proper clusters."""
        clusterer = EpisodicClusterer(
            time_window_hours=48.0,
            min_cluster_size=3
        )

        result = await clusterer.cluster(mock_memory_nodes)

        # Should create at least one cluster with 10 memories within 48h
        assert isinstance(result, list)
        for cluster in result:
            assert "cluster_id" in cluster
            assert "memory_count" in cluster
            assert "memory_ids" in cluster
            assert cluster["memory_count"] >= 3

    @pytest.mark.asyncio
    async def test_cluster_respects_time_window(self):
        """Test that clustering respects time window."""
        clusterer = EpisodicClusterer(
            time_window_hours=1.0,  # 1 hour window
            min_cluster_size=2
        )

        # Create memories far apart
        nodes = []
        for i in range(4):
            node = MagicMock()
            node.created_at = datetime.now(timezone.utc) - timedelta(hours=i * 3)
            node.content = f"Memory {i}"
            node.metadata = {}
            node.ltp_strength = 0.5
            nodes.append(node)

        result = await clusterer.cluster(nodes)

        # With 1-hour window and 3-hour gaps, should not form clusters
        assert len(result) == 0

    def test_create_cluster(self):
        """Test _create_cluster helper method."""
        clusterer = EpisodicClusterer()

        nodes = []
        base_time = datetime.now(timezone.utc)
        for i in range(3):
            node = MagicMock()
            node.id = f"node_{i}"
            node.created_at = base_time + timedelta(minutes=i * 10)
            node.content = f"Content {i}"
            node.metadata = {"category": "test"}
            node.ltp_strength = 0.5 + i * 0.1
            nodes.append(node)

        cluster = clusterer._create_cluster(nodes)

        assert "cluster_id" in cluster
        assert cluster["memory_count"] == 3
        assert len(cluster["memory_ids"]) == 3
        assert "test" in cluster["categories"]


# =====================================================================
# PatternExtractor Tests
# =====================================================================

class TestPatternExtractor:
    """Tests for the PatternExtractor component."""

    def test_init_defaults(self):
        """Test default initialization."""
        extractor = PatternExtractor()
        assert extractor.min_frequency == 2
        assert extractor.similarity_threshold == 0.75
        assert len(extractor._stopwords) > 0

    def test_init_custom_params(self):
        """Test custom initialization."""
        extractor = PatternExtractor(
            min_frequency=3,
            similarity_threshold=0.85
        )
        assert extractor.min_frequency == 3
        assert extractor.similarity_threshold == 0.85

    @pytest.mark.asyncio
    async def test_extract_empty_memories(self):
        """Test pattern extraction with empty memory list."""
        extractor = PatternExtractor()
        result = await extractor.extract([])
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_keyword_patterns(self, mock_memory_nodes):
        """Test keyword pattern extraction."""
        extractor = PatternExtractor(min_frequency=2)

        result = await extractor.extract(mock_memory_nodes)

        # Should find patterns
        keyword_patterns = [p for p in result if p["pattern_type"] == "keyword"]
        # May or may not find patterns depending on content
        for pattern in keyword_patterns:
            assert "pattern_value" in pattern
            assert "frequency" in pattern
            assert pattern["frequency"] >= 2

    @pytest.mark.asyncio
    async def test_extract_temporal_patterns(self, mock_memory_nodes):
        """Test temporal pattern extraction."""
        extractor = PatternExtractor(min_frequency=2)

        result = await extractor.extract(mock_memory_nodes)

        temporal_patterns = [p for p in result if p["pattern_type"].startswith("temporal")]
        # Temporal patterns are extracted based on creation times
        for pattern in temporal_patterns:
            assert "pattern_value" in pattern
            assert "frequency" in pattern

    @pytest.mark.asyncio
    async def test_extract_metadata_patterns(self, mock_memory_nodes):
        """Test metadata pattern extraction."""
        extractor = PatternExtractor(min_frequency=2)

        result = await extractor.extract(mock_memory_nodes)

        metadata_patterns = [
            p for p in result
            if p["pattern_type"] in ("category", "tag")
        ]
        # Should find category patterns (cat_0, cat_1)
        for pattern in metadata_patterns:
            assert "pattern_value" in pattern
            assert "frequency" in pattern

    @pytest.mark.asyncio
    async def test_extract_with_llm_client(self, mock_memory_nodes):
        """Test pattern extraction with LLM client for semantic patterns."""
        extractor = PatternExtractor(min_frequency=2)

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value='[{"theme": "testing", "description": "Test theme", "evidence_count": 5}]'
        )

        result = await extractor.extract(mock_memory_nodes, llm_client=mock_llm)

        # Should include semantic patterns from LLM
        semantic_patterns = [p for p in result if p["pattern_type"] == "semantic_theme"]
        # LLM might or might not be called depending on memory count
        assert isinstance(result, list)

    def test_tokenize(self):
        """Test tokenization helper."""
        extractor = PatternExtractor()

        tokens = extractor._tokenize("Hello World! This is a Test123.")

        assert "hello" in tokens
        assert "world" in tokens
        assert "test123" in tokens
        assert "is" not in tokens  # Too short


# =====================================================================
# DreamSynthesizer Tests
# =====================================================================

class TestDreamSynthesizer:
    """Tests for the DreamSynthesizer component."""

    def test_init(self, mock_haim_engine):
        """Test initialization."""
        synthesizer = DreamSynthesizer(
            engine=mock_haim_engine,
            max_depth=3,
            max_patterns=10
        )

        assert synthesizer.engine == mock_haim_engine
        assert synthesizer.max_depth == 3
        assert synthesizer.max_patterns == 10

    @pytest.mark.asyncio
    async def test_synthesize_patterns_empty(self, mock_haim_engine):
        """Test synthesis with empty patterns."""
        synthesizer = DreamSynthesizer(engine=mock_haim_engine)
        result = await synthesizer.synthesize_patterns([])
        assert result == []

    @pytest.mark.asyncio
    async def test_synthesize_patterns_with_mocked_synthesizer(
        self, mock_haim_engine
    ):
        """Test synthesis with mocked recursive synthesizer."""
        synthesizer = DreamSynthesizer(engine=mock_haim_engine)

        # Mock the internal synthesizer
        mock_synth = MagicMock()
        mock_result = MagicMock()
        mock_result.results = ["result1", "result2"]
        mock_result.synthesis = "Test synthesis output"
        mock_synth.synthesize = AsyncMock(return_value=mock_result)

        synthesizer._synthesizer = mock_synth

        patterns = [
            {"pattern_type": "keyword", "pattern_value": "python"}
        ]

        result = await synthesizer.synthesize_patterns(patterns)

        # Should return list of synthesis results
        assert isinstance(result, list)

    def test_pattern_to_query_keyword(self, mock_haim_engine):
        """Test pattern to query conversion for keyword."""
        synthesizer = DreamSynthesizer(engine=mock_haim_engine)

        pattern = {"pattern_type": "keyword", "pattern_value": "python"}
        query = synthesizer._pattern_to_query(pattern)

        assert "python" in query

    def test_pattern_to_query_category(self, mock_haim_engine):
        """Test pattern to query conversion for category."""
        synthesizer = DreamSynthesizer(engine=mock_haim_engine)

        pattern = {"pattern_type": "category", "pattern_value": "programming"}
        query = synthesizer._pattern_to_query(pattern)

        assert "programming" in query

    def test_pattern_to_query_semantic_theme(self, mock_haim_engine):
        """Test pattern to query conversion for semantic theme."""
        synthesizer = DreamSynthesizer(engine=mock_haim_engine)

        pattern = {
            "pattern_type": "semantic_theme",
            "pattern_value": "AI Development",
            "description": "Topics related to AI"
        }
        query = synthesizer._pattern_to_query(pattern)

        assert "AI Development" in query


# =====================================================================
# ContradictionResolver Tests
# =====================================================================

class TestContradictionResolver:
    """Tests for the ContradictionResolver component."""

    def test_init(self, mock_haim_engine):
        """Test initialization."""
        resolver = ContradictionResolver(
            engine=mock_haim_engine,
            similarity_threshold=0.80,
            auto_resolve=False
        )

        assert resolver.engine == mock_haim_engine
        assert resolver.similarity_threshold == 0.80
        assert resolver.auto_resolve is False

    @pytest.mark.asyncio
    async def test_scan_and_resolve_empty(self, mock_haim_engine):
        """Test scan with empty memory list."""
        resolver = ContradictionResolver(engine=mock_haim_engine)
        result = await resolver.scan_and_resolve([])

        assert result["contradictions_found"] == 0
        assert result["contradictions_resolved"] == 0

    @pytest.mark.asyncio
    async def test_scan_and_resolve_with_mocked_detector(
        self, mock_haim_engine, mock_memory_nodes
    ):
        """Test scan with mocked contradiction detector."""
        resolver = ContradictionResolver(
            engine=mock_haim_engine,
            auto_resolve=False
        )

        # Mock the detector
        mock_detector = MagicMock()
        mock_record = MagicMock()
        mock_record.resolved = False
        mock_record.similarity_score = 0.85
        mock_record.llm_confirmed = False
        mock_record.group_id = "contradiction_1"
        mock_detector.scan = AsyncMock(return_value=[mock_record])

        resolver._detector = mock_detector

        result = await resolver.scan_and_resolve(mock_memory_nodes)

        assert "contradictions_found" in result
        assert "contradictions_resolved" in result

    def test_is_simple_contradiction(self, mock_haim_engine):
        """Test simple contradiction detection."""
        resolver = ContradictionResolver(engine=mock_haim_engine)

        # High similarity, not LLM confirmed = simple
        mock_record = MagicMock()
        mock_record.similarity_score = 0.97
        mock_record.llm_confirmed = False

        assert resolver._is_simple_contradiction(mock_record) is True

        # Lower similarity = not simple
        mock_record.similarity_score = 0.85
        assert resolver._is_simple_contradiction(mock_record) is False


# =====================================================================
# SemanticPromoter Tests
# =====================================================================

class TestSemanticPromoter:
    """Tests for the SemanticPromoter component."""

    def test_init(self, mock_haim_engine):
        """Test initialization."""
        promoter = SemanticPromoter(
            engine=mock_haim_engine,
            ltp_threshold=0.7,
            access_threshold=5,
            auto_promote=True
        )

        assert promoter.engine == mock_haim_engine
        assert promoter.ltp_threshold == 0.7
        assert promoter.access_threshold == 5
        assert promoter.auto_promote is True

    @pytest.mark.asyncio
    async def test_promote_empty(self, mock_haim_engine):
        """Test promotion with empty memory list."""
        promoter = SemanticPromoter(engine=mock_haim_engine)
        result = await promoter.promote([])

        assert result["candidates_count"] == 0
        assert result["promoted_count"] == 0

    @pytest.mark.asyncio
    async def test_promote_hot_memories(self, mock_haim_engine):
        """Test promotion of HOT tier memories."""
        promoter = SemanticPromoter(
            engine=mock_haim_engine,
            ltp_threshold=0.6,
            access_threshold=3,
            auto_promote=False  # Just identify candidates
        )

        # Create nodes that should be promoted
        nodes = []
        for i in range(5):
            node = MagicMock()
            node.id = f"node_{i}"
            node.tier = "hot"
            node.ltp_strength = 0.7 + i * 0.05
            node.access_count = 10 + i
            node.metadata = {}
            nodes.append(node)

        result = await promoter.promote(nodes)

        # Should identify candidates
        assert result["candidates_count"] > 0

    def test_should_promote_high_ltp(self, mock_haim_engine):
        """Test promotion decision based on LTP strength."""
        promoter = SemanticPromoter(
            engine=mock_haim_engine,
            ltp_threshold=0.7
        )

        node = MagicMock()
        node.ltp_strength = 0.85
        node.access_count = 1
        node.metadata = {}

        assert promoter._should_promote(node) is True

    def test_should_promote_high_access(self, mock_haim_engine):
        """Test promotion decision based on access count."""
        promoter = SemanticPromoter(
            engine=mock_haim_engine,
            access_threshold=5
        )

        node = MagicMock()
        node.ltp_strength = 0.3
        node.access_count = 10
        node.metadata = {}

        assert promoter._should_promote(node) is True

    def test_should_promote_important_metadata(self, mock_haim_engine):
        """Test promotion decision based on importance metadata."""
        promoter = SemanticPromoter(engine=mock_haim_engine)

        node = MagicMock()
        node.ltp_strength = 0.3
        node.access_count = 1
        node.metadata = {"important": True}

        assert promoter._should_promote(node) is True

    def test_should_not_promote(self, mock_haim_engine):
        """Test when memory should not be promoted."""
        promoter = SemanticPromoter(
            engine=mock_haim_engine,
            ltp_threshold=0.7,
            access_threshold=10
        )

        node = MagicMock()
        node.ltp_strength = 0.5
        node.access_count = 3
        node.metadata = {}

        assert promoter._should_promote(node) is False


# =====================================================================
# DreamPipeline Tests
# =====================================================================

class TestDreamPipeline:
    """Tests for the main DreamPipeline orchestrator."""

    def test_init(self, mock_haim_engine):
        """Test pipeline initialization."""
        config = DreamPipelineConfig()
        pipeline = DreamPipeline(mock_haim_engine, config)

        assert pipeline.engine == mock_haim_engine
        assert pipeline.cfg == config
        assert pipeline.episodic_clusterer is not None
        assert pipeline.pattern_extractor is not None
        assert pipeline.dream_synthesizer is not None
        assert pipeline.contradiction_resolver is not None
        assert pipeline.semantic_promoter is not None

    @pytest.mark.asyncio
    async def test_run_empty_memories(self, mock_haim_engine):
        """Test pipeline run with no memories."""
        config = DreamPipelineConfig()
        pipeline = DreamPipeline(mock_haim_engine, config)

        # Mock empty memories
        mock_haim_engine.tier_manager.get_all_hot = AsyncMock(return_value=[])

        result = await pipeline.run()

        assert result["success"] is True
        assert result["memories_processed"] == 0

    @pytest.mark.asyncio
    async def test_run_full_pipeline(
        self, mock_haim_engine, mock_memory_nodes, dream_pipeline_config
    ):
        """Test full pipeline execution."""
        pipeline = DreamPipeline(mock_haim_engine, dream_pipeline_config)

        # Mock memories
        mock_haim_engine.tier_manager.get_all_hot = AsyncMock(
            return_value=mock_memory_nodes
        )

        # Mock tier manager promotion
        mock_haim_engine.tier_manager.promote_to_warm = MagicMock()

        result = await pipeline.run()

        assert result["success"] is True
        assert "duration_seconds" in result
        assert "memories_processed" in result
        assert "patterns_extracted" in result
        assert "dream_report" in result

    @pytest.mark.asyncio
    async def test_run_with_disabled_stages(self, mock_haim_engine, mock_memory_nodes):
        """Test pipeline with some stages disabled."""
        config = DreamPipelineConfig(
            enable_episodic_clustering=True,
            enable_pattern_extraction=False,
            enable_recursive_synthesis=False,
            enable_contradiction_resolution=False,
            enable_semantic_promotion=False,
            enable_dream_report=True,
        )
        pipeline = DreamPipeline(mock_haim_engine, config)

        mock_haim_engine.tier_manager.get_all_hot = AsyncMock(
            return_value=mock_memory_nodes
        )

        result = await pipeline.run()

        assert result["success"] is True
        # Patterns should be empty since extraction is disabled
        assert len(result.get("patterns_extracted", [])) == 0

    @pytest.mark.asyncio
    async def test_run_all_stages_disabled(self, mock_haim_engine, mock_memory_nodes):
        """Test pipeline with all processing stages disabled."""
        config = DreamPipelineConfig(
            enable_episodic_clustering=False,
            enable_pattern_extraction=False,
            enable_recursive_synthesis=False,
            enable_contradiction_resolution=False,
            enable_semantic_promotion=False,
            enable_dream_report=False,
        )
        pipeline = DreamPipeline(mock_haim_engine, config)

        mock_haim_engine.tier_manager.get_all_hot = AsyncMock(
            return_value=mock_memory_nodes
        )

        result = await pipeline.run()

        assert result["success"] is True
        assert result["episodic_clusters_count"] == 0
        assert result["patterns_extracted_count"] == 0

    @pytest.mark.asyncio
    async def test_run_handles_errors(self, mock_haim_engine):
        """Test pipeline handles errors gracefully."""
        config = DreamPipelineConfig()
        pipeline = DreamPipeline(mock_haim_engine, config)

        # Mock an error in fetching memories
        mock_haim_engine.tier_manager.get_all_hot = AsyncMock(
            side_effect=Exception("Test error")
        )

        result = await pipeline.run()

        assert result["success"] is False
        assert "error" in result

    def test_get_enabled_stages(self, mock_haim_engine):
        """Test getting list of enabled stages."""
        config = DreamPipelineConfig(
            enable_episodic_clustering=True,
            enable_pattern_extraction=True,
            enable_recursive_synthesis=False,
            enable_contradiction_resolution=True,
            enable_semantic_promotion=True,
            enable_dream_report=True,
        )
        pipeline = DreamPipeline(mock_haim_engine, config)

        stages = pipeline._get_enabled_stages()

        assert "episodic_clustering" in stages
        assert "pattern_extraction" in stages
        assert "contradiction_resolution" in stages
        assert "semantic_promotion" in stages
        assert "dream_report" in stages
        assert "recursive_synthesis" not in stages


# =====================================================================
# DreamPipelineResult Tests
# =====================================================================

class TestDreamPipelineResult:
    """Tests for DreamPipelineResult dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = DreamPipelineResult(
            success=True,
            duration_seconds=1.5,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            episodic_clusters=[{"cluster_id": "c1"}],
            patterns_extracted=[{"pattern_type": "keyword"}],
            memories_processed=10,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["duration_seconds"] == 1.5
        assert d["episodic_clusters_count"] == 1
        assert d["patterns_extracted_count"] == 1
        assert d["memories_processed"] == 10


# =====================================================================
# Data Class Tests
# =====================================================================

class TestDreamDataClasses:
    """Tests for dream pipeline data classes."""

    def test_episodic_cluster_to_dict(self):
        """Test EpisodicCluster serialization."""
        cluster = EpisodicCluster(
            cluster_id="cluster_1",
            memory_count=5,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(hours=1),
            duration_hours=1.0,
            memory_ids=["n1", "n2", "n3", "n4", "n5"],
            categories=["test"],
            avg_ltp=0.75,
        )

        d = cluster.to_dict()

        assert d["cluster_id"] == "cluster_1"
        assert d["memory_count"] == 5
        assert d["duration_hours"] == 1.0

    def test_contradiction_scan_result_to_dict(self):
        """Test ContradictionScanResult serialization."""
        result = ContradictionScanResult(
            contradictions_found=3,
            contradictions_resolved=2,
            resolved_ids=["c1", "c2"],
            unresolved_ids=["c3"],
        )

        d = result.to_dict()

        assert d["contradictions_found"] == 3
        assert d["contradictions_resolved"] == 2

    def test_promotion_result_to_dict(self):
        """Test PromotionResult serialization."""
        result = PromotionResult(
            candidates_count=10,
            promoted_count=5,
            promoted_ids=["n1", "n2", "n3", "n4", "n5"],
        )

        d = result.to_dict()

        assert d["candidates_count"] == 10
        assert d["promoted_count"] == 5
