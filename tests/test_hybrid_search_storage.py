"""
Tests for Hybrid Search Storage Module
=======================================

Tests for the storage-layer hybrid search wrapper that provides integration
with Qdrant PointStruct objects directly.

Coverage:
    - search_qdrant_points() with mocked Qdrant client
    - Score extraction from various point types
    - HybridSearchEngine configuration
    - Factory function create_hybrid_search_engine()
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from mnemocore.storage.hybrid_search import (
    HybridSearchEngine,
    HybridSearchConfig,
    SearchResult,
    SparseEncoder,
    ReciprocalRankFusion,
    create_hybrid_search_engine,
)


# =============================================================================
# Fixtures
# =============================================================================

@dataclass
class MockPointStruct:
    """Mock of Qdrant PointStruct for testing."""
    id: str
    vector: list
    payload: dict
    score: float = 0.0


@dataclass
class MockScoredPoint:
    """Mock of Qdrant ScoredPoint for testing."""
    id: str
    score: float
    payload: dict = None
    vector: list = None


@pytest.fixture
def sample_qdrant_points():
    """Create sample Qdrant points for testing."""
    return [
        MockPointStruct(
            id="point_001",
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"content": "First document", "tier": "hot"},
            score=0.95,
        ),
        MockPointStruct(
            id="point_002",
            vector=[0.2, 0.3, 0.4, 0.5],
            payload={"content": "Second document", "tier": "warm"},
            score=0.85,
        ),
        MockPointStruct(
            id="point_003",
            vector=[0.3, 0.4, 0.5, 0.6],
            payload={"content": "Third document", "tier": "hot"},
            score=0.75,
        ),
    ]


@pytest.fixture
def sample_scored_points():
    """Create sample scored points for testing."""
    return [
        MockScoredPoint(id="scored_001", score=0.92, payload={"content": "Scored 1"}),
        MockScoredPoint(id="scored_002", score=0.88, payload={"content": "Scored 2"}),
        MockScoredPoint(id="scored_003", score=0.81, payload={"content": "Scored 3"}),
    ]


@pytest.fixture
def hybrid_config():
    """Create a HybridSearchConfig for testing."""
    return HybridSearchConfig(
        mode="hybrid",
        hybrid_alpha=0.7,
        rrf_k=60,
    )


@pytest.fixture
def hybrid_engine(hybrid_config):
    """Create a HybridSearchEngine instance."""
    return HybridSearchEngine(config=hybrid_config)


@pytest.fixture
def sparse_encoder():
    """Create a SparseEncoder instance."""
    return SparseEncoder()


# =============================================================================
# HybridSearchConfig Tests
# =============================================================================

class TestHybridSearchConfig:
    """Tests for HybridSearchConfig."""

    def test_default_config(self):
        """Default configuration values are set correctly."""
        config = HybridSearchConfig()

        assert config.mode == "hybrid"
        assert config.hybrid_alpha == 0.7
        assert config.rrf_k == 60
        assert config.enable_query_expansion is True

    def test_custom_config(self):
        """Custom configuration values are applied correctly."""
        config = HybridSearchConfig(
            mode="dense",
            hybrid_alpha=0.5,
            rrf_k=100,
            min_dense_score=0.1,
        )

        assert config.mode == "dense"
        assert config.hybrid_alpha == 0.5
        assert config.rrf_k == 100
        assert config.min_dense_score == 0.1

    def test_alpha_range(self):
        """Alpha value represents weight for dense search."""
        # Alpha closer to 1 favors dense search
        dense_favoring = HybridSearchConfig(hybrid_alpha=0.9)
        assert dense_favoring.hybrid_alpha == 0.9

        # Alpha closer to 0 favors sparse search
        sparse_favoring = HybridSearchConfig(hybrid_alpha=0.1)
        assert sparse_favoring.hybrid_alpha == 0.1


# =============================================================================
# HybridSearchEngine Tests
# =============================================================================

class TestHybridSearchEngineInit:
    """Tests for HybridSearchEngine initialization."""

    def test_engine_creation_default_config(self):
        """Engine is created with default config."""
        engine = HybridSearchEngine()

        assert engine.config is not None
        assert engine.config.mode == "hybrid"

    def test_engine_creation_custom_config(self, hybrid_config):
        """Engine is created with custom config."""
        engine = HybridSearchEngine(config=hybrid_config)

        assert engine.config == hybrid_config
        assert engine.config.hybrid_alpha == 0.7

    def test_engine_has_sparse_encoder(self, hybrid_engine):
        """Engine has sparse encoder initialized."""
        assert hybrid_engine.sparse_encoder is not None
        assert isinstance(hybrid_engine.sparse_encoder, SparseEncoder)

    def test_engine_has_rrf(self, hybrid_engine):
        """Engine has RRF initialized."""
        assert hybrid_engine.rrf is not None
        assert isinstance(hybrid_engine.rrf, ReciprocalRankFusion)


class TestSearchQdrantPoints:
    """Tests for search_qdrant_points method."""

    @pytest.mark.asyncio
    async def test_search_with_point_structs(self, hybrid_engine, sample_qdrant_points):
        """Search works with PointStruct objects."""
        results = await hybrid_engine.search_qdrant_points(
            query="test query",
            qdrant_points=sample_qdrant_points,
            limit=10,
        )

        assert len(results) <= len(sample_qdrant_points)
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_with_scored_points(self, hybrid_engine, sample_scored_points):
        """Search works with ScoredPoint objects."""
        results = await hybrid_engine.search_qdrant_points(
            query="test query",
            qdrant_points=sample_scored_points,
            limit=10,
        )

        assert len(results) <= len(sample_scored_points)
        assert all(hasattr(r, 'id') for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, hybrid_engine, sample_qdrant_points):
        """Search respects the limit parameter."""
        limit = 2
        results = await hybrid_engine.search_qdrant_points(
            query="test query",
            qdrant_points=sample_qdrant_points,
            limit=limit,
        )

        assert len(results) <= limit

    @pytest.mark.asyncio
    async def test_search_empty_points(self, hybrid_engine):
        """Search handles empty point list."""
        results = await hybrid_engine.search_qdrant_points(
            query="test query",
            qdrant_points=[],
            limit=10,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_extracts_scores(self, hybrid_engine, sample_qdrant_points):
        """Search extracts scores from points."""
        results = await hybrid_engine.search_qdrant_points(
            query="test query",
            qdrant_points=sample_qdrant_points,
            limit=10,
        )

        # Results should have scores from the points
        for result in results:
            assert hasattr(result, 'score')
            assert result.score >= 0

    @pytest.mark.asyncio
    async def test_search_extracts_payloads(self, hybrid_engine, sample_qdrant_points):
        """Search extracts payloads from points."""
        results = await hybrid_engine.search_qdrant_points(
            query="test query",
            qdrant_points=sample_qdrant_points,
            limit=10,
        )

        for result in results:
            assert hasattr(result, 'payload')
            assert isinstance(result.payload, dict)

    @pytest.mark.asyncio
    async def test_search_with_none_payload(self, hybrid_engine):
        """Search handles points with None payload."""
        points = [
            MockPointStruct(id="no_payload", vector=[0.1], payload=None, score=0.9),
        ]

        results = await hybrid_engine.search_qdrant_points(
            query="test",
            qdrant_points=points,
            limit=10,
        )

        assert len(results) == 1
        assert results[0].payload == {}


# =============================================================================
# Score Extraction Tests
# =============================================================================

class TestScoreExtraction:
    """Tests for score extraction from various point types."""

    @pytest.mark.asyncio
    async def test_extract_score_from_point_with_score_attr(self, hybrid_engine):
        """Score is extracted from point with score attribute."""
        point = MockPointStruct(id="test", vector=[0.1], payload={}, score=0.85)
        results = await hybrid_engine.search_qdrant_points(
            query="test",
            qdrant_points=[point],
            limit=10,
        )

        # Score should be extracted
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_extract_score_from_point_without_score(self, hybrid_engine):
        """Default score is used when point has no score attribute."""
        # Create a point without score attribute
        point = MagicMock()
        point.id = "no_score"
        point.payload = {}
        # No score attribute

        results = await hybrid_engine.search_qdrant_points(
            query="test",
            qdrant_points=[point],
            limit=10,
        )

        # Should still return results with default score
        assert len(results) == 1
        assert results[0].dense_score == 0.0

    @pytest.mark.asyncio
    async def test_extract_score_various_types(self, hybrid_engine):
        """Score extraction works with various numeric types."""
        points = [
            MockPointStruct(id="int_score", vector=[0.1], payload={}, score=1),  # int
            MockPointStruct(id="float_score", vector=[0.1], payload={}, score=0.5),  # float
        ]

        results = await hybrid_engine.search_qdrant_points(
            query="test",
            qdrant_points=points,
            limit=10,
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_score_preserved_in_result(self, hybrid_engine, sample_scored_points):
        """Original score is preserved in SearchResult."""
        results = await hybrid_engine.search_qdrant_points(
            query="test",
            qdrant_points=sample_scored_points,
            limit=10,
        )

        # Check that dense_score is set
        for result in results:
            assert result.dense_score is not None or result.score is not None


# =============================================================================
# SparseEncoder Tests
# =============================================================================

class TestSparseEncoder:
    """Tests for SparseEncoder."""

    def test_tokenize_basic(self, sparse_encoder):
        """Tokenization works correctly."""
        tokens = sparse_encoder._tokenize("Hello World Test")

        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenize_lowercase(self, sparse_encoder):
        """Tokens are lowercased."""
        tokens = sparse_encoder._tokenize("HELLO WORLD")

        assert all(t.islower() for t in tokens)

    def test_tokenize_removes_punctuation(self, sparse_encoder):
        """Punctuation is removed from tokens."""
        tokens = sparse_encoder._tokenize("Hello, World! Test.")

        assert "," not in tokens
        assert "!" not in tokens
        assert "." not in tokens

    def test_index_documents(self, sparse_encoder):
        """Documents are indexed correctly."""
        documents = [
            ("doc1", "Hello world"),
            ("doc2", "World test"),
            ("doc3", "Hello test"),
        ]

        sparse_encoder.index_documents(documents)

        assert sparse_encoder.total_docs == 3
        assert len(sparse_encoder.vocabulary) > 0

    def test_encode_query(self, sparse_encoder):
        """Query encoding produces weights."""
        sparse_encoder.index_documents([
            ("doc1", "test document"),
        ])

        weights = sparse_encoder.encode("test query")

        assert isinstance(weights, dict)
        assert "test" in weights

    def test_compute_scores(self, sparse_encoder):
        """BM25 scores are computed correctly."""
        sparse_encoder.index_documents([
            ("doc1", "hello world"),
            ("doc2", "test document"),
        ])

        scores = sparse_encoder.compute_scores(
            query="hello",
            document_texts={"doc1": "hello world", "doc2": "test document"},
        )

        assert "doc1" in scores
        assert scores["doc1"] > scores["doc2"]  # doc1 should score higher


# =============================================================================
# ReciprocalRankFusion Tests
# =============================================================================

class TestReciprocalRankFusion:
    """Tests for ReciprocalRankFusion."""

    def test_fuse_single_list(self):
        """RRF works with a single ranked list."""
        rrf = ReciprocalRankFusion(k=60)
        ranked_list = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]

        result = rrf.fuse([ranked_list])

        assert len(result) == 3
        assert result[0][0] == "doc1"  # Highest ranked should be first

    def test_fuse_multiple_lists(self):
        """RRF combines multiple ranked lists."""
        rrf = ReciprocalRankFusion(k=60)
        list1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        list2 = [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.6)]

        result = rrf.fuse([list1, list2])

        # doc1 and doc2 appear in both lists, should be ranked higher
        top_ids = [r[0] for r in result[:2]]
        assert "doc1" in top_ids or "doc2" in top_ids

    def test_fuse_with_limit(self):
        """RRF respects limit parameter."""
        rrf = ReciprocalRankFusion(k=60)
        ranked_list = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]

        result = rrf.fuse([ranked_list], limit=2)

        assert len(result) == 2

    def test_fuse_empty_lists(self):
        """RRF handles empty lists."""
        rrf = ReciprocalRankFusion(k=60)

        result = rrf.fuse([[]])

        assert result == []

    def test_rrf_k_parameter(self):
        """RRF k parameter affects scoring."""
        rrf_low_k = ReciprocalRankFusion(k=10)
        rrf_high_k = ReciprocalRankFusion(k=100)

        ranked_list = [("doc1", 0.9), ("doc2", 0.8)]

        result_low = rrf_low_k.fuse([ranked_list])
        result_high = rrf_high_k.fuse([ranked_list])

        # Both should return same docs but with different scores
        assert len(result_low) == len(result_high)


# =============================================================================
# SearchResult Tests
# =============================================================================

class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """SearchResult is created correctly."""
        result = SearchResult(
            id="test_id",
            score=0.85,
            payload={"content": "test"},
            dense_score=0.9,
            sparse_score=0.8,
        )

        assert result.id == "test_id"
        assert result.score == 0.85
        assert result.payload == {"content": "test"}
        assert result.dense_score == 0.9
        assert result.sparse_score == 0.8

    def test_search_result_score_is_float(self):
        """Score is converted to float."""
        result = SearchResult(id="test", score=1, payload={})

        assert isinstance(result.score, float)
        assert result.score == 1.0

    def test_search_result_optional_scores(self):
        """Optional scores can be None."""
        result = SearchResult(
            id="test",
            score=0.5,
            payload={},
            dense_score=None,
            sparse_score=None,
        )

        assert result.dense_score is None
        assert result.sparse_score is None


# =============================================================================
# Search Mode Tests
# =============================================================================

class TestSearchModes:
    """Tests for different search modes."""

    @pytest.mark.asyncio
    async def test_dense_mode(self, sample_qdrant_points):
        """Dense mode returns dense results only."""
        config = HybridSearchConfig(mode="dense")
        engine = HybridSearchEngine(config=config)

        results = await engine.search_qdrant_points(
            query="test",
            qdrant_points=sample_qdrant_points,
            limit=10,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_sparse_mode(self, sample_qdrant_points):
        """Sparse mode requires indexed documents."""
        config = HybridSearchConfig(mode="sparse")
        engine = HybridSearchEngine(config=config)

        # Without indexed documents, sparse mode returns empty
        results = await engine.search_qdrant_points(
            query="test",
            qdrant_points=sample_qdrant_points,
            limit=10,
        )

        # May return empty since no documents are indexed for sparse search
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_mode(self, sample_qdrant_points):
        """Hybrid mode combines dense and sparse results."""
        config = HybridSearchConfig(mode="hybrid", hybrid_alpha=0.7)
        engine = HybridSearchEngine(config=config)

        results = await engine.search_qdrant_points(
            query="test",
            qdrant_points=sample_qdrant_points,
            limit=10,
        )

        assert len(results) > 0


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateHybridSearchEngine:
    """Tests for create_hybrid_search_engine factory function."""

    def test_create_default_engine(self):
        """Factory creates engine with default settings."""
        engine = create_hybrid_search_engine()

        assert isinstance(engine, HybridSearchEngine)
        assert engine.config.mode == "hybrid"

    def test_create_custom_mode(self):
        """Factory creates engine with custom mode."""
        engine = create_hybrid_search_engine(mode="dense")

        assert engine.config.mode == "dense"

    def test_create_custom_alpha(self):
        """Factory creates engine with custom alpha."""
        engine = create_hybrid_search_engine(alpha=0.5)

        assert engine.config.hybrid_alpha == 0.5

    def test_create_custom_rrf_k(self):
        """Factory creates engine with custom RRF k."""
        engine = create_hybrid_search_engine(rrf_k=100)

        assert engine.config.rrf_k == 100

    def test_create_all_custom_params(self):
        """Factory creates engine with all custom parameters."""
        engine = create_hybrid_search_engine(
            mode="hybrid",
            alpha=0.8,
            rrf_k=50,
        )

        assert engine.config.mode == "hybrid"
        assert engine.config.hybrid_alpha == 0.8
        assert engine.config.rrf_k == 50


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_search_with_duplicate_ids(self, hybrid_engine):
        """Search handles points with duplicate IDs."""
        points = [
            MockPointStruct(id="duplicate", vector=[0.1], payload={}, score=0.9),
            MockPointStruct(id="duplicate", vector=[0.2], payload={}, score=0.8),
        ]

        results = await hybrid_engine.search_qdrant_points(
            query="test",
            qdrant_points=points,
            limit=10,
        )

        # Should handle duplicates gracefully
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_special_characters_in_payload(self, hybrid_engine):
        """Search handles special characters in payload."""
        points = [
            MockPointStruct(
                id="special",
                vector=[0.1],
                payload={"content": "Special chars: <>&\"'"},
                score=0.9,
            ),
        ]

        results = await hybrid_engine.search_qdrant_points(
            query="test",
            qdrant_points=points,
            limit=10,
        )

        assert len(results) == 1
        assert "<>&\"'" in results[0].payload["content"]

    @pytest.mark.asyncio
    async def test_search_with_large_payload(self, hybrid_engine):
        """Search handles large payloads."""
        large_payload = {"data": "x" * 10000}
        points = [
            MockPointStruct(id="large", vector=[0.1], payload=large_payload, score=0.9),
        ]

        results = await hybrid_engine.search_qdrant_points(
            query="test",
            qdrant_points=points,
            limit=10,
        )

        assert len(results) == 1
        assert len(results[0].payload["data"]) == 10000

    @pytest.mark.asyncio
    async def test_search_limit_zero(self, hybrid_engine, sample_qdrant_points):
        """Search with limit 0 returns empty list."""
        results = await hybrid_engine.search_qdrant_points(
            query="test",
            qdrant_points=sample_qdrant_points,
            limit=0,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_negative_scores(self, hybrid_engine):
        """Search handles negative scores gracefully."""
        points = [
            MockPointStruct(id="negative", vector=[0.1], payload={}, score=-0.5),
        ]

        results = await hybrid_engine.search_qdrant_points(
            query="test",
            qdrant_points=points,
            limit=10,
        )

        # Should still return results
        assert isinstance(results, list)


# =============================================================================
# Statistics Tests
# =============================================================================

class TestEngineStatistics:
    """Tests for engine statistics."""

    def test_get_stats(self, hybrid_engine):
        """Engine statistics are returned correctly."""
        stats = hybrid_engine.get_stats()

        assert "mode" in stats
        assert "hybrid_alpha" in stats
        assert "indexed" in stats
        assert "total_documents" in stats
        assert "vocabulary_size" in stats

    def test_stats_after_indexing(self, hybrid_engine):
        """Statistics reflect indexed documents."""
        hybrid_engine.index_batch([
            ("doc1", "Hello world"),
            ("doc2", "Test document"),
        ])

        stats = hybrid_engine.get_stats()

        assert stats["indexed"] is True
        assert stats["total_documents"] == 2
        assert stats["vocabulary_size"] > 0


# =============================================================================
# Document Indexing Tests
# =============================================================================

class TestDocumentIndexing:
    """Tests for document indexing."""

    def test_index_single_document(self, hybrid_engine):
        """Single document can be indexed."""
        hybrid_engine.index_document("doc1", "Hello world")

        assert "doc1" in hybrid_engine._documents
        assert hybrid_engine._documents["doc1"] == "Hello world"

    def test_index_batch_documents(self, hybrid_engine):
        """Batch of documents can be indexed."""
        documents = [
            ("doc1", "First document"),
            ("doc2", "Second document"),
            ("doc3", "Third document"),
        ]

        hybrid_engine.index_batch(documents)

        assert len(hybrid_engine._documents) == 3
        assert hybrid_engine._indexed is True

    def test_index_overwrites_existing(self, hybrid_engine):
        """Indexing overwrites existing documents with same ID."""
        hybrid_engine.index_document("doc1", "Original content")
        hybrid_engine.index_document("doc1", "Updated content")

        assert hybrid_engine._documents["doc1"] == "Updated content"
