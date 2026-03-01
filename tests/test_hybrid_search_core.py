"""
Comprehensive Tests for Hybrid Search Engine
=============================================

Tests the hybrid (dense + sparse) search engine with BM25 and RRF.

Coverage:
- SparseEncoder.index_documents() and search()
- Combined vector+keyword search result merging
- expand_query() (even if stub - test it returns unchanged)
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from mnemocore.core.hybrid_search import (
    SparseEncoder,
    ReciprocalRankFusion,
    HybridSearchEngine,
    HybridSearchConfig,
    SearchResult,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def hybrid_config():
    """Create a HybridSearchConfig for testing."""
    return HybridSearchConfig(
        mode="hybrid",
        hybrid_alpha=0.7,
        rrf_k=60,
        enable_query_expansion=True,
        expansion_terms=3,
    )


@pytest.fixture
def sparse_encoder():
    """Create a SparseEncoder instance."""
    return SparseEncoder(k1=1.5, b=0.75)


@pytest.fixture
def sample_documents():
    """Sample documents for indexing."""
    return [
        ("doc1", "The quick brown fox jumps over the lazy dog"),
        ("doc2", "Machine learning is a subset of artificial intelligence"),
        ("doc3", "Python is a popular programming language"),
        ("doc4", "Deep learning uses neural networks for complex tasks"),
        ("doc5", "Natural language processing enables text understanding"),
    ]


@pytest.fixture
def hybrid_engine(hybrid_config):
    """Create a HybridSearchEngine instance."""
    return HybridSearchEngine(config=hybrid_config)


# =============================================================================
# HybridSearchConfig Tests
# =============================================================================

class TestHybridSearchConfig:
    """Test configuration."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = HybridSearchConfig()

        assert config.mode == "hybrid"
        assert config.hybrid_alpha == 0.7
        assert config.rrf_k == 60
        assert config.enable_query_expansion is True

    def test_custom_config(self):
        """Custom config values should be set correctly."""
        config = HybridSearchConfig(
            mode="dense",
            hybrid_alpha=0.5,
            rrf_k=100,
        )

        assert config.mode == "dense"
        assert config.hybrid_alpha == 0.5
        assert config.rrf_k == 100


# =============================================================================
# SparseEncoder Tests
# =============================================================================

class TestSparseEncoder:
    """Test BM25 sparse encoder."""

    def test_init(self, sparse_encoder):
        """Should initialize with correct parameters."""
        assert sparse_encoder.k1 == 1.5
        assert sparse_encoder.b == 0.75
        assert sparse_encoder.total_docs == 0

    def test_index_documents(self, sparse_encoder, sample_documents):
        """Should index documents correctly."""
        sparse_encoder.index_documents(sample_documents)

        assert sparse_encoder.total_docs == 5
        assert len(sparse_encoder.vocabulary) > 0
        assert sparse_encoder.avg_doc_length > 0

    def test_index_documents_updates_doc_freqs(self, sparse_encoder, sample_documents):
        """Should update document frequencies."""
        sparse_encoder.index_documents(sample_documents)

        # "the" appears in doc1
        assert "the" in sparse_encoder.vocabulary
        # "learning" appears in doc2 and doc4
        assert sparse_encoder.doc_freqs["learning"] == 2

    def test_tokenize(self, sparse_encoder):
        """Should tokenize text correctly."""
        tokens = sparse_encoder._tokenize("Hello, World! This is a test.")

        assert "hello" in tokens
        assert "world" in tokens
        assert "this" in tokens
        assert "is" in tokens
        assert "a" in tokens
        assert "test" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_encode(self, sparse_encoder, sample_documents):
        """Should encode query to sparse vector."""
        sparse_encoder.index_documents(sample_documents)

        weights = sparse_encoder.encode("machine learning algorithms")

        assert isinstance(weights, dict)
        assert "machine" in weights
        assert "learning" in weights
        # Weights should be positive
        assert all(w >= 0 for w in weights.values())

    def test_compute_scores(self, sparse_encoder, sample_documents):
        """Should compute BM25 scores for documents."""
        sparse_encoder.index_documents(sample_documents)

        doc_texts = {doc_id: text for doc_id, text in sample_documents}
        scores = sparse_encoder.compute_scores("machine learning", doc_texts)

        # doc2 should have highest score (contains both terms)
        assert scores["doc2"] > 0
        # doc4 also contains "learning"
        assert scores["doc4"] > 0
        # doc1 doesn't contain either term
        assert scores["doc1"] == 0

    def test_compute_scores_empty_query(self, sparse_encoder, sample_documents):
        """Should handle empty query."""
        sparse_encoder.index_documents(sample_documents)

        doc_texts = {doc_id: text for doc_id, text in sample_documents}
        scores = sparse_encoder.compute_scores("", doc_texts)

        # All scores should be 0
        assert all(s == 0 for s in scores.values())


# =============================================================================
# ReciprocalRankFusion Tests
# =============================================================================

class TestReciprocalRankFusion:
    """Test RRF fusion."""

    def test_init(self):
        """Should initialize with k parameter."""
        rrf = ReciprocalRankFusion(k=60)

        assert rrf.k == 60

    def test_fuse_single_list(self):
        """Should handle single ranked list."""
        rrf = ReciprocalRankFusion(k=60)
        ranked_list = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]

        result = rrf.fuse([ranked_list])

        assert len(result) == 3
        # Order should be preserved
        assert result[0][0] == "doc1"

    def test_fuse_multiple_lists(self):
        """Should fuse multiple ranked lists."""
        rrf = ReciprocalRankFusion(k=60)
        list1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        list2 = [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.75)]

        result = rrf.fuse([list1, list2])

        # doc1 and doc2 appear in both lists, should rank higher
        top_ids = [r[0] for r in result[:2]]
        assert "doc1" in top_ids or "doc2" in top_ids

    def test_fuse_respects_limit(self):
        """Should respect limit parameter."""
        rrf = ReciprocalRankFusion(k=60)
        ranked_list = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]

        result = rrf.fuse([ranked_list], limit=2)

        assert len(result) == 2

    def test_fuse_empty_lists(self):
        """Should handle empty input."""
        rrf = ReciprocalRankFusion(k=60)

        result = rrf.fuse([])

        assert result == []

    def test_fuse_formula(self):
        """RRF formula should be: sum(1 / (k + rank + 1))."""
        rrf = ReciprocalRankFusion(k=60)
        list1 = [("a", 1.0), ("b", 0.5)]
        list2 = [("b", 1.0), ("c", 0.5)]

        result = rrf.fuse([list1, list2])
        result_dict = dict(result)

        # b appears at rank 1 in list1 and rank 0 in list2
        # score = 1/(60+1) + 1/(60+0) = 1/61 + 1/60
        expected_b_score = 1 / 61 + 1 / 60
        assert abs(result_dict["b"] - expected_b_score) < 0.001


# =============================================================================
# HybridSearchEngine Tests
# =============================================================================

class TestHybridSearchEngine:
    """Test hybrid search engine."""

    def test_init(self, hybrid_config):
        """Should initialize correctly."""
        engine = HybridSearchEngine(config=hybrid_config)

        assert engine.config == hybrid_config
        assert engine.sparse_encoder is not None
        assert engine.rrf is not None
        assert engine._indexed is False

    def test_index_document(self, hybrid_engine):
        """Should index single document."""
        hybrid_engine.index_document("doc1", "Test document content")

        assert "doc1" in hybrid_engine._documents
        assert hybrid_engine._documents["doc1"] == "Test document content"

    def test_index_batch(self, hybrid_engine, sample_documents):
        """Should index batch of documents."""
        hybrid_engine.index_batch(sample_documents)

        assert len(hybrid_engine._documents) == 5
        assert hybrid_engine._indexed is True

    @pytest.mark.asyncio
    async def test_search_dense_only_mode(self, hybrid_engine):
        """Should return only dense results in dense mode."""
        hybrid_engine.config.mode = "dense"
        dense_results = [("doc1", 0.9), ("doc2", 0.8)]

        results = await hybrid_engine.search(
            query="test query",
            dense_results=dense_results,
            limit=10,
        )

        assert len(results) == 2
        assert all(r.dense_score is not None for r in results)
        assert all(r.sparse_score is None for r in results)

    @pytest.mark.asyncio
    async def test_search_sparse_only_mode(self, hybrid_engine, sample_documents):
        """Should return only sparse results in sparse mode."""
        hybrid_engine.config.mode = "sparse"
        hybrid_engine.index_batch(sample_documents)

        results = await hybrid_engine.search(
            query="machine learning",
            dense_results=[],
            limit=10,
        )

        assert len(results) >= 1
        assert all(r.sparse_score is not None for r in results)

    @pytest.mark.asyncio
    async def test_search_sparse_without_index(self, hybrid_engine):
        """Should return empty if sparse mode but not indexed."""
        hybrid_engine.config.mode = "sparse"

        results = await hybrid_engine.search(
            query="test query",
            dense_results=[],
            limit=10,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_hybrid_mode(self, hybrid_engine, sample_documents):
        """Should combine dense and sparse in hybrid mode."""
        hybrid_engine.index_batch(sample_documents)

        # Dense results favor doc2
        dense_results = [("doc2", 0.95), ("doc4", 0.8)]

        results = await hybrid_engine.search(
            query="machine learning",
            dense_results=dense_results,
            limit=10,
        )

        # Should have results
        assert len(results) >= 1
        # Results should have both scores
        for r in results:
            assert r.dense_score is not None or r.sparse_score is not None

    @pytest.mark.asyncio
    async def test_search_hybrid_empty_dense(self, hybrid_engine, sample_documents):
        """Should fall back to sparse if no dense results."""
        hybrid_engine.index_batch(sample_documents)

        results = await hybrid_engine.search(
            query="machine learning",
            dense_results=[],
            limit=10,
        )

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_hybrid_empty_sparse(self, hybrid_engine):
        """Should fall back to dense if not indexed."""
        dense_results = [("doc1", 0.9), ("doc2", 0.8)]

        results = await hybrid_engine.search(
            query="test query",
            dense_results=dense_results,
            limit=10,
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_respects_min_scores(self, sample_documents):
        """Should filter results below minimum scores."""
        config = HybridSearchConfig(
            mode="dense",
            min_dense_score=0.5,
        )
        engine = HybridSearchEngine(config=config)

        dense_results = [("doc1", 0.9), ("doc2", 0.3), ("doc3", 0.6)]

        results = await engine.search(
            query="test",
            dense_results=dense_results,
            limit=10,
        )

        # Should filter out doc2 (0.3 < 0.5)
        result_ids = [r.id for r in results]
        assert "doc2" not in result_ids


# =============================================================================
# RRF Search Tests
# =============================================================================

class TestSearchRRF:
    """Test RRF-based hybrid search."""

    @pytest.mark.asyncio
    async def test_search_rrf_combines_ranks(self, hybrid_engine, sample_documents):
        """RRF should combine based on ranks, not scores."""
        hybrid_engine.index_batch(sample_documents)

        dense_results = [("doc2", 0.95), ("doc4", 0.8)]
        results = await hybrid_engine.search_rrf(
            query="machine learning",
            dense_results=dense_results,
            limit=10,
        )

        assert len(results) >= 1
        # All results should have RRF score
        for r in results:
            assert r.score > 0

    @pytest.mark.asyncio
    async def test_search_rrf_empty_inputs(self, hybrid_engine):
        """RRF should handle empty inputs."""
        results = await hybrid_engine.search_rrf(
            query="test",
            dense_results=[],
            limit=10,
        )

        assert results == []


# =============================================================================
# Query Expansion Tests
# =============================================================================

class TestExpandQuery:
    """Test query expansion."""

    def test_expand_query_disabled(self, sample_documents):
        """Should return unchanged if expansion disabled."""
        config = HybridSearchConfig(enable_query_expansion=False)
        engine = HybridSearchEngine(config=config)
        engine.index_batch(sample_documents)

        result = engine.expand_query("machine learning")

        assert result == "machine learning"

    def test_expand_query_not_indexed(self, hybrid_engine):
        """Should return unchanged if not indexed."""
        result = hybrid_engine.expand_query("machine learning")

        assert result == "machine learning"

    def test_expand_query_returns_unchanged(self, hybrid_engine, sample_documents):
        """expand_query is currently a stub - returns unchanged."""
        hybrid_engine.index_batch(sample_documents)

        result = hybrid_engine.expand_query("machine learning")

        # Current implementation returns unchanged
        assert result == "machine learning"


# =============================================================================
# SearchResult Tests
# =============================================================================

class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Should create SearchResult correctly."""
        result = SearchResult(
            id="doc1",
            score=0.95,
            payload={"text": "Sample text"},
            dense_score=0.9,
            sparse_score=0.5,
        )

        assert result.id == "doc1"
        assert result.score == 0.95
        assert result.dense_score == 0.9
        assert result.sparse_score == 0.5

    def test_search_result_score_is_float(self):
        """Score should always be float."""
        result = SearchResult(id="doc1", score=1, payload={})

        assert isinstance(result.score, float)
        assert result.score == 1.0


# =============================================================================
# Stats Tests
# =============================================================================

class TestGetStats:
    """Test statistics reporting."""

    def test_get_stats(self, hybrid_engine, sample_documents):
        """Should return engine statistics."""
        hybrid_engine.index_batch(sample_documents)

        stats = hybrid_engine.get_stats()

        assert "mode" in stats
        assert "hybrid_alpha" in stats
        assert "indexed" in stats
        assert "total_documents" in stats
        assert "vocabulary_size" in stats

        assert stats["indexed"] is True
        assert stats["total_documents"] == 5


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_tokenize_empty_string(self, sparse_encoder):
        """Should handle empty string."""
        tokens = sparse_encoder._tokenize("")

        assert tokens == []

    def test_tokenize_special_characters(self, sparse_encoder):
        """Should handle special characters."""
        tokens = sparse_encoder._tokenize("hello@world.com #hashtag $100")

        # Should extract alphanumeric tokens
        assert "hello" in tokens
        assert "world" in tokens
        assert "com" in tokens
        assert "hashtag" in tokens
        assert "100" in tokens

    def test_index_empty_documents(self, sparse_encoder):
        """Should handle empty document list."""
        sparse_encoder.index_documents([])

        assert sparse_encoder.total_docs == 0

    @pytest.mark.asyncio
    async def test_search_unknown_mode(self, hybrid_engine):
        """Should fall back to dense for unknown mode."""
        hybrid_engine.config.mode = "unknown"
        dense_results = [("doc1", 0.9)]

        results = await hybrid_engine.search(
            query="test",
            dense_results=dense_results,
            limit=10,
        )

        # Should fall back to dense mode
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_with_payloads(self, hybrid_engine, sample_documents):
        """Should include payloads in results."""
        hybrid_engine.index_batch(sample_documents)

        payloads = {
            "doc1": {"title": "Fox Story"},
            "doc2": {"title": "ML Article"},
        }

        results = await hybrid_engine.search(
            query="machine learning",
            dense_results=[("doc2", 0.9)],
            dense_payloads=payloads,
            limit=10,
        )

        # Results for doc2 should have payload
        for r in results:
            if r.id == "doc2":
                assert r.payload.get("title") == "ML Article"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
