"""
Comprehensive Tests for Attention Module (XOR Attention)
========================================================

Tests the XOR-based attention mechanism including:
- AttentionConfig validation
- XORAttentionMasker functionality
- Context key building
- Novelty scoring
- Re-ranking functionality
"""

import pytest
from unittest.mock import MagicMock
from hypothesis import given, strategies as st, assume, settings, HealthCheck

from mnemocore.core.attention import (
    AttentionConfig,
    AttentionResult,
    XORAttentionMasker,
)
from mnemocore.core.binary_hdv import BinaryHDV


class TestAttentionConfig:
    """Test AttentionConfig dataclass."""

    def test_default_config(self):
        """Default configuration should have sensible values."""
        config = AttentionConfig()
        assert config.alpha == 0.6
        assert config.beta == 0.4
        assert config.context_sample_n == 50
        assert config.min_novelty_boost == 0.0
        assert config.enabled is True

    def test_custom_config(self):
        """Custom values should be set correctly."""
        config = AttentionConfig(
            alpha=0.7,
            beta=0.3,
            context_sample_n=100,
            min_novelty_boost=0.1,
            enabled=False,
        )
        assert config.alpha == 0.7
        assert config.beta == 0.3
        assert config.context_sample_n == 100
        assert config.min_novelty_boost == 0.1
        assert config.enabled is False

    def test_validate_valid_config(self):
        """Valid config should not raise assertion."""
        config = AttentionConfig(alpha=0.5, beta=0.5)
        config.validate()  # Should not raise

    def test_validate_alpha_bounds(self):
        """Alpha must be in [0, 1]."""
        with pytest.raises(AssertionError):
            AttentionConfig(alpha=1.5, beta=0.5).validate()

        with pytest.raises(AssertionError):
            AttentionConfig(alpha=-0.1, beta=0.5).validate()

    def test_validate_beta_bounds(self):
        """Beta must be in [0, 1]."""
        with pytest.raises(AssertionError):
            AttentionConfig(alpha=0.5, beta=1.5).validate()

    def test_validate_sum_equals_one(self):
        """Alpha + beta must equal 1.0."""
        with pytest.raises(AssertionError):
            AttentionConfig(alpha=0.5, beta=0.6).validate()

        # Should pass
        AttentionConfig(alpha=0.3, beta=0.7).validate()

    def test_validate_edge_cases(self):
        """Edge cases should be valid."""
        AttentionConfig(alpha=0.0, beta=1.0).validate()
        AttentionConfig(alpha=1.0, beta=0.0).validate()


class TestAttentionResult:
    """Test AttentionResult dataclass."""

    def test_result_creation(self):
        """Result should store all required fields."""
        result = AttentionResult(
            node_id="node123",
            raw_score=0.8,
            novelty_score=0.6,
            composite_score=0.72,
        )
        assert result.node_id == "node123"
        assert result.raw_score == 0.8
        assert result.novelty_score == 0.6
        assert result.composite_score == 0.72
        assert result.attention_mask is None

    def test_result_with_mask(self):
        """Result can include attention mask."""
        mask = BinaryHDV.random(1024)
        result = AttentionResult(
            node_id="node123",
            raw_score=0.8,
            novelty_score=0.6,
            composite_score=0.72,
            attention_mask=mask,
        )
        assert result.attention_mask is not None


class TestXORAttentionMasker:
    """Test XORAttentionMasker class."""

    def test_initialization_default_config(self):
        """Should initialize with default config."""
        masker = XORAttentionMasker()
        assert masker.config is not None
        assert masker.config.enabled is True

    def test_initialization_custom_config(self):
        """Should initialize with custom config."""
        config = AttentionConfig(alpha=0.7, beta=0.3)
        masker = XORAttentionMasker(config)
        assert masker.config is config

    def test_build_context_key_empty(self):
        """Empty context should return zero vector."""
        masker = XORAttentionMasker()
        context_key = masker.build_context_key([])

        assert context_key.dimension == 16384
        # Zero vector should have all bits unset
        assert context_key.similarity(BinaryHDV.zeros(16384)) == 1.0

    def test_build_context_key_single_vector(self):
        """Single vector context should return that vector."""
        masker = XORAttentionMasker()
        vec = BinaryHDV.random(1024)

        context_key = masker.build_context_key([vec])

        # Should be similar to the input
        assert context_key.similarity(vec) > 0.95

    def test_build_context_key_multiple_vectors(self):
        """Multiple vectors should be bundled."""
        masker = XORAttentionMasker()
        vec1 = BinaryHDV.from_seed("vec1", 1024)
        vec2 = BinaryHDV.from_seed("vec2", 1024)
        vec3 = BinaryHDV.from_seed("vec3", 1024)

        context_key = masker.build_context_key([vec1, vec2, vec3])

        # Context key should be different from individual vectors
        # but represent their combination
        assert context_key.dimension == 1024

    def test_build_attention_mask(self):
        """Should build XOR attention mask from query and context."""
        masker = XORAttentionMasker()
        query = BinaryHDV.from_seed("query", 1024)
        context = BinaryHDV.from_seed("context", 1024)

        mask = masker.build_attention_mask(query, context)

        assert mask.dimension == 1024
        # Mask should be different from both
        assert mask.similarity(query) < 0.9
        assert mask.similarity(context) < 0.9

    def test_novelty_score(self):
        """Novelty score should be similarity with mask."""
        masker = XORAttentionMasker()
        mask = BinaryHDV.from_seed("mask", 1024)

        # Candidate similar to mask should have high novelty
        similar_candidate = BinaryHDV.from_seed("mask", 1024)
        score1 = masker.novelty_score(mask, similar_candidate)
        assert score1 > 0.9

        # Dissimilar candidate should have low novelty
        different_candidate = BinaryHDV.random(1024)
        score2 = masker.novelty_score(mask, different_candidate)
        # Should be around 0.5 for random vectors
        assert 0.4 < score2 < 0.6

    def test_rerank_basic(self):
        """Re-ranking should produce sorted results."""
        masker = XORAttentionMasker()
        raw_scores = {
            "node1": 0.8,
            "node2": 0.6,
            "node3": 0.9,
        }
        memory_vectors = {
            "node1": BinaryHDV.from_seed("mem1", 1024),
            "node2": BinaryHDV.from_seed("mem2", 1024),
            "node3": BinaryHDV.from_seed("mem3", 1024),
        }
        mask = BinaryHDV.from_seed("mask", 1024)

        results = masker.rerank(raw_scores, memory_vectors, mask)

        assert len(results) == 3
        # Should be sorted by composite_score
        assert results[0].composite_score >= results[1].composite_score
        assert results[1].composite_score >= results[2].composite_score

    def test_rerank_missing_vector(self):
        """Should handle missing memory vectors."""
        masker = XORAttentionMasker()
        raw_scores = {
            "node1": 0.8,
            "node2": 0.6,
        }
        memory_vectors = {
            "node1": BinaryHDV.from_seed("mem1", 1024),
            # node2 is missing
        }
        mask = BinaryHDV.from_seed("mask", 1024)

        results = masker.rerank(raw_scores, memory_vectors, mask)

        assert len(results) == 2
        # node2 should use min_novelty_boost
        node2_result = next(r for r in results if r.node_id == "node2")
        assert node2_result.novelty_score == masker.config.min_novelty_boost

    def test_rerank_composite_calculation(self):
        """Composite score should be weighted combination."""
        config = AttentionConfig(alpha=0.7, beta=0.3)
        masker = XORAttentionMasker(config)

        raw_scores = {"node1": 0.8}
        memory_vectors = {"node1": BinaryHDV.from_seed("mem1", 1024)}
        mask = BinaryHDV.from_seed("mask", 1024)

        results = masker.rerank(raw_scores, memory_vectors, mask)

        # composite = alpha * raw + beta * novelty
        result = results[0]
        expected = 0.7 * 0.8 + 0.3 * result.novelty_score
        assert abs(result.composite_score - expected) < 0.01

    def test_extract_scores(self):
        """Extract scores should convert results to tuples."""
        masker = XORAttentionMasker()
        results = [
            AttentionResult("node1", 0.8, 0.5, 0.7),
            AttentionResult("node2", 0.6, 0.7, 0.64),
        ]

        scores = masker.extract_scores(results)

        assert scores == [("node1", 0.7), ("node2", 0.64)]


class TestXORAttentionMaskerEdgeCases:
    """Test edge cases."""

    def test_empty_raw_scores(self):
        """Should handle empty score dict."""
        masker = XORAttentionMasker()
        results = masker.rerank({}, {}, BinaryHDV.zeros(1024))
        assert results == []

    def test_different_dimensions(self):
        """Should handle vectors of various dimensions."""
        for dim in [512, 1024, 2048, 4096, 8192, 16384]:
            masker = XORAttentionMasker()
            vec = BinaryHDV.random(dim)
            context_key = masker.build_context_key([vec])
            assert context_key.dimension == dim

    def test_identical_vectors_in_context(self):
        """Should handle identical vectors in context."""
        masker = XORAttentionMasker()
        vec = BinaryHDV.from_seed("same", 1024)

        context_key = masker.build_context_key([vec, vec, vec])

        # Should still produce valid context key
        assert context_key.dimension == 1024

    def test_min_novelty_boost_floor(self):
        """Novelty score should have minimum floor."""
        config = AttentionConfig(min_novelty_boost=0.2)
        masker = XORAttentionMasker(config)

        raw_scores = {"node1": 0.8}
        memory_vectors = {"node1": BinaryHDV.zeros(1024)}  # Zero similarity with mask
        mask = BinaryHDV.zeros(1024)

        results = masker.rerank(raw_scores, memory_vectors, mask)

        # Even with 0 similarity, should get min_novelty_boost
        assert results[0].novelty_score >= 0.2


class TestXORAttentionMaskerProperties:
    """Test mathematical properties."""

    def test_xor_mask_properties(self):
        """XOR mask should have specific properties."""
        masker = XORAttentionMasker()
        query = BinaryHDV.from_seed("query", 1024)
        context = BinaryHDV.from_seed("context", 1024)

        mask = masker.build_attention_mask(query, context)

        # XOR is self-inverse: query XOR context XOR context == query
        # But we're only doing one XOR here
        # Just verify mask is different from both
        assert mask != query
        assert mask != context

    def test_mask_preserves_dimension(self):
        """Mask should always preserve input dimension."""
        masker = XORAttentionMasker()
        for dim in [256, 512, 1024, 2048]:
            query = BinaryHDV.random(dim)
            context = BinaryHDV.random(dim)
            mask = masker.build_attention_mask(query, context)
            assert mask.dimension == dim


class TestXORAttentionIntegration:
    """Integration tests with attention workflow."""

    def test_full_attention_workflow(self):
        """Test complete attention workflow."""
        masker = XORAttentionMasker()

        # 1. Build context from recent memories
        context_vecs = [
            BinaryHDV.from_seed(f"mem{i}", 1024)
            for i in range(10)
        ]
        context_key = masker.build_context_key(context_vecs)

        # 2. Build attention mask for query
        query = BinaryHDV.from_seed("user query", 1024)
        mask = masker.build_attention_mask(query, context_key)

        # 3. Score candidates
        raw_scores = {f"node{i}": 0.5 + i * 0.05 for i in range(5)}
        memory_vectors = {
            f"node{i}": BinaryHDV.from_seed(f"node{i}", 1024)
            for i in range(5)
        }

        results = masker.rerank(raw_scores, memory_vectors, mask)

        # 4. Extract final scores
        final_scores = masker.extract_scores(results)

        assert len(final_scores) == 5
        # All scores should be in valid range
        for _, score in final_scores:
            assert 0.0 <= score <= 1.0

    def test_attention_ordering_stability(self):
        """Results should be consistently ordered."""
        masker = XORAttentionMasker()

        raw_scores = {"a": 0.7, "b": 0.5, "c": 0.9}
        memory_vectors = {
            k: BinaryHDV.from_seed(k, 1024)
            for k in raw_scores.keys()
        }
        mask = BinaryHDV.random(1024)

        results1 = masker.rerank(raw_scores, memory_vectors, mask)
        results2 = masker.rerank(raw_scores, memory_vectors, mask)

        # Same inputs should produce same order
        assert [r.node_id for r in results1] == [r.node_id for r in results2]


class TestXORAttentionPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(st.lists(st.just("x"), min_size=1, max_size=100))
    def test_context_key_dimension_matches_input(self, _):
        """Context key dimension should match input vectors."""
        masker = XORAttentionMasker()
        dim = 1024

        vecs = [BinaryHDV.random(dim) for _ in range(10)]
        context_key = masker.build_context_key(vecs)

        assert context_key.dimension == dim

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_alpha_beta_valid_combinations(self, alpha):
        """Only valid alpha/beta combinations should validate."""
        beta = 1.0 - alpha

        config = AttentionConfig(alpha=alpha, beta=beta)
        config.validate()  # Should not raise

    @given(st.integers(min_value=64, max_value=2048).map(lambda x: x * 8))
    def test_various_dimensions(self, dim):
        """Should handle various vector dimensions."""
        masker = XORAttentionMasker()
        query = BinaryHDV.random(dim)
        context = BinaryHDV.random(dim)

        mask = masker.build_attention_mask(query, context)
        assert mask.dimension == dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
