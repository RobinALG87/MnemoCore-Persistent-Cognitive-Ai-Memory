"""
Tests for Context Window Prioritizer and Optimizer
==================================================
Comprehensive tests for LLM context window optimization.

Tests cover:
- Token counting accuracy (tiktoken vs heuristic fallback)
- Context window optimization for different model limits
- Paragraph splitting with overlap
- Diversity filtering
"""

import math
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from mnemocore.cognitive.context_optimizer import (
    ModelProvider,
    ModelContextLimits,
    MODEL_LIMITS,
    RankedMemory,
    ChunkConfig,
    ScoringWeights,
    OptimizationResult,
    TokenCounter,
    SemanticChunker,
    ContextWindowPrioritizer,
    ContextBuilder,
    create_prioritizer,
    rank_memories,
)
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.node import MemoryNode


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """This is a sample text for testing the token counter.
It contains multiple sentences and paragraphs.

The second paragraph has additional content.
This helps test the chunking functionality.

The third paragraph concludes the sample text with more words
to ensure we have enough content for meaningful tests."""


@pytest.fixture
def sample_memories():
    """Create sample memory nodes for testing."""
    memories = []
    for i in range(5):
        node = MemoryNode(
            id=f"mem_{i}",
            hdv=BinaryHDV.random(1024),
            content=f"Memory content {i}: " + "word " * (50 + i * 20),
            created_at=datetime.now(timezone.utc) - timedelta(hours=i * 2)
        )
        node.ltp_strength = 0.5 + i * 0.1
        node.access_count = i + 1
        memories.append(node)
    return memories


@pytest.fixture
def chunk_config():
    """Create default chunk configuration."""
    return ChunkConfig(
        max_chunk_tokens=512,
        min_chunk_tokens=128,
        chunk_overlap_tokens=64,
        split_by_sentences=True,
        preserve_structure=True
    )


@pytest.fixture
def token_counter():
    """Create a token counter for testing."""
    return TokenCounter(model_provider=ModelProvider.OPENAI)


@pytest.fixture
def prioritizer():
    """Create a context window prioritizer for testing."""
    return ContextWindowPrioritizer(
        model_name="gpt-4o",
        token_budget=4096
    )


class TestModelProvider:
    """Tests for ModelProvider enum."""

    def test_provider_values(self):
        """Test ModelProvider has expected values."""
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.GOOGLE.value == "google"
        assert ModelProvider.COHERE.value == "cohere"
        assert ModelProvider.META.value == "meta"
        assert ModelProvider.MISTRAL.value == "mistral"
        assert ModelProvider.GENERIC.value == "generic"


class TestModelContextLimits:
    """Tests for ModelContextLimits dataclass."""

    def test_gpt4o_limits(self):
        """Test GPT-4o context limits."""
        limits = MODEL_LIMITS["gpt-4o"]
        assert limits.max_tokens == 128000
        assert limits.output_tokens == 4096
        assert limits.provider == ModelProvider.OPENAI

    def test_input_budget(self):
        """Test input_budget calculation."""
        limits = ModelContextLimits(
            model_name="test",
            max_tokens=10000,
            output_tokens=2000
        )
        assert limits.input_budget == 8000

    def test_claude_limits(self):
        """Test Claude context limits."""
        limits = MODEL_LIMITS["claude-3-5-sonnet-20241022"]
        assert limits.max_tokens == 200000
        assert limits.provider == ModelProvider.ANTHROPIC

    def test_gemini_limits(self):
        """Test Gemini context limits."""
        limits = MODEL_LIMITS["gemini-1.5-pro"]
        assert limits.max_tokens == 1000000
        assert limits.provider == ModelProvider.GOOGLE


class TestTokenCounting:
    """Tests for token counting accuracy."""

    def test_count_tokens_non_empty(self, token_counter):
        """Test counting tokens in non-empty text."""
        text = "Hello world, this is a test."
        count = token_counter.count_tokens(text)
        assert count > 0

    def test_count_tokens_empty(self, token_counter):
        """Test counting tokens in empty text."""
        count = token_counter.count_tokens("")
        assert count == 0

    def test_count_tokens_whitespace(self, token_counter):
        """Test counting tokens in whitespace-only text."""
        count = token_counter.count_tokens("   \n\t  ")
        assert count >= 0

    def test_count_tokens_longer_text(self, token_counter, sample_text):
        """Test counting tokens in longer text."""
        count = token_counter.count_tokens(sample_text)
        assert count > 10  # Should have multiple tokens

    def test_count_tokens_batch(self, token_counter):
        """Test batch token counting."""
        texts = [
            "First text",
            "Second text with more words",
            "Third text"
        ]
        counts = token_counter.count_tokens_batch(texts)
        assert len(counts) == 3
        assert all(c > 0 for c in counts)

    def test_heuristic_fallback(self):
        """Test heuristic fallback when tiktoken is unavailable."""
        counter = TokenCounter(model_provider=ModelProvider.OPENAI)
        # Force encoding to None to test fallback
        counter.encoding = None

        text = "This is a test of heuristic token counting"
        count = counter.count_tokens(text)

        # Heuristic uses ~4 chars per token
        expected = max(1, len(text) // 4)
        assert count == expected

    def test_heuristic_minimum_one(self):
        """Test heuristic returns at least 1 for non-empty text."""
        counter = TokenCounter(model_provider=ModelProvider.OPENAI)
        counter.encoding = None

        count = counter.count_tokens("a")
        assert count >= 1

    def test_for_model_openai(self):
        """Test creating counter for OpenAI model."""
        counter = TokenCounter.for_model("gpt-4o")
        assert counter.provider == ModelProvider.OPENAI

    def test_for_model_anthropic(self):
        """Test creating counter for Anthropic model."""
        counter = TokenCounter.for_model("claude-3-opus-20240229")
        assert counter.provider == ModelProvider.ANTHROPIC

    def test_for_model_google(self):
        """Test creating counter for Google model."""
        counter = TokenCounter.for_model("gemini-1.5-pro")
        assert counter.provider == ModelProvider.GOOGLE

    def test_for_model_meta(self):
        """Test creating counter for Meta model."""
        counter = TokenCounter.for_model("llama-3.1-70b")
        assert counter.provider == ModelProvider.META

    def test_for_model_mistral(self):
        """Test creating counter for Mistral model."""
        counter = TokenCounter.for_model("mistral-large")
        assert counter.provider == ModelProvider.MISTRAL

    def test_for_model_generic(self):
        """Test creating counter for unknown model."""
        counter = TokenCounter.for_model("unknown-model")
        assert counter.provider == ModelProvider.GENERIC


class TestContextWindowOptimization:
    """Tests for context window optimization for different model limits."""

    def test_prioritizer_default_budget(self):
        """Test prioritizer uses model's default budget."""
        prioritizer = ContextWindowPrioritizer(model_name="gpt-4o")
        assert prioritizer.token_budget == MODEL_LIMITS["gpt-4o"].input_budget

    def test_prioritizer_custom_budget(self):
        """Test prioritizer with custom budget."""
        prioritizer = ContextWindowPrioritizer(
            model_name="gpt-4o",
            token_budget=2000
        )
        assert prioritizer.token_budget == 2000

    def test_select_within_budget(self, prioritizer, sample_memories):
        """Test that select() stays within token budget."""
        result = prioritizer.select(sample_memories, query="test")

        assert result.total_tokens <= prioritizer.token_budget
        assert result.remaining_budget >= 0

    def test_select_returns_optimization_result(self, prioritizer, sample_memories):
        """Test that select() returns OptimizationResult."""
        result = prioritizer.select(sample_memories, query="test")

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.ranked_memories, list)
        assert isinstance(result.coverage_stats, dict)

    def test_rank_returns_ranked_memories(self, prioritizer, sample_memories):
        """Test that rank() returns RankedMemory list."""
        ranked = prioritizer.rank(sample_memories, query="memory content")

        assert all(isinstance(r, RankedMemory) for r in ranked)
        # Should be sorted by score descending
        if len(ranked) > 1:
            for i in range(len(ranked) - 1):
                assert ranked[i].score >= ranked[i + 1].score

    def test_rank_with_budget(self, prioritizer, sample_memories):
        """Test rank_with_budget filters by token limit."""
        ranked = prioritizer.rank_with_budget(
            sample_memories,
            token_budget=500,
            query="test"
        )

        total_tokens = sum(r.token_count for r in ranked)
        assert total_tokens <= 500

    def test_max_memories_limit(self, prioritizer, sample_memories):
        """Test that max_memories limit is respected."""
        result = prioritizer.select(
            sample_memories,
            query="test",
            max_memories=2
        )

        assert len(result.ranked_memories) <= 2


class TestParagraphSplitting:
    """Tests for paragraph splitting with overlap."""

    def test_short_text_not_split(self, token_counter, chunk_config):
        """Test that short text is not split."""
        chunker = SemanticChunker(chunk_config, token_counter)
        text = "This is a short text."

        chunks = chunker.split(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_by_paragraphs(self, token_counter, chunk_config):
        """Test splitting by paragraphs."""
        chunker = SemanticChunker(chunk_config, token_counter)
        # Create text with multiple paragraphs
        text = "\n\n".join([
            "Paragraph one with some content.",
            "Paragraph two with more content.",
            "Paragraph three with additional content."
        ])

        chunks = chunker.split(text)

        assert len(chunks) >= 1
        # Each chunk should preserve paragraph structure
        for chunk in chunks:
            assert len(chunk) > 0

    def test_empty_text_returns_empty(self, token_counter, chunk_config):
        """Test that empty text returns empty list."""
        chunker = SemanticChunker(chunk_config, token_counter)

        chunks = chunker.split("")
        assert chunks == []

        chunks = chunker.split("   ")
        assert chunks == []

    def test_long_text_is_chunked(self, token_counter):
        """Test that long text is chunked."""
        # Use small chunk size to force chunking
        config = ChunkConfig(
            max_chunk_tokens=20,  # Very small
            min_chunk_tokens=5,
            chunk_overlap_tokens=5
        )
        chunker = SemanticChunker(config, token_counter)

        # Create long text
        text = " ".join(["word"] * 200)
        chunks = chunker.split(text)

        assert len(chunks) > 1

    def test_split_by_sentences(self, token_counter):
        """Test splitting by sentences."""
        config = ChunkConfig(
            max_chunk_tokens=20,
            split_by_sentences=True,
            preserve_structure=False
        )
        chunker = SemanticChunker(config, token_counter)

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.split(text)

        assert len(chunks) >= 1

    def test_within_limits_check(self, token_counter, chunk_config):
        """Test _within_limits check."""
        # Clean config for this test to avoid min_chunk_tokens restrictions
        test_config = ChunkConfig(
            max_chunk_tokens=512,
            min_chunk_tokens=1, 
            chunk_overlap_tokens=0,
        )
        chunker = SemanticChunker(test_config, token_counter)

        # Short chunks should be within limits of the loose config
        chunks = ["Short text", "Another short text"]
        assert chunker._within_limits(chunks) is True


class TestDiversityFiltering:
    """Tests for diversity filtering."""

    def test_diversity_enabled(self, prioritizer, sample_memories):
        """Test selection with diversity enabled."""
        # Create similar memories
        similar_memories = []
        for i in range(3):
            node = MemoryNode(
                id=f"similar_{i}",
                hdv=BinaryHDV.random(1024),
                content="This is very similar content for testing diversity",
                created_at=datetime.now(timezone.utc)
            )
            similar_memories.append(node)

        result = prioritizer.select(
            similar_memories,
            query="test",
            promote_diversity=True
        )

        # Should filter out redundant content
        assert len(result.ranked_memories) <= len(similar_memories)

    def test_diversity_disabled(self, prioritizer):
        """Test selection with diversity disabled."""
        # Create similar memories
        similar_memories = []
        for i in range(3):
            node = MemoryNode(
                id=f"similar_{i}",
                hdv=BinaryHDV.random(1024),
                content="This is very similar content for testing diversity",
                created_at=datetime.now(timezone.utc)
            )
            node.ltp_strength = 0.8
            similar_memories.append(node)

        result_with_diversity = prioritizer.select(
            similar_memories.copy(),
            query="test",
            promote_diversity=True
        )

        result_without_diversity = prioritizer.select(
            similar_memories.copy(),
            query="test",
            promote_diversity=False
        )

        # Without diversity should potentially include more
        assert result_without_diversity.metadata["promoted_diversity"] is False

    def test_content_signature_generation(self, prioritizer):
        """Test content signature generation for diversity."""
        ranked = RankedMemory(
            memory={"content": "Test content for signature"},
            score=0.5,
            token_count=10,
            relevance=0.5,
            recency_weight=0.5,
            importance=0.5
        )

        signature = prioritizer._get_content_signature(ranked)
        assert isinstance(signature, str)
        assert len(signature) > 0


class TestRankedMemory:
    """Tests for RankedMemory dataclass."""

    def test_ranked_memory_creation(self):
        """Test RankedMemory creation."""
        memory = MemoryNode(
            id="test_id",
            hdv=BinaryHDV.random(1024),
            content="Test content"
        )
        ranked = RankedMemory(
            memory=memory,
            score=0.75,
            token_count=50,
            relevance=0.8,
            recency_weight=0.9,
            importance=0.7
        )

        assert ranked.score == 0.75
        assert ranked.token_count == 50
        assert ranked.chunk_index == 0
        assert ranked.total_chunks == 1

    def test_content_property_memory_node(self):
        """Test content property with MemoryNode."""
        node = MemoryNode(id="test", hdv=BinaryHDV.random(1024), content="Node content")
        ranked = RankedMemory(
            memory=node,
            score=0.5,
            token_count=10,
            relevance=0.5,
            recency_weight=0.5,
            importance=0.5
        )

        assert ranked.content == "Node content"

    def test_content_property_dict(self):
        """Test content property with dict."""
        ranked = RankedMemory(
            memory={"content": "Dict content"},
            score=0.5,
            token_count=10,
            relevance=0.5,
            recency_weight=0.5,
            importance=0.5
        )

        assert ranked.content == "Dict content"

    def test_memory_id_property(self):
        """Test memory_id property."""
        node = MemoryNode(id="test_id", hdv=BinaryHDV.random(1024), content="Test")
        ranked = RankedMemory(
            memory=node,
            score=0.5,
            token_count=10,
            relevance=0.5,
            recency_weight=0.5,
            importance=0.5
        )

        assert ranked.memory_id == "test_id"


class TestScoringWeights:
    """Tests for ScoringWeights dataclass."""

    def test_default_weights(self):
        """Test default scoring weights."""
        weights = ScoringWeights()

        assert weights.relevance == 1.0
        assert weights.recency == 0.5
        assert weights.importance == 0.7
        assert weights.token_efficiency == 0.3

    def test_normalize(self):
        """Test weight normalization."""
        weights = ScoringWeights(
            relevance=2.0,
            recency=2.0,
            importance=2.0,
            token_efficiency=2.0
        )

        normalized = weights.normalize()

        # Sum should be 1.0
        total = (normalized.relevance + normalized.recency +
                 normalized.importance + normalized.token_efficiency)
        assert abs(total - 1.0) < 0.01

    def test_normalize_zero_weights(self):
        """Test normalization with all zero weights."""
        weights = ScoringWeights(
            relevance=0.0,
            recency=0.0,
            importance=0.0,
            token_efficiency=0.0
        )

        normalized = weights.normalize()
        # Should return defaults
        assert normalized.relevance == 1.0


class TestChunkConfig:
    """Tests for ChunkConfig dataclass."""

    def test_default_config(self):
        """Test default chunk configuration."""
        config = ChunkConfig()

        assert config.max_chunk_tokens == 512
        assert config.min_chunk_tokens == 128
        assert config.chunk_overlap_tokens == 64
        assert config.split_by_sentences is True
        assert config.preserve_structure is True

    def test_custom_config(self):
        """Test custom chunk configuration."""
        config = ChunkConfig(
            max_chunk_tokens=1024,
            min_chunk_tokens=256,
            chunk_overlap_tokens=128
        )

        assert config.max_chunk_tokens == 1024
        assert config.min_chunk_tokens == 256


class TestContextBuilder:
    """Tests for ContextBuilder."""

    def test_build_context(self, prioritizer, sample_memories):
        """Test building context string."""
        builder = ContextBuilder(prioritizer)

        context = builder.build_context(sample_memories, query="test")

        assert isinstance(context, str)
        assert len(context) > 0

    def test_build_context_with_max_tokens(self, prioritizer, sample_memories):
        """Test building context with token limit."""
        builder = ContextBuilder(prioritizer)

        context = builder.build_context(
            sample_memories,
            query="test",
            max_tokens=500
        )

        # Context should fit within budget (approximately)
        token_count = prioritizer.counter.count_tokens(context)
        assert token_count <= 600  # Allow some margin

    def test_build_rag_context(self, prioritizer, sample_memories):
        """Test building RAG context with citations."""
        builder = ContextBuilder(prioritizer)

        context = builder.build_rag_context(
            sample_memories,
            query="test query"
        )

        assert "Query: test query" in context
        assert "Citations:" in context


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_prioritizer(self):
        """Test create_prioritizer factory function."""
        prioritizer = create_prioritizer(
            model_name="gpt-4o",
            token_budget=8000
        )

        assert isinstance(prioritizer, ContextWindowPrioritizer)
        assert prioritizer.model_name == "gpt-4o"
        assert prioritizer.token_budget == 8000

    def test_rank_memories_convenience(self, sample_memories):
        """Test rank_memories convenience function."""
        contents = rank_memories(
            memories=sample_memories,
            query="memory",
            token_budget=1000,
            model="gpt-4o"
        )

        assert isinstance(contents, list)
        assert all(isinstance(c, str) for c in contents)


class TestRecencyWeight:
    """Tests for recency weight calculation."""

    def test_recent_memory_high_weight(self, prioritizer):
        """Test recent memory has high recency weight."""
        recent_time = datetime.now(timezone.utc)
        weight = prioritizer._calculate_recency_weight(recent_time, recent_time)

        assert weight > 0.9  # Should be close to 1.0

    def test_old_memory_lower_weight(self, prioritizer):
        """Test old memory has lower recency weight."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(hours=48)

        weight = prioritizer._calculate_recency_weight(old_time, now)

        assert weight < 0.5  # Should be decayed significantly

    def test_minimum_weight(self, prioritizer):
        """Test minimum recency weight."""
        now = datetime.now(timezone.utc)
        very_old = now - timedelta(days=365)

        weight = prioritizer._calculate_recency_weight(very_old, now)

        assert weight >= 0.1  # Should not go below minimum


class TestCoverageStats:
    """Tests for coverage statistics."""

    def test_coverage_stats_calculation(self, prioritizer, sample_memories):
        """Test coverage statistics are calculated."""
        result = prioritizer.select(sample_memories, query="test")

        stats = result.coverage_stats

        assert "selected_count" in stats
        assert "total_count" in stats
        assert "coverage_ratio" in stats
        assert "avg_importance" in stats
        assert "avg_relevance" in stats

    def test_coverage_ratio(self, prioritizer, sample_memories):
        """Test coverage ratio calculation."""
        result = prioritizer.select(
            sample_memories,
            query="test",
            max_memories=2
        )

        ratio = result.coverage_stats["coverage_ratio"]
        assert 0.0 <= ratio <= 1.0
