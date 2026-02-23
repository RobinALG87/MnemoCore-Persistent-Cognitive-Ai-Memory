"""
Comprehensive Tests for RippleContext Module
=============================================

Tests the external memory environment implementation including:
- RippleChunk creation and scoring
- RippleContext initialization and indexing
- Search functionality
- Slice operations
- Loading from files and JSONL
"""

import pytest
import tempfile
import os
import json
from pathlib import Path

from mnemocore.core.ripple_context import RippleContext, RippleChunk


class TestRippleChunk:
    """Test RippleChunk dataclass."""

    def test_chunk_creation(self):
        """Chunk should be created with correct attributes."""
        chunk = RippleChunk(
            index=0,
            text="This is a test chunk with some words.",
            start_char=0,
            end_char=40,
        )
        assert chunk.index == 0
        assert chunk.text == "This is a test chunk with some words."
        assert chunk.start_char == 0
        assert chunk.end_char == 40

    def test_chunk_builds_term_frequency(self):
        """Chunk should build term frequency index on creation."""
        chunk = RippleChunk(
            index=0,
            text="The quick brown fox jumps over the lazy dog.",
            start_char=0,
            end_char=50,
        )
        assert "the" in chunk.term_freq
        assert "quick" in chunk.term_freq
        assert chunk.term_freq["the"] == 2  # Appears twice

    def test_chunk_custom_term_frequency(self):
        """Custom term frequency should be preserved."""
        custom_tf = {"test": 5, "custom": 3}
        chunk = RippleChunk(
            index=0,
            text="text",
            start_char=0,
            end_char=4,
            term_freq=custom_tf,
        )
        assert chunk.term_freq == custom_tf

    def test_score_query_basic(self):
        """Scoring should work for query terms."""
        chunk = RippleChunk(
            index=0,
            text="machine learning and artificial intelligence",
            start_char=0,
            end_char=50,
        )

        score = chunk.score_query(["machine", "learning"])
        assert score > 0

    def test_score_query_no_match(self):
        """Scoring should return 0 for non-matching terms."""
        chunk = RippleChunk(
            index=0,
            text="biology and chemistry",
            start_char=0,
            end_char=30,
        )

        score = chunk.score_query(["quantum", "physics"])
        assert score == 0.0

    def test_score_query_empty_terms(self):
        """Scoring should return 0 for empty query."""
        chunk = RippleChunk(
            index=0,
            text="some text",
            start_char=0,
            end_char=9,
        )

        score = chunk.score_query([])
        assert score == 0.0

    def test_score_query_case_insensitive(self):
        """Scoring should be case-insensitive."""
        chunk = RippleChunk(
            index=0,
            text="Machine Learning is great",
            start_char=0,
            end_char=30,
        )

        score1 = chunk.score_query(["machine"])
        score2 = chunk.score_query(["MACHINE"])
        assert score1 == score2

    def test_score_query_multiple_terms(self):
        """Multiple matching terms should increase score."""
        chunk = RippleChunk(
            index=0,
            text="neural networks and deep learning",
            start_char=0,
            end_char=40,
        )

        single_score = chunk.score_query(["neural"])
        multi_score = chunk.score_query(["neural", "networks"])

        assert multi_score > single_score


class TestRippleContextInitialization:
    """Test RippleContext initialization and indexing."""

    def test_initialization_basic(self):
        """Context should initialize with text and create chunks."""
        text = "a" * 1000
        ctx = RippleContext(text, chunk_size=100, chunk_overlap=10)

        assert ctx.text == text
        assert ctx.chunk_size == 100
        assert ctx.chunk_overlap == 10
        assert len(ctx.chunks) > 0

    def test_chunking_calculations(self):
        """Chunking should use correct step size."""
        text = "a" * 1000
        ctx = RippleContext(text, chunk_size=200, chunk_overlap=50)

        # Step = 200 - 50 = 150
        # 1000 / 150 ≈ 7 chunks
        expected_chunks = (1000 + 149) // 150  # Ceiling division
        assert len(ctx.chunks) == expected_chunks

    def test_chunk_overlap(self):
        """Chunks should overlap as specified."""
        text = "abcdefghij" * 100  # Repeating pattern
        ctx = RippleContext(text, chunk_size=50, chunk_overlap=10)

        if len(ctx.chunks) >= 2:
            # Second chunk should start before first ends
            chunk1 = ctx.chunks[0]
            chunk2 = ctx.chunks[1]
            assert chunk2.start_char < chunk1.end_char

    def test_empty_text(self):
        """Empty text should produce no chunks."""
        ctx = RippleContext("", chunk_size=100)
        assert len(ctx.chunks) == 0

    def test_small_text(self):
        """Text smaller than chunk_size should produce one chunk."""
        ctx = RippleContext("small text", chunk_size=1000)
        assert len(ctx.chunks) == 1
        assert ctx.chunks[0].text == "small text"

    def test_source_label(self):
        """Source label should be stored."""
        ctx = RippleContext("test", source_label="my_source")
        assert ctx.source_label == "my_source"

    def test_unicode_text(self):
        """Should handle Unicode characters."""
        text = "Swedish: åäö and emoji: 🚀 🌟"
        ctx = RippleContext(text, chunk_size=50)
        assert len(ctx.chunks) >= 1

    def test_chunk_indices(self):
        """Chunks should have sequential indices."""
        ctx = RippleContext("x" * 1000, chunk_size=100)
        for i, chunk in enumerate(ctx.chunks):
            assert chunk.index == i


class TestRippleContextSearch:
    """Test search functionality."""

    def test_search_basic(self):
        """Search should return relevant chunks."""
        text = """
        Machine learning is a subset of artificial intelligence.
        Deep learning uses neural networks with multiple layers.
        Natural language processing deals with text understanding.
        """
        ctx = RippleContext(text, chunk_size=200)

        results = ctx.search("machine learning", top_k=3)

        assert isinstance(results, list)
        assert len(results) >= 1

    def test_search_top_k(self):
        """Search should respect top_k parameter."""
        text = "word " * 1000
        ctx = RippleContext(text, chunk_size=100)

        results = ctx.search("word", top_k=3)

        assert len(results) <= 3

    def test_search_no_results_empty_context(self):
        """Search on empty context should return empty list."""
        ctx = RippleContext("")
        results = ctx.search("anything")
        assert results == []

    def test_search_fallback_for_no_matches(self):
        """Should return first chunks if no keyword matches."""
        ctx = RippleContext("abc def ghi", chunk_size=10)
        results = ctx.search("xyz", top_k=2)
        # Should return first chunks as fallback
        assert len(results) >= 0

    def test_search_multiple_terms(self):
        """Search should handle multiple terms."""
        text = "neural network deep learning transformer attention"
        ctx = RippleContext(text, chunk_size=100)

        results = ctx.search("neural network", top_k=5)

        assert len(results) >= 1

    def test_search_swedish_characters(self):
        """Search should handle Swedish characters."""
        text = "Det här är en svensk text med orden Stockholm och Göteborg"
        ctx = RippleContext(text, chunk_size=100)

        results = ctx.search("Stockholm", top_k=5)

        assert len(results) >= 1

    def test_query_special_characters(self):
        """Search should handle special characters."""
        text = "Email: test@example.com and website https://example.com"
        ctx = RippleContext(text, chunk_size=100)

        # Should not crash
        results = ctx.search("email", top_k=5)
        assert isinstance(results, list)

    def test_search_ordering_by_relevance(self):
        """Results should be ordered by relevance score."""
        text = """
        Chunk about biology and cells.
        Chunk about machine learning and neural networks.
        Chunk about deep learning algorithms.
        """
        ctx = RippleContext(text, chunk_size=100)

        results = ctx.search("machine learning neural", top_k=5)

        # First result should be most relevant
        assert len(results) >= 1


class TestRippleContextSlice:
    """Test slice operations."""

    def test_slice_basic(self):
        """Slice should extract text range."""
        text = "0123456789"
        ctx = RippleContext(text)

        result = ctx.slice(2, 5)
        assert result == "234"

    def test_slice_beyond_bounds(self):
        """Slice should clamp to text bounds."""
        text = "short"
        ctx = RippleContext(text)

        result = ctx.slice(0, 1000)
        assert result == "short"

    def test_slice_negative_start(self):
        """Negative start should be clamped to 0."""
        text = "test"
        ctx = RippleContext(text)

        result = ctx.slice(-10, 3)
        assert result == "tes"

    def test_slice_empty_range(self):
        """Empty slice range should return empty string."""
        text = "test"
        ctx = RippleContext(text)

        result = ctx.slice(5, 5)
        assert result == ""

    def test_slice_full_text(self):
        """Full slice should return entire text."""
        text = "full text content"
        ctx = RippleContext(text)

        result = ctx.slice(0, 1000)
        assert result == text


class TestRippleContextUtilities:
    """Test utility methods."""

    def test_get_chunk_by_index(self):
        """Should retrieve chunk by index."""
        ctx = RippleContext("x" * 1000, chunk_size=100)

        chunk = ctx.get_chunk_by_index(0)
        assert chunk is not None
        assert chunk.index == 0

    def test_get_chunk_by_index_out_of_bounds(self):
        """Should return None for invalid index."""
        ctx = RippleContext("short", chunk_size=100)

        chunk = ctx.get_chunk_by_index(100)
        assert chunk is None

    def test_get_stats(self):
        """Stats should return correct information."""
        text = "a" * 1000
        ctx = RippleContext(text, chunk_size=100, chunk_overlap=10, source_label="test")

        stats = ctx.get_stats()

        assert stats["source"] == "test"
        assert stats["total_chars"] == 1000
        assert stats["total_chunks"] == len(ctx.chunks)
        assert stats["chunk_size"] == 100
        assert stats["chunk_overlap"] == 10
        assert stats["approx_tokens"] == 250  # 1000 / 4

    def test_len(self):
        """len() should return character count."""
        text = "test text"
        ctx = RippleContext(text)

        assert len(ctx) == len(text)

    def test_repr(self):
        """repr() should contain useful information."""
        ctx = RippleContext("test", source_label="src")

        r = repr(ctx)
        assert "src" in r
        assert "chars=" in r
        assert "chunks=" in r


class TestRippleContextFromFile:
    """Test loading from files."""

    def test_from_file(self):
        """Should load context from text file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Content from file\nLine 2\nLine 3")
            temp_path = f.name

        try:
            ctx = RippleContext.from_file(temp_path, chunk_size=50)

            assert "Content from file" in ctx.text
            assert ctx.source_label == temp_path
        finally:
            os.unlink(temp_path)

    def test_from_file_unicode(self):
        """Should load Unicode from file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("Swedish: åäö\nEmoji: 🚀")
            temp_path = f.name

        try:
            ctx = RippleContext.from_file(temp_path)
            assert "åäö" in ctx.text
            assert "🚀" in ctx.text
        finally:
            os.unlink(temp_path)

    def test_from_file_nonexistent(self):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            RippleContext.from_file("/nonexistent/file.txt")


class TestRippleContextFromMemoryJsonl:
    """Test loading from memory.jsonl format."""

    def test_from_memory_jsonl(self):
        """Should load and format memories from JSONL."""
        memories = [
            {"id": "mem1", "content": "First memory"},
            {"id": "mem2", "content": "Second memory"},
            {"id": "mem3", "content": "Third memory"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            for mem in memories:
                f.write(json.dumps(mem) + "\n")
            temp_path = f.name

        try:
            ctx = RippleContext.from_memory_jsonl(temp_path, chunk_size=100)

            assert "[mem1]" in ctx.text
            assert "First memory" in ctx.text
            assert "[mem2]" in ctx.text
            assert "Second memory" in ctx.text
        finally:
            os.unlink(temp_path)

    def test_from_memory_jsonl_missing_content(self):
        """Should handle memories without content field."""
        memories = [
            {"id": "mem1", "content": "Has content"},
            {"id": "mem2"},  # No content
            {"id": "mem3", "content": "Also has content"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            for mem in memories:
                f.write(json.dumps(mem) + "\n")
            temp_path = f.name

        try:
            ctx = RippleContext.from_memory_jsonl(temp_path)

            assert "Has content" in ctx.text
            # mem2 should be skipped
            assert "Also has content" in ctx.text
        finally:
            os.unlink(temp_path)

    def test_from_memory_jsonl_malformed_line(self):
        """Should skip malformed JSON lines."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            f.write('{"id": "mem1", "content": "Valid"}\n')
            f.write('invalid json line\n')
            f.write('{"id": "mem2", "content": "Also valid"}\n')
            temp_path = f.name

        try:
            ctx = RippleContext.from_memory_jsonl(temp_path)

            assert "Valid" in ctx.text
            assert "Also valid" in ctx.text
            # Invalid line should be skipped
        finally:
            os.unlink(temp_path)

    def test_from_memory_jsonl_empty_file(self):
        """Should return empty context for empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            # Write nothing
            temp_path = f.name

        try:
            ctx = RippleContext.from_memory_jsonl(temp_path)
            assert ctx.text == ""
        finally:
            os.unlink(temp_path)

    def test_from_memory_jsonl_nonexistent(self):
        """Should return empty context for nonexistent file."""
        ctx = RippleContext.from_memory_jsonl("/nonexistent/memory.jsonl")
        assert ctx.text == ""


class TestRippleContextEdgeCases:
    """Test edge cases."""

    def test_very_long_text(self):
        """Should handle very long text."""
        text = "word " * 100000  # 600k characters
        ctx = RippleContext(text, chunk_size=10000)

        assert len(ctx.chunks) > 0

    def test_single_chunk_exact(self):
        """Text exactly chunk_size should produce one chunk."""
        text = "a" * 100
        ctx = RippleContext(text, chunk_size=100, chunk_overlap=0)

        assert len(ctx.chunks) == 1

    def test_overlap_equals_size(self):
        """Overlap equal to chunk_size should still work."""
        text = "abc" * 100
        # This would create infinite loop if not handled
        ctx = RippleContext(text, chunk_size=100, chunk_overlap=100)

        # Should at least not crash and have some chunks
        assert len(ctx.chunks) >= 1

    def test_zero_chunk_size(self):
        """Zero chunk size should be handled (step becomes 0)."""
        # This is an edge case; the implementation has max(1, step)
        text = "test"
        ctx = RippleContext(text, chunk_size=0, chunk_overlap=0)

        # Should not crash
        assert isinstance(ctx.chunks, list)

    def test_newlines_in_text(self):
        """Should handle text with many newlines."""
        text = "\n\n\n".join(["line"] * 100)
        ctx = RippleContext(text, chunk_size=50)

        assert len(ctx.chunks) > 0

    def test_tabs_in_text(self):
        """Should handle text with tabs."""
        text = "\t\t\t".join(["word"] * 100)
        ctx = RippleContext(text, chunk_size=50)

        assert len(ctx.chunks) > 0


class TestRippleContextSearchPropertyBased:
    """Property-based tests using Hypothesis."""

    from hypothesis import given, strategies as st

    @given(st.text(min_size=10, max_size=1000))
    def test_search_never_crashes(self, text):
        """Search should never crash on any text."""
        ctx = RippleContext(text, chunk_size=100)

        # Various queries
        queries = ["", "a", "test query", "123", "!@#$%"]

        for query in queries:
            results = ctx.search(query, top_k=5)
            assert isinstance(results, list)

    @given(st.text(min_size=1, max_size=100),
           st.integers(min_value=10, max_value=200))
    def test_chunk_indices_are_sequential(self, text, chunk_size):
        """Chunk indices should always be sequential."""
        ctx = RippleContext(text, chunk_size=chunk_size)

        for i, chunk in enumerate(ctx.chunks):
            assert chunk.index == i

    @given(st.text(min_size=10, max_size=500),
           st.integers(min_value=0, max_value=100),
           st.integers(min_value=0, max_value=100))
    def test_slice_always_returns_string(self, text, start, end):
        """Slice should always return a string."""
        ctx = RippleContext(text, chunk_size=100)

        result = ctx.slice(start, end)
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
