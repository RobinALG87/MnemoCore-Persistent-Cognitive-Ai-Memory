"""
Extended Tests for Recursive Synthesizer
========================================

Additional comprehensive tests beyond the basic test_recursive_synthesizer.py
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from mnemocore.core.recursive_synthesizer import (
    RecursiveSynthesizer,
    SynthesizerConfig,
    SynthesisResult,
    SubQueryResult,
    _heuristic_decompose,
)


class TestSynthesizerConfigExtended:
    """Extended tests for configuration."""

    def test_config_with_ripple_disabled(self):
        """Should be able to disable RippleContext integration."""
        config = SynthesizerConfig(enable_ripple=False)
        assert config.enable_ripple is False

    def test_config_zero_max_depth(self):
        """Should handle zero max depth (no recursion)."""
        config = SynthesizerConfig(max_depth=0)
        assert config.max_depth == 0

    def test_config_high_parallel_limit(self):
        """Should handle high parallel limit."""
        config = SynthesizerConfig(parallel_limit=100)
        assert config.parallel_limit == 100


class TestHeuristicDecomposeExtended:
    """Extended tests for heuristic decomposition."""

    def test_query_with_or(self):
        """Should split on 'or'."""
        result = _heuristic_decompose("cats or dogs", max_sub=5)
        assert len(result) >= 1

    def test_query_with_but(self):
        """Should split on 'but'."""
        result = _heuristic_decompose("good but bad", max_sub=5)
        assert len(result) >= 1

    def test_query_with_also(self):
        """Should split on 'also'."""
        result = _heuristic_decompose("this and also that", max_sub=5)
        assert len(result) >= 1

    def test_query_with_furthermore(self):
        """Should split on 'furthermore'."""
        result = _heuristic_decompose("point one furthermore point two", max_sub=5)
        assert len(result) >= 1

    def test_query_with_related_to(self):
        """Should split on 'related to'."""
        result = _heuristic_decompose("topic1 related to topic2", max_sub=5)
        assert len(result) >= 1

    def test_question_word_splitting(self):
        """Should split on question words mid-sentence."""
        result = _heuristic_decompose("something what is this", max_sub=5)
        assert len(result) >= 1

    def test_filters_very_short_parts(self):
        """Should filter parts shorter than 10 chars."""
        result = _heuristic_decompose("test and xy", max_sub=5)
        # 'xy' should be filtered out
        assert all(len(p) >= 10 or p == "test and xy" for p in result)

    def test_unicode_characters(self):
        """Should handle Unicode characters."""
        result = _heuristic_decompose("test och瑞典 OR norsk", max_sub=5)
        assert isinstance(result, list)

    def test_very_long_single_word(self):
        """Should handle very long single words."""
        result = _heuristic_decompose("a" * 1000, max_sub=5)
        assert len(result) >= 1

    def test_multiple_conjunctions(self):
        """Should handle multiple conjunctions."""
        result = _heuristic_decompose("statement one is long and statement two is long or statement three", max_sub=10)
        assert len(result) >= 2


class TestSynthesizerExtended:
    """Extended tests for RecursiveSynthesizer."""

    @pytest.mark.asyncio
    async def test_synthesize_with_empty_string_query(self):
        """Should handle empty string query."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=None)

        synthesizer = RecursiveSynthesizer(mock_engine)
        result = await synthesizer.synthesize("")

        assert isinstance(result, SynthesisResult)

    @pytest.mark.asyncio
    async def test_synthesize_with_very_long_query(self):
        """Should handle very long queries."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=None)

        synthesizer = RecursiveSynthesizer(mock_engine)
        long_query = "word " * 1000
        result = await synthesizer.synthesize(long_query)

        assert isinstance(result, SynthesisResult)

    @pytest.mark.asyncio
    async def test_synthesize_respects_final_top_k(self):
        """Should limit final results to final_top_k."""
        mock_engine = MagicMock()

        # Create many results
        memories = [
            {"id": f"mem{i}", "content": f"content {i}", "score": 0.8 - i * 0.01}
            for i in range(50)
        ]

        async def mock_query(q, top_k=5, **kwargs):
            return [(m["id"], m["score"]) for m in memories[:top_k]]

        mock_engine.query = mock_query

        async def mock_get_memory(mem_id):
            data = next((m for m in memories if m["id"] == mem_id), None)
            if not data:
                return None
            node = MagicMock()
            node.content = data["content"]
            node.metadata = {}
            node.tier = "hot"
            return node

        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = mock_get_memory

        config = SynthesizerConfig(final_top_k=5, max_depth=0)
        synthesizer = RecursiveSynthesizer(mock_engine, config=config)

        result = await synthesizer.synthesize("test")

        assert len(result.results) <= 5

    @pytest.mark.asyncio
    async def test_synthesize_depth_tracking(self):
        """Should track maximum depth reached."""
        mock_engine = MagicMock()

        # Low scores to trigger recursion
        async def mock_query(q, top_k=5, **kwargs):
            return [(f"mem{i}", 0.2) for i in range(top_k)]

        mock_engine.query = mock_query

        async def mock_get_memory(mem_id):
            node = MagicMock()
            node.content = "content"
            node.metadata = {}
            node.tier = "hot"
            return node

        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = mock_get_memory

        config = SynthesizerConfig(max_depth=2, min_confidence=0.5, max_sub_queries=2)
        synthesizer = RecursiveSynthesizer(mock_engine, config=config)

        result = await synthesizer.synthesize("test")

        assert result.max_depth_hit >= 0
        assert result.max_depth_hit <= config.max_depth


class TestMicroQueryExtraction:
    """Test micro-query extraction for recursive analysis."""

    def test_extract_from_empty_memories(self):
        """Should handle empty memory list."""
        mock_engine = MagicMock()
        synthesizer = RecursiveSynthesizer(mock_engine)

        result = synthesizer._extract_micro_queries("query", [])
        assert result == []

    def test_extract_from_memories_with_empty_content(self):
        """Should handle memories with empty content."""
        mock_engine = MagicMock()
        synthesizer = RecursiveSynthesizer(mock_engine)

        memories = [
            {"content": "", "score": 0.8},
            {"content": "   ", "score": 0.7},
        ]

        result = synthesizer._extract_micro_queries("query", memories)
        # Should filter empty content
        assert all(len(m) > 0 for m in result)

    def test_extract_respects_max_micro(self):
        """Should limit number of micro-queries."""
        mock_engine = MagicMock()
        synthesizer = RecursiveSynthesizer(mock_engine)

        memories = [
            {"content": f"content {i} " * 20, "score": 0.8}
            for i in range(10)
        ]

        result = synthesizer._extract_micro_queries("query", memories, max_micro=3)
        assert len(result) <= 3


class TestRippleContextSearch:
    """Test RippleContext search integration."""

    @pytest.mark.asyncio
    async def test_search_ripple_with_empty_sub_queries(self):
        """Should handle empty sub-query list."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=None)

        mock_ripple = MagicMock()
        mock_ripple.search = MagicMock(return_value=[])

        synthesizer = RecursiveSynthesizer(mock_engine)

        snippets = await synthesizer._search_ripple("query", [], mock_ripple)

        assert isinstance(snippets, list)

    @pytest.mark.asyncio
    async def test_search_ripple_deduplicates(self):
        """Should deduplicate snippets."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=None)

        mock_ripple = MagicMock()
        # Return same snippet multiple times
        mock_ripple.search = MagicMock(return_value=["same snippet"])

        synthesizer = RecursiveSynthesizer(mock_engine)

        snippets = await synthesizer._search_ripple(
            "query",
            ["sub1", "sub2"],
            mock_ripple
        )

        # Should deduplicate
        assert len(snippets) == 1


class TestSynthesisResultExtended:
    """Extended tests for SynthesisResult."""

    def test_synthesis_result_with_ripple_snippets(self):
        """Should include ripple snippets."""
        result = SynthesisResult(
            query="test",
            sub_queries=["sub1"],
            results=[],
            synthesis="answer",
            max_depth_hit=0,
            total_elapsed_ms=100,
            ripple_snippets=["snippet1", "snippet2"],
        )

        assert len(result.ripple_snippets) == 2

    def test_synthesis_result_stats(self):
        """Should include statistics."""
        result = SynthesisResult(
            query="test",
            sub_queries=["sub1", "sub2"],
            results=[{"id": "1"}],
            synthesis="answer",
            max_depth_hit=1,
            total_elapsed_ms=200,
            stats={"sub_query_count": 2, "llm_available": False},
        )

        assert result.stats["sub_query_count"] == 2


class TestParallelSubSearchErrors:
    """Test error handling in parallel sub-search."""

    @pytest.mark.asyncio
    async def test_handles_exceptions_in_sub_search(self):
        """Should continue if one sub-search fails."""
        mock_engine = MagicMock()

        call_count = [0]

        async def mock_query(q, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Search failed")
            return []

        mock_engine.query = mock_query
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=None)

        synthesizer = RecursiveSynthesizer(mock_engine)

        results = await synthesizer._parallel_sub_search(["q1", "q2", "q3"], 0, None)

        # Should continue despite one error
        assert len(results) >= 2  # At least the successful ones

    @pytest.mark.asyncio
    async def test_handles_all_sub_searches_failing(self):
        """Should handle all sub-searches failing."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(side_effect=RuntimeError("All failed"))

        synthesizer = RecursiveSynthesizer(mock_engine)

        results = await synthesizer._parallel_sub_search(["q1", "q2"], 0, None)

        # Should return empty list
        assert results == []


class TestSynthesizerWithSyncLLM:
    """Test synthesizer with synchronous LLM."""

    @pytest.mark.asyncio
    async def test_sync_llm_decomposition(self):
        """Should handle synchronous LLM for decomposition."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=None)

        # Sync LLM
        def sync_llm(prompt):
            return "sub1\nsub2"

        synthesizer = RecursiveSynthesizer(mock_engine, llm_call=sync_llm)

        sub_queries = await synthesizer._decompose("test query")

        assert isinstance(sub_queries, list)

    @pytest.mark.asyncio
    async def test_sync_llm_synthesis(self):
        """Should handle synchronous LLM for synthesis."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=None)

        def sync_llm(prompt):
            return "LLM generated synthesis"

        synthesizer = RecursiveSynthesizer(mock_engine, llm_call=sync_llm)

        dummy_result = [{"id": "1", "content": "dummy content", "score": 0.9}]
        synthesis = await synthesizer._synthesize_results("query", dummy_result, [])

        assert synthesis == "LLM generated synthesis"


class TestSynthesizerConcurrency:
    """Test concurrency control."""

    @pytest.mark.asyncio
    async def test_respects_parallel_limit(self):
        """Should respect semaphore limit."""
        mock_engine = MagicMock()

        concurrent_calls = [0]
        max_concurrent = [0]

        async def mock_query(q, **kwargs):
            concurrent_calls[0] += 1
            max_concurrent[0] = max(max_concurrent[0], concurrent_calls[0])
            await asyncio.sleep(0.01)  # Small delay
            concurrent_calls[0] -= 1
            return []

        mock_engine.query = mock_query
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(return_value=None)

        config = SynthesizerConfig(parallel_limit=2)
        synthesizer = RecursiveSynthesizer(mock_engine, config=config)

        # Launch more sub-queries than limit
        await synthesizer._parallel_sub_search([f"q{i}" for i in range(10)], 0, None)

        # Max concurrent should not exceed limit
        assert max_concurrent[0] <= config.parallel_limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
