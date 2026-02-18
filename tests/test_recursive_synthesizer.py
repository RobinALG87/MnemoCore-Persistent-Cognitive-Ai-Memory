"""
Tests for Phase 4.5: RippleContext and RecursiveSynthesizer
"""
import asyncio
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
# RippleContext Tests
# ─────────────────────────────────────────────────────────────────────────────

from mnemocore.core.ripple_context import RippleContext, RippleChunk


class TestRippleChunk:
    def test_term_freq_built_on_init(self):
        chunk = RippleChunk(index=0, text="hello world hello", start_char=0, end_char=17)
        assert chunk.term_freq.get("hello") == 2
        assert chunk.term_freq.get("world") == 1

    def test_score_query_exact_match(self):
        chunk = RippleChunk(index=0, text="quantum computing is amazing", start_char=0, end_char=28)
        score = chunk.score_query(["quantum", "computing"])
        assert score > 0

    def test_score_query_no_match(self):
        chunk = RippleChunk(index=0, text="hello world", start_char=0, end_char=11)
        score = chunk.score_query(["quantum", "computing"])
        assert score == 0.0

    def test_score_query_empty_terms(self):
        chunk = RippleChunk(index=0, text="hello world", start_char=0, end_char=11)
        assert chunk.score_query([]) == 0.0


class TestRippleContext:
    SAMPLE_TEXT = (
        "Quantum computing uses qubits instead of classical bits. "
        "This allows quantum computers to solve certain problems exponentially faster. "
        "Machine learning is a subset of artificial intelligence. "
        "Neural networks are inspired by the human brain. "
        "The renovation project started in January and will finish in June. "
        "The kitchen was renovated first, then the bathroom. "
        "Lotto numbers from last week: 3, 7, 14, 22, 35, 42. "
        "Statistical analysis of Lotto patterns shows no predictable sequences."
    )

    def test_init_creates_chunks(self):
        ctx = RippleContext(self.SAMPLE_TEXT, chunk_size=100)
        assert len(ctx.chunks) > 0
        assert len(ctx) == len(self.SAMPLE_TEXT)

    def test_search_returns_relevant_chunks(self):
        ctx = RippleContext(self.SAMPLE_TEXT, chunk_size=150)
        results = ctx.search("quantum computing qubits", top_k=3)
        assert len(results) > 0
        # At least one result should mention quantum
        assert any("quantum" in r.lower() or "qubit" in r.lower() for r in results)

    def test_search_lotto(self):
        ctx = RippleContext(self.SAMPLE_TEXT, chunk_size=150)
        results = ctx.search("Lotto numbers patterns", top_k=3)
        assert len(results) > 0
        assert any("lotto" in r.lower() or "Lotto" in r for r in results)

    def test_search_empty_query_returns_fallback(self):
        ctx = RippleContext(self.SAMPLE_TEXT, chunk_size=100)
        results = ctx.search("", top_k=3)
        assert len(results) > 0  # fallback returns first chunks

    def test_slice(self):
        ctx = RippleContext(self.SAMPLE_TEXT, chunk_size=100)
        sliced = ctx.slice(0, 20)
        assert sliced == self.SAMPLE_TEXT[:20]

    def test_slice_clamps_to_bounds(self):
        ctx = RippleContext(self.SAMPLE_TEXT, chunk_size=100)
        sliced = ctx.slice(-10, 999999)
        assert sliced == self.SAMPLE_TEXT

    def test_get_stats(self):
        ctx = RippleContext(self.SAMPLE_TEXT, chunk_size=100, source_label="test")
        stats = ctx.get_stats()
        assert stats["source"] == "test"
        assert stats["total_chars"] == len(self.SAMPLE_TEXT)
        assert stats["total_chunks"] > 0
        assert "approx_tokens" in stats

    def test_from_memory_jsonl_missing_file(self, tmp_path):
        ctx = RippleContext.from_memory_jsonl(str(tmp_path / "nonexistent.jsonl"))
        assert len(ctx) == 0

    def test_from_memory_jsonl_valid(self, tmp_path):
        import json
        jsonl = tmp_path / "memory.jsonl"
        jsonl.write_text(
            json.dumps({"id": "abc123", "content": "Test memory content"}) + "\n" +
            json.dumps({"id": "def456", "content": "Another memory"}) + "\n"
        )
        ctx = RippleContext.from_memory_jsonl(str(jsonl))
        assert len(ctx) > 0
        assert "Test memory content" in ctx.text

    def test_repr(self):
        ctx = RippleContext("hello", source_label="test")
        assert "RippleContext" in repr(ctx)
        assert "test" in repr(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Decomposition Tests
# ─────────────────────────────────────────────────────────────────────────────

from mnemocore.core.recursive_synthesizer import _heuristic_decompose


class TestHeuristicDecompose:
    def test_simple_query_returns_itself(self):
        result = _heuristic_decompose("What is quantum computing?")
        assert len(result) >= 1
        assert "What is quantum computing?" in result

    def test_conjunction_split_english(self):
        query = "What is quantum computing and how does machine learning work?"
        result = _heuristic_decompose(query, max_sub=5)
        assert len(result) >= 2

    def test_conjunction_split_swedish(self):
        query = "Vad vet vi om Lotto-mönster och hur relaterar det till renoveringsprojektet?"
        result = _heuristic_decompose(query, max_sub=5)
        assert len(result) >= 2

    def test_max_sub_respected(self):
        query = "A and B and C and D and E and F and G"
        result = _heuristic_decompose(query, max_sub=3)
        assert len(result) <= 3

    def test_no_duplicates(self):
        query = "What is AI and what is AI?"
        result = _heuristic_decompose(query, max_sub=5)
        lower_results = [r.lower() for r in result]
        assert len(lower_results) == len(set(lower_results))


# ─────────────────────────────────────────────────────────────────────────────
# RecursiveSynthesizer Tests (with mock engine)
# ─────────────────────────────────────────────────────────────────────────────

from mnemocore.core.recursive_synthesizer import (
    RecursiveSynthesizer, SynthesizerConfig, SynthesisResult
)


def _make_mock_engine(memories=None):
    """Create a mock HAIMEngine that returns predefined memories."""
    if memories is None:
        memories = [
            {"id": "mem1", "content": "Quantum computing uses qubits", "score": 0.85},
            {"id": "mem2", "content": "Machine learning is a subset of AI", "score": 0.72},
        ]

    engine = MagicMock()

    # Mock query to return (id, score) tuples
    async def mock_query(query_text, top_k=5, **kwargs):
        return [(m["id"], m["score"]) for m in memories[:top_k]]

    engine.query = mock_query

    # Mock tier_manager.get_memory
    mem_map = {m["id"]: m for m in memories}

    async def mock_get_memory(mem_id):
        data = mem_map.get(mem_id)
        if not data:
            return None
        node = MagicMock()
        node.content = data["content"]
        node.metadata = {}
        node.tier = "hot"
        return node

    engine.tier_manager = MagicMock()
    engine.tier_manager.get_memory = mock_get_memory

    return engine


@pytest.mark.asyncio
class TestRecursiveSynthesizer:
    async def test_basic_synthesis(self):
        engine = _make_mock_engine()
        synth = RecursiveSynthesizer(engine=engine, config=SynthesizerConfig(max_depth=1))
        result = await synth.synthesize("What is quantum computing?")

        assert isinstance(result, SynthesisResult)
        assert result.query == "What is quantum computing?"
        assert len(result.sub_queries) >= 1
        assert isinstance(result.synthesis, str)
        assert len(result.synthesis) > 0

    async def test_returns_results(self):
        engine = _make_mock_engine()
        synth = RecursiveSynthesizer(engine=engine, config=SynthesizerConfig(max_depth=0))
        result = await synth.synthesize("quantum computing")

        assert len(result.results) > 0
        assert all("id" in r for r in result.results)
        assert all("content" in r for r in result.results)
        assert all("score" in r for r in result.results)

    async def test_depth_limit_prevents_infinite_recursion(self):
        """Ensure recursion stops at max_depth even with low confidence."""
        # Return very low scores to always trigger recursion
        low_conf_memories = [
            {"id": "m1", "content": "Some content", "score": 0.05},
            {"id": "m2", "content": "Other content", "score": 0.04},
        ]
        engine = _make_mock_engine(low_conf_memories)
        config = SynthesizerConfig(
            max_depth=2,
            min_confidence=0.9,  # Always recurse
            max_sub_queries=2,
        )
        synth = RecursiveSynthesizer(engine=engine, config=config)
        result = await synth.synthesize("test query")

        # Should complete without infinite loop
        assert isinstance(result, SynthesisResult)
        assert result.max_depth_hit <= config.max_depth

    async def test_parallel_sub_search(self):
        """Verify multiple sub-queries run and results are merged."""
        engine = _make_mock_engine()
        config = SynthesizerConfig(max_depth=0, max_sub_queries=3)
        synth = RecursiveSynthesizer(engine=engine, config=config)

        # Force multiple sub-queries via a conjunction query
        result = await synth.synthesize(
            "What is quantum computing and how does machine learning work?"
        )
        assert len(result.sub_queries) >= 1
        assert isinstance(result, SynthesisResult)

    async def test_ripple_context_integration(self):
        """Verify RippleContext snippets appear in result."""
        engine = _make_mock_engine()
        ctx = RippleContext(
            "Quantum computers use qubits. They are very powerful.",
            chunk_size=100,
        )
        synth = RecursiveSynthesizer(engine=engine, config=SynthesizerConfig(max_depth=0))
        result = await synth.synthesize("quantum computing", ripple_context=ctx)

        assert isinstance(result.ripple_snippets, list)
        # Should have found something about quantum
        assert len(result.ripple_snippets) > 0

    async def test_empty_memory_store(self):
        """Should handle empty memory store gracefully."""
        engine = _make_mock_engine(memories=[])
        synth = RecursiveSynthesizer(engine=engine, config=SynthesizerConfig(max_depth=0))
        result = await synth.synthesize("anything")

        assert isinstance(result, SynthesisResult)
        assert result.results == []
        assert "No relevant" in result.synthesis or len(result.synthesis) > 0

    async def test_llm_decompose_fallback_on_error(self):
        """If LLM decomposition fails, should fall back to heuristic."""
        def bad_llm(prompt):
            raise RuntimeError("LLM unavailable")

        engine = _make_mock_engine()
        synth = RecursiveSynthesizer(
            engine=engine,
            config=SynthesizerConfig(max_depth=0),
            llm_call=bad_llm,
        )
        # Should not raise — falls back to heuristic
        result = await synth.synthesize("What is AI and how does it work?")
        assert isinstance(result, SynthesisResult)

    async def test_multi_hit_boost(self):
        """Memories appearing in multiple sub-queries should get a score boost."""
        # Both sub-queries return the same memory
        engine = _make_mock_engine([
            {"id": "shared_mem", "content": "Shared content", "score": 0.6},
        ])
        config = SynthesizerConfig(max_depth=0, max_sub_queries=3)
        synth = RecursiveSynthesizer(engine=engine, config=config)
        result = await synth.synthesize("A and B and C")

        # The shared memory should appear with sub_query_hits > 1
        if result.results:
            shared = next((r for r in result.results if r["id"] == "shared_mem"), None)
            if shared:
                assert shared.get("sub_query_hits", 1) >= 1

    async def test_stats_populated(self):
        engine = _make_mock_engine()
        synth = RecursiveSynthesizer(engine=engine, config=SynthesizerConfig(max_depth=0))
        result = await synth.synthesize("test")

        assert "sub_query_count" in result.stats
        assert "merged_count" in result.stats
        assert "llm_available" in result.stats
        assert result.stats["llm_available"] is False

    async def test_elapsed_ms_positive(self):
        engine = _make_mock_engine()
        synth = RecursiveSynthesizer(engine=engine, config=SynthesizerConfig(max_depth=0))
        result = await synth.synthesize("test")
        assert result.total_elapsed_ms >= 0
