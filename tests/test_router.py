"""
Comprehensive Tests for Cognitive Router Module
==============================================

Tests the System 1 / System 2 cognitive routing including:
- Complexity assessment
- System 1 reflex routing
- System 2 reasoning routing
- Integration with HAIMEngine
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mnemocore.core.router import CognitiveRouter
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.exceptions import MnemoCoreError


class TestCognitiveRouterInitialization:
    """Test CognitiveRouter initialization."""

    def test_initialization(self):
        """Router should initialize with engine."""
        mock_engine = MagicMock()
        router = CognitiveRouter(mock_engine)
        assert router.engine == mock_engine
        assert router.complexity_threshold == 0.6

    def test_custom_threshold(self):
        """Should accept custom complexity threshold."""
        mock_engine = MagicMock()
        router = CognitiveRouter(mock_engine, complexity_threshold=0.7)
        assert router.complexity_threshold == 0.7


class TestComplexityAssessment:
    """Test complexity assessment logic."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Empty query should have low complexity."""
        mock_engine = MagicMock()
        router = CognitiveRouter(mock_engine)
        complexity = await router._assess_complexity("")
        assert 0.0 <= complexity <= 1.0

    @pytest.mark.asyncio
    async def test_short_query(self):
        """Short query should have low complexity."""
        mock_engine = MagicMock()
        router = CognitiveRouter(mock_engine)
        complexity = await router._assess_complexity("test")
        assert complexity < 0.5  # Should be below threshold

    @pytest.mark.asyncio
    async def test_length_heuristic(self):
        """Long queries should get complexity boost."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        # Short query
        short = await router._assess_complexity("short query")

        # Long query (20+ words)
        long_query = "word " * 25
        long = await router._assess_complexity(long_query)

        assert long > short  # Long should have higher complexity

    @pytest.mark.asyncio
    async def test_complex_markers(self):
        """Queries with complex markers should get boost."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        # Simple query
        simple = await router._assess_complexity("what is this")

        # Query with complex marker
        complex_query = "analyze the relationship between these concepts"
        complex_score = await router._assess_complexity(complex_query)

        assert complex_score > simple

    @pytest.mark.asyncio
    async def test_uncertainty_markers(self):
        """Uncertainty markers should increase complexity."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        # Certain query
        certain = await router._assess_complexity("I know this fact")

        # Uncertain query
        uncertain = await router._assess_complexity("maybe this is unknown")
        assert uncertain > certain

    @pytest.mark.asyncio
    async def test_familiarity_reduction(self):
        """Familiar topics should have lower complexity."""
        mock_engine = MagicMock()
        # High similarity result = familiar
        mock_engine.query = AsyncMock(return_value=[("node1", 0.9)])
        router = CognitiveRouter(mock_engine)

        familiar = await router._assess_complexity("well known topic")

        # Reset mock for unfamiliar
        mock_engine.query = AsyncMock(return_value=[])
        unfamiliar = await router._assess_complexity("well known topic")

        assert unfamiliar > familiar

    @pytest.mark.asyncio
    async def test_novelty_boost(self):
        """Novel (no results) queries should get complexity boost."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        novelty = await router._assess_complexity("unknown topic")
        # Should get boost for having no results
        assert novelty > 0.0

    @pytest.mark.asyncio
    async def test_complexity_clamped(self):
        """Complexity should always be in [0, 1]."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        # Very long query with all markers
        long_query = " ".join(["analyze"] * 100) + " maybe unknown"
        complexity = await router._assess_complexity(long_query)

        assert 0.0 <= complexity <= 1.0

    @pytest.mark.asyncio
    async def test_handles_mnemo_core_error(self):
        """Should handle domain errors gracefully."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(side_effect=MnemoCoreError("test error"))
        router = CognitiveRouter(mock_engine)

        # Should not raise, just log and continue
        complexity = await router._assess_complexity("test query")
        assert 0.0 <= complexity <= 1.0

    @pytest.mark.asyncio
    async def test_handles_generic_exception(self):
        """Should handle unexpected errors gracefully."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(side_effect=RuntimeError("unexpected"))
        router = CognitiveRouter(mock_engine)

        # Should not raise, just log and continue
        complexity = await router._assess_complexity("test query")
        assert 0.0 <= complexity <= 1.0


class TestSystem1Reflex:
    """Test System 1 (fast) reflex processing."""

    @pytest.mark.asyncio
    async def test_reflex_with_results(self):
        """Reflex should return quick answer from memory."""
        mock_engine = MagicMock()
        mock_node = MagicMock()
        mock_node.content = "Remembered fact"
        mock_engine.query = AsyncMock(return_value=[("node1", 0.9)])
        mock_engine.get_memory = AsyncMock(return_value=mock_node)

        router = CognitiveRouter(mock_engine)
        response = await router._system_1_reflex("quick question")

        assert "Reflex" in response
        assert "0.90" in response or "remembered" in response.lower()

    @pytest.mark.asyncio
    async def test_reflex_no_results(self):
        """Reflex should indicate no immediate match."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        mock_engine.get_memory = AsyncMock(return_value=None)

        router = CognitiveRouter(mock_engine)
        response = await router._system_1_reflex("unknown question")

        assert "don't have" in response.lower() or "no immediate" in response.lower()

    @pytest.mark.asyncio
    async def test_reflex_uses_top_result(self):
        """Reflex should use the top scoring result."""
        mock_engine = MagicMock()
        mock_node = MagicMock()
        mock_node.content = "Top result content"
        mock_engine.query = AsyncMock(return_value=[
            ("node3", 0.95),
            ("node2", 0.8),
            ("node1", 0.7),
        ])
        mock_engine.get_memory = AsyncMock(return_value=mock_node)

        router = CognitiveRouter(mock_engine)
        response = await router._system_1_reflex("question")

        assert "Top result content" in response

    @pytest.mark.asyncio
    async def test_reflex_handles_missing_node(self):
        """Reflex should handle when get_memory returns None."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[("node1", 0.9)])
        mock_engine.get_memory = AsyncMock(return_value=None)

        router = CognitiveRouter(mock_engine)
        response = await router._system_1_reflex("question")

        # Should not crash
        assert "Unknown" in response


class TestSystem2Reasoning:
    """Test System 2 (slow) reasoning processing."""

    @pytest.mark.asyncio
    async def test_reasoning_with_results(self):
        """Reasoning should analyze multiple results."""
        mock_engine = MagicMock()

        nodes = []
        for i in range(3):
            node = MagicMock()
            node.content = f"Memory content {i}"
            nodes.append(node)

        mock_engine.query = AsyncMock(return_value=[
            (f"node{i}", 0.9 - i * 0.1) for i in range(3)
        ])
        mock_engine.get_memory = AsyncMock(side_effect=nodes)

        router = CognitiveRouter(mock_engine)
        response = await router._system_2_reasoning("complex question", None)

        assert "Reasoning" in response
        assert "3 data points" in response or "3" in response

    @pytest.mark.asyncio
    async def test_reasoning_with_epistemic_drive(self):
        """Reasoning should include EIG when active."""
        mock_engine = MagicMock()
        mock_engine.epistemic_drive_active = True
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))

        mock_vec = BinaryHDV.random(1024)
        mock_engine._current_context_vector = AsyncMock(return_value=mock_vec)
        mock_engine.calculate_eig = MagicMock(return_value=0.75)

        nodes = []
        for i in range(2):
            node = MagicMock()
            node.content = f"Content {i}"
            nodes.append(node)

        mock_engine.query = AsyncMock(return_value=[
            (f"node{i}", 0.8) for i in range(2)
        ])
        mock_engine.get_memory = AsyncMock(side_effect=nodes)

        router = CognitiveRouter(mock_engine)
        response = await router._system_2_reasoning("question", None)

        assert "EIG" in response or "0.75" in response

    @pytest.mark.asyncio
    async def test_reasoning_with_working_memory(self):
        """Reasoning should use working memory context if provided."""
        mock_engine = MagicMock()
        mock_engine.epistemic_drive_active = True
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))
        mock_engine.calculate_eig = MagicMock(return_value=0.5)

        mock_engine.query = AsyncMock(return_value=[("node1", 0.8)])
        mock_node = MagicMock()
        mock_node.content = "Result"
        mock_engine.get_memory = AsyncMock(return_value=mock_node)

        router = CognitiveRouter(mock_engine)

        context = {
            "working_memory": ["memory item 1", "memory item 2"]
        }
        response = await router._system_2_reasoning("question", context)

        # Should not crash
        assert "Reasoning" in response

    @pytest.mark.asyncio
    async def test_reasoning_empty_results(self):
        """Reasoning should handle empty results."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])

        router = CognitiveRouter(mock_engine)
        response = await router._system_2_reasoning("question", None)

        assert "Reasoning" in response
        assert "0 data points" in response


class TestRouting:
    """Test main routing functionality."""

    @pytest.mark.asyncio
    async def test_route_to_system_1(self):
        """Low complexity should route to System 1."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[("node1", 0.9)])
        mock_node = MagicMock()
        mock_node.content = "Simple answer"
        mock_engine.get_memory = AsyncMock(return_value=mock_node)

        router = CognitiveRouter(mock_engine, complexity_threshold=0.8)
        response, debug = await router.route("simple question")

        assert debug["system"] == "Sys1 (Fast)"
        assert "Reflex" in response

    @pytest.mark.asyncio
    async def test_route_to_system_2(self):
        """High complexity should route to System 2."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))
        mock_engine._current_context_vector = AsyncMock(return_value=BinaryHDV.random(1024))

        router = CognitiveRouter(mock_engine, complexity_threshold=0.3)
        response, debug = await router.route("analyze this complex problem")

        assert debug["system"] == "Sys2 (Slow)"
        assert "Reasoning" in response

    @pytest.mark.asyncio
    async def test_route_includes_timing(self):
        """Route should include duration in debug info."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        response, debug = await router.route("test")

        assert "duration" in debug
        assert debug["duration"] >= 0
        assert isinstance(debug["duration"], float)

    @pytest.mark.asyncio
    async def test_route_includes_complexity(self):
        """Route should include complexity score in debug info."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        response, debug = await router.route("test")

        assert "complexity_score" in debug
        assert 0.0 <= debug["complexity_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_route_with_context(self):
        """Route should pass context to System 2."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))
        mock_engine._current_context_vector = AsyncMock(return_value=BinaryHDV.random(1024))

        router = CognitiveRouter(mock_engine, complexity_threshold=0.0)
        context = {"working_memory": ["item1"]}

        response, debug = await router.route("complex", context)

        assert debug["system"] == "Sys2 (Slow)"


class TestRouterEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_unicode_query(self):
        """Should handle Unicode characters."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        response, debug = await router.route("Vad är detta? 🚀")

        assert isinstance(response, str)
        assert isinstance(debug, dict)

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Should handle very long queries."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        long_query = "word " * 1000
        response, debug = await router.route(long_query)

        assert isinstance(response, str)
        assert debug["complexity_score"] >= 0.3  # Should be high due to length

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Should handle special characters."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        special_query = "test @#$%^&*() query"
        response, debug = await router.route(special_query)

        assert isinstance(response, str)


class TestRouterPropertyBased:
    """Property-based tests using Hypothesis."""

    from hypothesis import given, strategies as st

    @pytest.mark.asyncio
    @given(st.text(min_size=1, max_size=200))
    async def test_complexity_always_valid(self, query):
        """Complexity should always be in valid range."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        router = CognitiveRouter(mock_engine)

        complexity = await router._assess_complexity(query)
        assert 0.0 <= complexity <= 1.0

    @pytest.mark.asyncio
    @given(st.text(min_size=1, max_size=100))
    async def test_route_never_crashes(self, query):
        """Route should never crash on any input."""
        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value=[])
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))
        mock_engine._current_context_vector = AsyncMock(return_value=BinaryHDV.random(1024))

        router = CognitiveRouter(mock_engine)

        response, debug = await router.route(query)

        assert isinstance(response, str)
        assert isinstance(debug, dict)
        assert "system" in debug


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
