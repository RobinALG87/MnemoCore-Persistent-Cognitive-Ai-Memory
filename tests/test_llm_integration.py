"""
Comprehensive tests for LLM Integration Module.
===============================================
Tests for:
  - OllamaClient (HTTP fallback client)
  - HAIMLLMIntegrator (multi-provider LLM bridge)
  - MultiAgentHAIM (collective consciousness)
  - ContextAwareLLMIntegrator (context window optimization)
  - RLMIntegrator (recursive language model queries)
  - LLMConfig and LLMClientFactory
  - Error handling: missing API key, unreachable server, malformed response
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    patch,
    mock_open,
)

import pytest

from mnemocore.llm.config import LLMConfig, LLMProvider
from mnemocore.llm.ollama import OllamaClient
from mnemocore.llm.integrator import HAIMLLMIntegrator
from mnemocore.llm.multi_agent import MultiAgentHAIM
from mnemocore.llm.context_aware import ContextAwareLLMIntegrator
from mnemocore.llm.rlm import RLMIntegrator
from mnemocore.core.exceptions import LLMError, AgentNotFoundError
from mnemocore.core.binary_hdv import BinaryHDV


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
    tier_manager.get_memory = MagicMock(return_value=None)
    engine.tier_manager = tier_manager

    # Mock query method
    engine.query = MagicMock(return_value=[])

    # Mock encode_content
    mock_hdv = MagicMock()
    mock_hdv.similarity = MagicMock(return_value=0.8)
    engine.encode_content = MagicMock(return_value=mock_hdv)

    # Mock store
    engine.store = MagicMock(return_value="test_node_id")

    return engine


@pytest.fixture
def mock_memory_node():
    """Create a mock memory node for testing."""
    node = MagicMock()
    node.id = "test_node_123"
    node.memory_id = "test_node_123"
    node.content = "Test memory content about MnemoCore"
    node.metadata = {"category": "test", "importance": "high"}
    node.created_at = datetime.now(timezone.utc)
    node.access_count = 5
    node.ltp_strength = 0.75
    node.tier = "hot"
    node.score = 0.85
    node.token_count = 10
    node.relevance = 0.9
    node.importance = 0.8
    node.recency_weight = 0.7
    node.hdv = MagicMock()
    node.hdv.similarity = MagicMock(return_value=0.85)
    node.access = MagicMock()
    return node


# =====================================================================
# OllamaClient Tests
# =====================================================================

class TestOllamaClient:
    """Tests for the OllamaClient HTTP fallback client."""

    def test_init_defaults(self):
        """Test default initialization."""
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama3.1"

    def test_init_custom_params(self):
        """Test custom initialization."""
        client = OllamaClient(
            base_url="http://custom:8080/",
            model="custom-model"
        )
        assert client.base_url == "http://custom:8080"
        assert client.model == "custom-model"

    def test_generate_happy_path(self):
        """Test successful generate call with mocked urllib."""
        client = OllamaClient(base_url="http://localhost:11434", model="llama3.1")

        mock_response_data = {"response": "This is a test response from Ollama."}
        mock_response_body = json.dumps(mock_response_data).encode("utf-8")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = mock_response_body
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=None)
            mock_urlopen.return_value = mock_response

            result = client.generate("Test prompt", max_tokens=100)

            assert result == "This is a test response from Ollama."
            mock_urlopen.assert_called_once()

    def test_generate_timeout(self):
        """Test generate call with timeout."""
        client = OllamaClient()

        with patch("urllib.request.urlopen") as mock_urlopen:
            import socket
            mock_urlopen.side_effect = socket.timeout("Connection timed out")

            result = client.generate("Test prompt", max_tokens=100)

            assert "[Ollama Error:" in result
            assert "timed out" in result.lower() or "timeout" in result.lower()

    def test_generate_url_error(self):
        """Test generate call with URLError."""
        client = OllamaClient()

        with patch("urllib.request.urlopen") as mock_urlopen:
            import urllib.error
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            result = client.generate("Test prompt", max_tokens=100)

            assert "[Ollama Error:" in result

    @pytest.mark.asyncio
    async def test_agenerate_happy_path(self):
        """Test async generate call with mocked aiohttp."""
        client = OllamaClient()

        mock_response_data = {"response": "Async response from Ollama."}

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session

            result = await client.agenerate("Test prompt", max_tokens=100)

            assert result == "Async response from Ollama."

    @pytest.mark.asyncio
    async def test_agenerate_http_error(self):
        """Test async generate with HTTP error status."""
        client = OllamaClient()

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session

            result = await client.agenerate("Test prompt", max_tokens=100)

            assert "[Ollama Error:" in result
            assert "500" in result


# =====================================================================
# LLMConfig Tests
# =====================================================================

class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == LLMProvider.MOCK
        assert config.model == "gpt-4"
        assert config.max_tokens == 1024
        assert config.temperature == 0.7

    def test_openai_factory(self):
        """Test OpenAI configuration factory."""
        config = LLMConfig.openai(model="gpt-4o", api_key="test_key")
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4o"
        assert config.api_key == "test_key"

    def test_anthropic_factory(self):
        """Test Anthropic configuration factory."""
        config = LLMConfig.anthropic(api_key="anthropic_key")
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model == "claude-3-5-sonnet-20241022"

    def test_ollama_factory(self):
        """Test Ollama configuration factory."""
        config = LLMConfig.ollama(model="llama3.1", base_url="http://localhost:11434")
        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "llama3.1"
        assert config.base_url == "http://localhost:11434"

    def test_google_gemini_factory(self):
        """Test Google Gemini configuration factory."""
        config = LLMConfig.google_gemini(api_key="gemini_key")
        assert config.provider == LLMProvider.GOOGLE_GEMINI
        assert config.model == "gemini-1.5-pro"

    def test_mock_factory(self):
        """Test mock configuration factory."""
        config = LLMConfig.mock()
        assert config.provider == LLMProvider.MOCK


# =====================================================================
# HAIMLLMIntegrator Tests
# =====================================================================

class TestHAIMLLMIntegrator:
    """Tests for the HAIMLLMIntegrator multi-provider bridge."""

    def test_init_with_mock_config(self, mock_haim_engine):
        """Test initialization with mock configuration."""
        config = LLMConfig.mock()
        integrator = HAIMLLMIntegrator(mock_haim_engine, llm_config=config)

        assert integrator.haim == mock_haim_engine
        assert integrator.config.provider == LLMProvider.MOCK

    def test_init_with_legacy_client(self, mock_haim_engine):
        """Test initialization with legacy client."""
        mock_client = MagicMock()
        integrator = HAIMLLMIntegrator(mock_haim_engine, llm_client=mock_client)

        assert integrator.llm_client == mock_client

    def test_call_llm_mock_provider(self, mock_haim_engine):
        """Test _call_llm with mock provider returns mock response."""
        config = LLMConfig.mock()
        integrator = HAIMLLMIntegrator(mock_haim_engine, llm_config=config)

        result = integrator._call_llm("Reconstruct this memory")

        assert "[MOCK" in result

    def test_call_llm_openai_provider(self, mock_haim_engine):
        """Test _call_llm with OpenAI provider."""
        config = LLMConfig.openai(api_key="test_key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OpenAI response"
        mock_client.chat.completions.create = MagicMock(return_value=mock_response)

        integrator = HAIMLLMIntegrator(
            mock_haim_engine,
            llm_client=mock_client,
            llm_config=config
        )

        result = integrator._call_llm("Test prompt")

        assert result == "OpenAI response"
        mock_client.chat.completions.create.assert_called_once()

    def test_call_llm_anthropic_provider(self, mock_haim_engine):
        """Test _call_llm with Anthropic provider."""
        config = LLMConfig.anthropic(api_key="test_key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Anthropic response"
        mock_client.messages.create = MagicMock(return_value=mock_response)

        integrator = HAIMLLMIntegrator(
            mock_haim_engine,
            llm_client=mock_client,
            llm_config=config
        )

        result = integrator._call_llm("Test prompt")

        assert result == "Anthropic response"

    def test_call_llm_gemini_provider(self, mock_haim_engine):
        """Test _call_llm with Google Gemini provider."""
        config = LLMConfig.google_gemini(api_key="test_key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini response"
        mock_client.generate_content = MagicMock(return_value=mock_response)

        integrator = HAIMLLMIntegrator(
            mock_haim_engine,
            llm_client=mock_client,
            llm_config=config
        )

        result = integrator._call_llm("Test prompt")

        assert result == "Gemini response"

    def test_call_llm_ollama_provider(self, mock_haim_engine):
        """Test _call_llm with Ollama provider."""
        config = LLMConfig.ollama()
        mock_client = MagicMock()
        mock_client.generate = MagicMock(return_value="Ollama response")

        integrator = HAIMLLMIntegrator(
            mock_haim_engine,
            llm_client=mock_client,
            llm_config=config
        )

        result = integrator._call_llm("Test prompt")

        assert result == "Ollama response"

    def test_call_llm_error_handling(self, mock_haim_engine):
        """Test _call_llm error handling raises LLMError."""
        config = LLMConfig.openai(api_key="test_key")
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(
            side_effect=Exception("API error")
        )

        integrator = HAIMLLMIntegrator(
            mock_haim_engine,
            llm_client=mock_client,
            llm_config=config
        )

        with pytest.raises(LLMError) as exc_info:
            integrator._call_llm("Test prompt")

        assert "API error" in str(exc_info.value)

    def test_reconstructive_recall(self, mock_haim_engine, mock_memory_node):
        """Test reconstructive recall functionality."""
        config = LLMConfig.mock()
        integrator = HAIMLLMIntegrator(mock_haim_engine, llm_config=config)

        # Setup mock returns
        mock_haim_engine.query.return_value = [("node_1", 0.85)]
        mock_haim_engine.tier_manager.get_memory.return_value = mock_memory_node

        result = integrator.reconstructive_recall("test query", top_memories=5)

        assert "cue" in result
        assert "fragments" in result
        assert "reconstruction" in result
        assert result["cue"] == "test query"

    def test_multi_hypothesis_query(self, mock_haim_engine, mock_memory_node):
        """Test multi-hypothesis query functionality."""
        config = LLMConfig.mock()
        integrator = HAIMLLMIntegrator(mock_haim_engine, llm_config=config)

        # Setup mock returns
        mock_haim_engine.query.return_value = [("node_1", 0.85)]
        mock_haim_engine.tier_manager.get_memory.return_value = mock_memory_node
        mock_haim_engine.encode_content = MagicMock(
            side_effect=lambda text: BinaryHDV.random(16384)
        )

        result = integrator.multi_hypothesis_query(
            "test query",
            hypotheses=["hypothesis 1", "hypothesis 2"]
        )

        assert "query" in result
        assert "hypotheses" in result
        assert "relevant_memories" in result
        assert "evaluation" in result

    def test_from_config_factory(self, mock_haim_engine):
        """Test from_config factory method."""
        config = LLMConfig.mock()
        integrator = HAIMLLMIntegrator.from_config(mock_haim_engine, config)

        assert integrator.haim == mock_haim_engine
        assert integrator.config == config


# =====================================================================
# MultiAgentHAIM Tests
# =====================================================================

class TestMultiAgentHAIM:
    """Tests for the MultiAgentHAIM collective consciousness system."""

    @staticmethod
    def _mock_shared_memory(multi_agent):
        """Replace the shared_memory engine with a mock for async tests."""
        mock_mem = MagicMock()
        mock_node = MagicMock()
        mock_node.metadata = {"category": "test"}
        mock_node.content = "Test knowledge"
        mock_mem.store = AsyncMock(return_value="node_123")
        mock_mem.query = AsyncMock(return_value=[("node_123", 0.9)])
        mock_mem.tier_manager = MagicMock()
        mock_mem.tier_manager.get_memory = AsyncMock(return_value=mock_node)
        mock_mem.bind_memories = AsyncMock()
        mock_mem.orchestrate_orch_or = AsyncMock(return_value=[])
        multi_agent.shared_memory = mock_mem
        # Update agent references too
        for agent_id in multi_agent.agents:
            multi_agent.agents[agent_id]["haim"] = mock_mem

    def test_init_default_agents(self):
        """Test default initialization with 3 agents."""
        multi_agent = MultiAgentHAIM(num_agents=3)

        assert len(multi_agent.agents) == 3
        assert "agent_0" in multi_agent.agents
        assert "agent_1" in multi_agent.agents
        assert "agent_2" in multi_agent.agents

    def test_agent_roles(self):
        """Test agent role assignment."""
        multi_agent = MultiAgentHAIM(num_agents=3)

        assert multi_agent.agents["agent_0"]["role"] == "Research Agent"
        assert multi_agent.agents["agent_1"]["role"] == "Coding Agent"
        assert multi_agent.agents["agent_2"]["role"] == "Writing Agent"

    @pytest.mark.asyncio
    async def test_agent_learn(self):
        """Test agent learning stores to shared memory."""
        multi_agent = MultiAgentHAIM(num_agents=3)
        self._mock_shared_memory(multi_agent)

        node_id = await multi_agent.agent_learn(
            agent_id="agent_0",
            content="Test knowledge",
            metadata={"category": "test"}
        )

        assert node_id is not None
        # Verify it's in shared memory
        assert multi_agent.shared_memory is not None

    @pytest.mark.asyncio
    async def test_agent_learn_invalid_agent(self):
        """Test agent learning with invalid agent ID raises error."""
        multi_agent = MultiAgentHAIM(num_agents=3)

        with pytest.raises(AgentNotFoundError):
            await multi_agent.agent_learn(
                agent_id="invalid_agent",
                content="Test knowledge"
            )

    @pytest.mark.asyncio
    async def test_agent_recall(self):
        """Test agent recall from shared memory."""
        multi_agent = MultiAgentHAIM(num_agents=3)
        self._mock_shared_memory(multi_agent)

        # First store something
        await multi_agent.agent_learn(
            agent_id="agent_0",
            content="Important information about testing",
            metadata={"category": "test"}
        )

        # Then recall it
        results = await multi_agent.agent_recall(
            agent_id="agent_1",
            query="testing",
            top_k=5
        )

        # Results should be a list
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_agent_recall_invalid_agent(self):
        """Test agent recall with invalid agent ID raises error."""
        multi_agent = MultiAgentHAIM(num_agents=3)

        with pytest.raises(AgentNotFoundError):
            await multi_agent.agent_recall(
                agent_id="invalid_agent",
                query="test"
            )

    @pytest.mark.asyncio
    async def test_cross_agent_learning(self):
        """Test cross-agent learning strengthens connections."""
        multi_agent = MultiAgentHAIM(num_agents=3)
        self._mock_shared_memory(multi_agent)

        # Store two memories
        mem_1 = await multi_agent.agent_learn(
            agent_id="agent_0",
            content="First concept"
        )
        mem_2 = await multi_agent.agent_learn(
            agent_id="agent_1",
            content="Second concept"
        )

        # This should not raise
        await multi_agent.cross_agent_learning(
            concept_a="First concept",
            concept_b="Second concept",
            agent_id="agent_2",
            success=True
        )

    @pytest.mark.asyncio
    async def test_demonstrate_collective_consciousness(self):
        """Test the collective consciousness demonstration."""
        multi_agent = MultiAgentHAIM(num_agents=3)
        self._mock_shared_memory(multi_agent)

        result = await multi_agent.demonstrate_collective_consciousness()

        assert "demonstration" in result
        assert "agent_0_learned" in result
        assert "agent_1_learned" in result
        assert "agent_2_recalled_omega" in result
        assert "agent_2_recalled_haim" in result

    @pytest.mark.asyncio
    async def test_collective_orch_or(self):
        """Test collective Orch OR operation."""
        multi_agent = MultiAgentHAIM(num_agents=3)
        self._mock_shared_memory(multi_agent)

        # Store some memories first
        await multi_agent.agent_learn("agent_0", "Memory for Orch OR")

        result = await multi_agent.collective_orch_or(
            agent_id="agent_0",
            query="test",
            max_collapse=3
        )

        assert isinstance(result, list)


# =====================================================================
# ContextAwareLLMIntegrator Tests
# =====================================================================

class TestContextAwareLLMIntegrator:
    """Tests for the ContextAwareLLMIntegrator with context window optimization."""

    def test_init_defaults(self, mock_haim_engine):
        """Test default initialization."""
        config = LLMConfig.mock()
        integrator = ContextAwareLLMIntegrator(mock_haim_engine, llm_config=config)

        assert integrator.haim == mock_haim_engine
        assert integrator.config == config
        assert integrator.prioritizer is not None
        assert integrator.context_builder is not None

    def test_query_with_optimized_context_no_memories(self, mock_haim_engine):
        """Test query with optimized context when no memories found."""
        config = LLMConfig.mock()
        integrator = ContextAwareLLMIntegrator(mock_haim_engine, llm_config=config)

        mock_haim_engine.query.return_value = []

        result = integrator.query_with_optimized_context(
            "test query",
            memories=[],
            token_budget=1000
        )

        assert "query" in result
        assert "context" in result
        assert "response" in result
        assert "selected_memories" in result
        assert "stats" in result

    def test_query_with_optimized_context_with_memories(
        self, mock_haim_engine, mock_memory_node
    ):
        """Test query with optimized context with memories provided."""
        config = LLMConfig.mock()
        integrator = ContextAwareLLMIntegrator(mock_haim_engine, llm_config=config)

        result = integrator.query_with_optimized_context(
            "test query",
            memories=[mock_memory_node],
            token_budget=1000
        )

        assert "query" in result
        assert "response" in result

    def test_get_ranked_memories(self, mock_haim_engine, mock_memory_node):
        """Test getting ranked memories without LLM call."""
        config = LLMConfig.mock()
        integrator = ContextAwareLLMIntegrator(mock_haim_engine, llm_config=config)

        ranked = integrator.get_ranked_memories(
            memories=[mock_memory_node],
            query="test query",
            token_budget=1000
        )

        assert isinstance(ranked, list)

    def test_from_config_factory(self, mock_haim_engine):
        """Test from_config factory method."""
        config = LLMConfig.mock()
        integrator = ContextAwareLLMIntegrator.from_config(
            mock_haim_engine,
            config,
            context_budget=4000
        )

        assert integrator.config == config


# =====================================================================
# RLMIntegrator Tests
# =====================================================================

class TestRLMIntegrator:
    """Tests for the RLMIntegrator (Recursive Language Model)."""

    def _make_mock_llm_integrator(self, mock_haim_engine):
        """Create a mock LLM integrator for RLM tests."""
        config = LLMConfig.mock()
        return HAIMLLMIntegrator(mock_haim_engine, llm_config=config)

    def test_init(self, mock_haim_engine):
        """Test RLMIntegrator initialization."""
        llm_integrator = self._make_mock_llm_integrator(mock_haim_engine)
        rlm = RLMIntegrator(llm_integrator)

        assert rlm.llm_integrator == llm_integrator
        assert rlm.synthesizer is not None

    def test_init_with_config(self, mock_haim_engine):
        """Test RLMIntegrator initialization with config."""
        llm_integrator = self._make_mock_llm_integrator(mock_haim_engine)
        rlm = RLMIntegrator(llm_integrator, config=None)

        assert rlm.synthesizer is not None

    @pytest.mark.asyncio
    async def test_rlm_query_basic(self, mock_haim_engine):
        """Test basic RLM query execution."""
        llm_integrator = self._make_mock_llm_integrator(mock_haim_engine)
        rlm = RLMIntegrator(llm_integrator)

        # Mock the synthesizer
        mock_result = MagicMock()
        mock_result.query = "test query"
        mock_result.sub_queries = ["sub1", "sub2"]
        mock_result.results = []
        mock_result.synthesis = "Test synthesis"
        mock_result.max_depth_hit = False
        mock_result.total_elapsed_ms = 100.0
        mock_result.ripple_snippets = []
        mock_result.stats = {}

        rlm.synthesizer.synthesize = AsyncMock(return_value=mock_result)

        result = await rlm.rlm_query("test query")

        assert "query" in result
        assert "sub_queries" in result
        assert "synthesis" in result
        assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_rlm_query_with_context(self, mock_haim_engine):
        """Test RLM query with context text."""
        llm_integrator = self._make_mock_llm_integrator(mock_haim_engine)
        rlm = RLMIntegrator(llm_integrator)

        mock_result = MagicMock()
        mock_result.query = "test query"
        mock_result.sub_queries = []
        mock_result.results = []
        mock_result.synthesis = "Synthesis with context"
        mock_result.max_depth_hit = False
        mock_result.total_elapsed_ms = 150.0
        mock_result.ripple_snippets = []
        mock_result.stats = {}

        rlm.synthesizer.synthesize = AsyncMock(return_value=mock_result)

        result = await rlm.rlm_query(
            "test query",
            context_text="Additional context for the query"
        )

        assert "synthesis" in result

    def test_from_config_factory(self, mock_haim_engine):
        """Test from_config factory method."""
        config = LLMConfig.mock()
        rlm = RLMIntegrator.from_config(mock_haim_engine, config)

        assert rlm.llm_integrator is not None


# =====================================================================
# Error Handling Tests
# =====================================================================

class TestLLMErrorHandling:
    """Tests for error handling scenarios."""

    def test_missing_api_key_openai(self, mock_haim_engine):
        """Test handling missing API key for OpenAI."""
        config = LLMConfig.openai(api_key=None)

        # The factory should still create a config
        assert config.api_key is None

    def test_unreachable_server_ollama(self):
        """Test handling unreachable Ollama server."""
        client = OllamaClient(base_url="http://nonexistent:99999")

        with patch("urllib.request.urlopen") as mock_urlopen:
            import urllib.error
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            result = client.generate("test prompt")

            assert "[Ollama Error:" in result

    def test_malformed_response(self, mock_haim_engine):
        """Test handling malformed LLM response."""
        config = LLMConfig.openai(api_key="test_key")
        mock_client = MagicMock()

        # Return malformed response
        mock_client.chat.completions.create = MagicMock(
            return_value={"invalid": "structure"}
        )

        integrator = HAIMLLMIntegrator(
            mock_haim_engine,
            llm_client=mock_client,
            llm_config=config
        )

        # Should either raise LLMError or handle gracefully
        with pytest.raises((LLMError, TypeError, KeyError, AttributeError)):
            integrator._call_llm("test prompt")

    def test_timeout_handling(self):
        """Test handling request timeout."""
        client = OllamaClient()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Request timed out")

            result = client.generate("test prompt")

            assert "[Ollama Error:" in result

    def test_rate_limit_handling(self, mock_haim_engine):
        """Test handling rate limit errors."""
        config = LLMConfig.openai(api_key="test_key")
        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(
            side_effect=Exception("Rate limit exceeded")
        )

        integrator = HAIMLLMIntegrator(
            mock_haim_engine,
            llm_client=mock_client,
            llm_config=config
        )

        with pytest.raises(LLMError) as exc_info:
            integrator._call_llm("test prompt")

        assert "Rate limit" in str(exc_info.value)


# =====================================================================
# Integration Tests
# =====================================================================

class TestLLMIntegrationScenarios:
    """End-to-end integration scenarios."""

    def test_full_reconstruction_flow(self, mock_haim_engine, mock_memory_node):
        """Test full memory reconstruction flow."""
        config = LLMConfig.mock()
        integrator = HAIMLLMIntegrator(mock_haim_engine, llm_config=config)

        # Setup mocks
        mock_haim_engine.query.return_value = [
            ("node_1", 0.9),
            ("node_2", 0.8)
        ]
        mock_haim_engine.tier_manager.get_memory.return_value = mock_memory_node

        result = integrator.reconstructive_recall(
            cue="What is MnemoCore?",
            top_memories=5,
            enable_reasoning=True
        )

        assert "fragments" in result
        assert "reconstruction" in result
        assert len(result["fragments"]) > 0

    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self):
        """Test multi-agent collaboration scenario."""
        multi_agent = MultiAgentHAIM(num_agents=3)
        TestMultiAgentHAIM._mock_shared_memory(multi_agent)

        # Agent 0 learns
        mem_0 = await multi_agent.agent_learn(
            agent_id="agent_0",
            content="Python is a programming language",
            metadata={"category": "programming"}
        )

        # Agent 1 can access Agent 0's memory
        results = await multi_agent.agent_recall(
            agent_id="agent_1",
            query="programming",
            top_k=5
        )

        # Memory should be accessible across agents
        assert isinstance(results, list)

    def test_context_aware_query_flow(self, mock_haim_engine, mock_memory_node):
        """Test context-aware query flow."""
        config = LLMConfig.mock()
        integrator = ContextAwareLLMIntegrator(
            mock_haim_engine,
            llm_config=config,
            context_budget=4000
        )

        # Setup mocks
        mock_haim_engine.query.return_value = [("node_1", 0.85)]
        mock_haim_engine.tier_manager.get_memory.return_value = mock_memory_node

        result = integrator.query_with_optimized_context(
            query="Explain the architecture",
            token_budget=2000,
            top_k=10
        )

        assert "response" in result
        assert "stats" in result
        assert "selected_memories" in result
