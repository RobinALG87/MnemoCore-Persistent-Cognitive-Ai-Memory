"""
Comprehensive Tests for Subconscious AI Worker (Phase 4.4)
===========================================================

Tests the background AI worker that performs memory sorting,
enhanced dreaming, and micro self-improvement.

Coverage:
- OllamaClient.generate() with mocked aiohttp.ClientSession
- LMStudioClient.generate() with same patterns
- APIClient.generate() for OpenAI and Anthropic formats
- ResourceGuard CPU and rate limit checks
- SubconsciousAIWorker lifecycle (start/stop/pulse)
- _init_model_client() factory for each provider
- dry_run mode behavior
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import time

from mnemocore.core.subconscious_ai import (
    OllamaClient,
    LMStudioClient,
    APIClient,
    ResourceGuard,
    SubconsciousAIWorker,
    SubconsciousCycleResult,
    Suggestion,
)
from mnemocore.core.config import SubconsciousAIConfig
from mnemocore.core.binary_hdv import BinaryHDV


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
    """Create a mock SubconsciousAIConfig for testing."""
    return SubconsciousAIConfig(
        enabled=True,
        beta_mode=True,
        model_provider="ollama",
        model_name="test-model",
        model_url="http://localhost:11434",
        pulse_interval_seconds=1,  # Short for testing
        pulse_backoff_enabled=False,
        max_cpu_percent=80.0,
        cycle_timeout_seconds=10,
        rate_limit_per_hour=100,
        memory_sorting_enabled=True,
        enhanced_dreaming_enabled=True,
        micro_self_improvement_enabled=True,
        dry_run=True,
        log_all_decisions=False,
        audit_trail_path=None,
        max_memories_per_cycle=5,
    )


@pytest.fixture
def mock_engine():
    """Create a mock HAIMEngine for testing."""
    engine = MagicMock()
    engine.tier_manager = MagicMock()
    engine.tier_manager.get_hot_recent = AsyncMock(return_value=[])
    engine.tier_manager.search = AsyncMock(return_value=[])
    engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=[])
    engine.get_stats = AsyncMock(return_value={
        "gap_detector": {"total_gaps": 0},
        "tiers": {"hot_count": 0},
    })
    engine.config = MagicMock()
    engine.config.tiers_hot = MagicMock()
    engine.config.tiers_hot.max_memories = 100
    engine.binary_encoder = MagicMock()
    engine.binary_encoder.encode = MagicMock(return_value=BinaryHDV.random(1024))
    engine.bind_memories = AsyncMock()
    engine.persist_memory_snapshot = AsyncMock()
    return engine


# =============================================================================
# OllamaClient Tests
# =============================================================================

class TestOllamaClient:
    """Test Ollama API client."""

    def test_ollama_client_init(self):
        """OllamaClient should initialize with correct URL."""
        client = OllamaClient(
            model_name="llama2",
            model_url="http://localhost:11434",
            timeout=30
        )
        assert client.model_name == "llama2"
        assert client.timeout == 30
        assert client._generate_url == "http://localhost:11434/api/generate"

    @pytest.mark.asyncio
    async def test_ollama_generate_happy_path(self):
        """OllamaClient.generate() should return response on success."""
        client = OllamaClient("llama2", "http://localhost:11434", timeout=30)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"response": "Hello, world!"})
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.ClientTimeout"):
                result = await client.generate("test prompt")

        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_ollama_generate_timeout(self):
        """OllamaClient.generate() should return empty string on timeout."""
        client = OllamaClient("llama2", "http://localhost:11434", timeout=30)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aiohttp.ClientTimeout"):
                result = await client.generate("test prompt")

        assert result == ""

    @pytest.mark.asyncio
    async def test_ollama_generate_connection_error(self):
        """OllamaClient.generate() should return empty string on connection error."""
        client = OllamaClient("llama2", "http://localhost:11434", timeout=30)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(side_effect=ConnectionError("Cannot connect"))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aiohttp.ClientTimeout"):
                result = await client.generate("test prompt")

        assert result == ""

    @pytest.mark.asyncio
    async def test_ollama_generate_http_500(self):
        """OllamaClient.generate() should return empty string on HTTP 500."""
        client = OllamaClient("llama2", "http://localhost:11434", timeout=30)

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.json = AsyncMock()

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.ClientTimeout"):
                result = await client.generate("test prompt")

        assert result == ""


# =============================================================================
# LMStudioClient Tests
# =============================================================================

class TestLMStudioClient:
    """Test LM Studio API client (OpenAI-compatible)."""

    def test_lm_studio_client_init(self):
        """LMStudioClient should initialize with correct URL."""
        client = LMStudioClient(
            model_name="local-model",
            model_url="http://localhost:1234",
            timeout=30
        )
        assert client.model_name == "local-model"
        assert client._chat_url == "http://localhost:1234/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_lm_studio_generate_happy_path(self):
        """LMStudioClient.generate() should return response on success."""
        client = LMStudioClient("local-model", "http://localhost:1234", timeout=30)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "AI response"}}]
        })
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.ClientTimeout"):
                result = await client.generate("test prompt")

        assert result == "AI response"

    @pytest.mark.asyncio
    async def test_lm_studio_generate_timeout(self):
        """LMStudioClient.generate() should return empty string on timeout."""
        client = LMStudioClient("local-model", "http://localhost:1234", timeout=30)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            with patch("aiohttp.ClientTimeout"):
                result = await client.generate("test prompt")

        assert result == ""

    @pytest.mark.asyncio
    async def test_lm_studio_generate_empty_choices(self):
        """LMStudioClient.generate() should return empty string if no choices."""
        client = LMStudioClient("local-model", "http://localhost:1234", timeout=30)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"choices": []})
        mock_response.text = AsyncMock(return_value="")

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.ClientTimeout"):
                result = await client.generate("test prompt")

        assert result == ""


# =============================================================================
# APIClient Tests
# =============================================================================

class TestAPIClient:
    """Test external API client (OpenAI/Anthropic)."""

    def test_api_client_init_openai(self):
        """APIClient should initialize for OpenAI provider."""
        client = APIClient(
            model_name="gpt-4",
            model_url="https://api.openai.com",
            api_key="test-key",
            provider="openai",
            timeout=30
        )
        assert client.provider == "openai"
        assert client.api_key == "test-key"

    def test_api_client_init_anthropic(self):
        """APIClient should initialize for Anthropic provider."""
        client = APIClient(
            model_name="claude-3",
            model_url="https://api.anthropic.com",
            api_key="anthropic-key",
            provider="anthropic",
            timeout=30
        )
        assert client.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_api_client_openai_format(self):
        """APIClient should use OpenAI format for openai provider."""
        client = APIClient("gpt-4", "https://api.openai.com", api_key="test-key", provider="openai")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "OpenAI response"}}]
        })
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        captured_payload = {}
        captured_headers = {}

        def mock_post(url, json=None, headers=None, timeout=None):
            captured_payload.update(json or {})
            captured_headers.update(headers or {})
            return mock_response

        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.ClientTimeout"):
                result = await client.generate("test prompt")

        assert result == "OpenAI response"
        assert "messages" in captured_payload
        assert captured_headers["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_api_client_anthropic_format(self):
        """APIClient should use Anthropic format for anthropic provider."""
        client = APIClient("claude-3", "https://api.anthropic.com", api_key="anthropic-key", provider="anthropic")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "content": [{"text": "Anthropic response"}]
        })
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        captured_headers = {}

        def mock_post(url, json=None, headers=None, timeout=None):
            captured_headers.update(headers or {})
            return mock_response

        mock_session = AsyncMock()
        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.ClientTimeout"):
                result = await client.generate("test prompt")

        assert result == "Anthropic response"
        assert captured_headers["x-api-key"] == "anthropic-key"
        assert captured_headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    async def test_api_client_missing_api_key(self):
        """APIClient should still work without API key (may fail at server)."""
        client = APIClient("gpt-4", "https://api.openai.com", api_key=None, provider="openai")

        mock_response = AsyncMock()
        mock_response.status = 401  # Unauthorized
        mock_response.text = AsyncMock(return_value="Unauthorized")
        mock_response.json = AsyncMock()

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.ClientTimeout"):
                result = await client.generate("test prompt")

        assert result == ""  # Returns empty on error

    @pytest.mark.asyncio
    async def test_api_client_unknown_provider(self):
        """APIClient should return empty for unknown provider."""
        client = APIClient("unknown", "https://api.unknown.com", provider="unknown")

        result = await client.generate("test prompt")
        assert result == ""


# =============================================================================
# ResourceGuard Tests
# =============================================================================

class TestResourceGuard:
    """Test resource monitoring and throttling."""

    def test_resource_guard_init(self):
        """ResourceGuard should initialize with correct thresholds."""
        guard = ResourceGuard(max_cpu_percent=50.0, rate_limit_per_hour=10)
        assert guard.max_cpu_percent == 50.0
        assert guard.rate_limit_per_hour == 10

    def test_check_cpu_below_threshold(self):
        """check_cpu should return True when CPU is below threshold."""
        pytest.importorskip("psutil")
        guard = ResourceGuard(max_cpu_percent=80.0, rate_limit_per_hour=10)

        with patch("psutil.cpu_percent", return_value=50.0):
            result = guard.check_cpu()

        assert result is True

    def test_check_cpu_above_threshold(self):
        """check_cpu should return False when CPU is above threshold."""
        pytest.importorskip("psutil")
        guard = ResourceGuard(max_cpu_percent=50.0, rate_limit_per_hour=10)

        with patch("psutil.cpu_percent", return_value=60.0):
            result = guard.check_cpu()

        assert result is False

    def test_check_cpu_no_psutil(self):
        """check_cpu should return True if psutil is not available."""
        guard = ResourceGuard(max_cpu_percent=50.0, rate_limit_per_hour=10)

        with patch.dict("sys.modules", {"psutil": None}):
            # Force ImportError
            with patch("builtins.__import__", side_effect=ImportError("No psutil")):
                result = guard.check_cpu()

        # Should allow when psutil not available
        assert result is True

    def test_check_rate_limit_not_exceeded(self):
        """check_rate_limit should return True when under limit."""
        guard = ResourceGuard(max_cpu_percent=80.0, rate_limit_per_hour=5)

        # Add 3 calls (under limit)
        for _ in range(3):
            guard.record_call()

        result = guard.check_rate_limit()
        assert result is True

    def test_check_rate_limit_exceeded(self):
        """check_rate_limit should return False when limit reached."""
        guard = ResourceGuard(max_cpu_percent=80.0, rate_limit_per_hour=5)

        # Add 5 calls (at limit)
        for _ in range(5):
            guard.record_call()

        result = guard.check_rate_limit()
        assert result is False

    def test_rate_limit_expires_old_calls(self):
        """Old calls should expire from rate limit after 1 hour."""
        guard = ResourceGuard(max_cpu_percent=80.0, rate_limit_per_hour=5)

        # Add 5 calls with old timestamps (2 hours ago)
        old_time = time.time() - 7200
        guard._call_history.extend([old_time] * 5)

        # New check should pass since old calls expired
        result = guard.check_rate_limit()
        assert result is True

    def test_record_error_increments_counter(self):
        """record_error should increment consecutive errors."""
        guard = ResourceGuard(max_cpu_percent=80.0, rate_limit_per_hour=10)

        guard.record_error()
        guard.record_error()
        guard.record_error()

        assert guard.consecutive_errors == 3

    def test_record_success_resets_errors(self):
        """record_success should reset consecutive errors."""
        guard = ResourceGuard(max_cpu_percent=80.0, rate_limit_per_hour=10)

        guard.record_error()
        guard.record_error()
        guard.record_success()

        assert guard.consecutive_errors == 0

    def test_get_backoff_seconds_no_errors(self):
        """Backoff should be base interval with no errors."""
        guard = ResourceGuard(max_cpu_percent=80.0, rate_limit_per_hour=10)

        backoff = guard.get_backoff_seconds(base_interval=30, max_backoff=300)
        assert backoff == 30

    def test_get_backoff_seconds_with_errors(self):
        """Backoff should increase exponentially with errors."""
        guard = ResourceGuard(max_cpu_percent=80.0, rate_limit_per_hour=10)

        guard.record_error()
        backoff1 = guard.get_backoff_seconds(base_interval=30, max_backoff=300)
        assert backoff1 == 60  # 30 * 2^1

        guard.record_error()
        backoff2 = guard.get_backoff_seconds(base_interval=30, max_backoff=300)
        assert backoff2 == 120  # 30 * 2^2

    def test_get_backoff_respects_max(self):
        """Backoff should not exceed max_backoff."""
        guard = ResourceGuard(max_cpu_percent=80.0, rate_limit_per_hour=10)

        # Many errors
        for _ in range(10):
            guard.record_error()

        backoff = guard.get_backoff_seconds(base_interval=30, max_backoff=100)
        assert backoff == 100


# =============================================================================
# SubconsciousAIWorker Lifecycle Tests
# =============================================================================

class TestSubconsciousAIWorkerLifecycle:
    """Test worker start/stop lifecycle."""

    def test_worker_init(self, mock_config, mock_engine):
        """Worker should initialize with correct configuration."""
        worker = SubconsciousAIWorker(mock_engine, mock_config)

        assert worker.cfg == mock_config
        assert worker.engine == mock_engine
        assert worker._running is False
        assert worker._model_client is not None

    def test_worker_init_creates_ollama_client(self, mock_config, mock_engine):
        """Worker should create OllamaClient for ollama provider."""
        object.__setattr__(mock_config, 'model_provider', 'ollama')
        worker = SubconsciousAIWorker(mock_engine, mock_config)

        assert isinstance(worker._model_client, OllamaClient)

    def test_worker_init_creates_lm_studio_client(self, mock_engine):
        """Worker should create LMStudioClient for lm_studio provider."""
        config = SubconsciousAIConfig(
            enabled=True,
            model_provider="lm_studio",
            model_url="http://localhost:1234",
        )
        worker = SubconsciousAIWorker(mock_engine, config)

        assert isinstance(worker._model_client, LMStudioClient)

    def test_worker_init_creates_api_client_openai(self, mock_engine):
        """Worker should create APIClient for openai_api provider."""
        config = SubconsciousAIConfig(
            enabled=True,
            model_provider="openai_api",
            api_key="test-key",
        )
        worker = SubconsciousAIWorker(mock_engine, config)

        assert isinstance(worker._model_client, APIClient)
        assert worker._model_client.provider == "openai_api"

    def test_worker_init_creates_api_client_anthropic(self, mock_engine):
        """Worker should create APIClient for anthropic_api provider."""
        config = SubconsciousAIConfig(
            enabled=True,
            model_provider="anthropic_api",
            api_key="test-key",
        )
        worker = SubconsciousAIWorker(mock_engine, config)

        assert isinstance(worker._model_client, APIClient)
        assert worker._model_client.provider == "anthropic_api"

    def test_worker_init_unknown_provider_defaults_to_ollama(self, mock_engine):
        """Worker should default to OllamaClient for unknown provider."""
        config = SubconsciousAIConfig(
            enabled=True,
            model_provider="unknown_provider",
        )
        worker = SubconsciousAIWorker(mock_engine, config)

        assert isinstance(worker._model_client, OllamaClient)

    @pytest.mark.asyncio
    async def test_worker_start(self, mock_config, mock_engine):
        """Worker should start successfully when enabled."""
        worker = SubconsciousAIWorker(mock_engine, mock_config)

        # Start and immediately stop
        start_task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.1)  # Let start complete

        assert worker._running is True
        assert worker._task is not None

        await worker.stop()

    @pytest.mark.asyncio
    async def test_worker_start_disabled(self, mock_engine):
        """Worker should not start when disabled."""
        config = SubconsciousAIConfig(enabled=False)
        worker = SubconsciousAIWorker(mock_engine, config)

        await worker.start()

        assert worker._running is False
        assert worker._task is None

    @pytest.mark.asyncio
    async def test_worker_stop(self, mock_config, mock_engine):
        """Worker should stop cleanly."""
        worker = SubconsciousAIWorker(mock_engine, mock_config)

        await worker.start()
        await asyncio.sleep(0.1)

        await worker.stop()

        assert worker._running is False
        assert worker._task is None

    @pytest.mark.asyncio
    async def test_worker_pulse_fires_at_interval(self, mock_config, mock_engine):
        """Worker should pulse at configured interval."""
        object.__setattr__(mock_config, 'pulse_interval_seconds', 0.1)  # 100ms for testing

        worker = SubconsciousAIWorker(mock_engine, mock_config)

        # Mock the cycle to track calls
        original_run_cycle = worker._run_cycle
        call_count = [0]

        async def counted_run_cycle():
            call_count[0] += 1
            return await original_run_cycle()

        worker._run_cycle = counted_run_cycle

        await worker.start()
        await asyncio.sleep(0.35)  # Should allow ~3 pulses
        await worker.stop()

        assert call_count[0] >= 2  # At least 2 pulses should have fired


# =============================================================================
# SubconsciousAIWorker Dry Run Tests
# =============================================================================

class TestSubconsciousAIWorkerDryRun:
    """Test dry_run mode behavior."""

    @pytest.mark.asyncio
    async def test_dry_run_does_not_apply_sorting(self, mock_engine):
        """In dry_run mode, sorting should not modify nodes."""
        config = SubconsciousAIConfig(
            enabled=True,
            dry_run=True,
            memory_sorting_enabled=True,
            enhanced_dreaming_enabled=False,
            micro_self_improvement_enabled=False,
            pulse_interval_seconds=1,
        )

        # Create a mock memory node
        from mnemocore.core.node import MemoryNode
        mock_node = MemoryNode(
            id="test-node-1",
            content="Test memory content",
            hdv=BinaryHDV.random(1024),
        )
        mock_engine.tier_manager.get_hot_recent = AsyncMock(return_value=[mock_node])

        worker = SubconsciousAIWorker(mock_engine, config)

        # Mock the model client to return valid JSON
        worker._model_client.generate = AsyncMock(return_value='{"categories": ["test"], "memory_tags": {"1": ["tag1"]}}')

        result = await worker._memory_sorting_cycle()

        assert result.dry_run is True
        # Tags should NOT be applied in dry_run mode
        assert "tags" not in mock_node.metadata

    @pytest.mark.asyncio
    async def test_dry_run_does_not_create_synapses(self, mock_engine):
        """In dry_run mode, dreaming should not create synapses."""
        config = SubconsciousAIConfig(
            enabled=True,
            dry_run=True,
            memory_sorting_enabled=False,
            enhanced_dreaming_enabled=True,
            micro_self_improvement_enabled=False,
            pulse_interval_seconds=1,
        )

        from mnemocore.core.node import MemoryNode
        mock_node = MemoryNode(
            id="test-node-1",
            content="Test memory for dreaming",
            hdv=BinaryHDV.random(1024),
            ltp_strength=0.3,
        )
        mock_engine.tier_manager.get_hot_recent = AsyncMock(return_value=[mock_node])

        worker = SubconsciousAIWorker(mock_engine, config)
        worker._model_client.generate = AsyncMock(return_value='{"bridges": {"1": ["concept1"]}}')

        result = await worker._enhanced_dreaming_cycle()

        assert result.dry_run is True
        # bind_memories should NOT have been called
        mock_engine.bind_memories.assert_not_called()


# =============================================================================
# SubconsciousAIWorker Stats Tests
# =============================================================================

class TestSubconsciousAIWorkerStats:
    """Test worker statistics reporting."""

    def test_stats_property(self, mock_config, mock_engine):
        """stats property should return correct statistics."""
        worker = SubconsciousAIWorker(mock_engine, mock_config)

        stats = worker.stats

        assert "enabled" in stats
        assert "running" in stats
        assert "dry_run" in stats
        assert "model_provider" in stats
        assert "model_name" in stats
        assert "total_cycles" in stats
        assert "successful_cycles" in stats
        assert "failed_cycles" in stats
        assert "operations" in stats

    def test_stats_tracks_cycles(self, mock_config, mock_engine):
        """stats should track cycle counts."""
        worker = SubconsciousAIWorker(mock_engine, mock_config)

        worker._total_cycles = 10
        worker._successful_cycles = 8
        worker._failed_cycles = 2

        stats = worker.stats

        assert stats["total_cycles"] == 10
        assert stats["successful_cycles"] == 8
        assert stats["failed_cycles"] == 2


# =============================================================================
# SubconsciousCycleResult Tests
# =============================================================================

class TestSubconsciousCycleResult:
    """Test SubconsciousCycleResult dataclass."""

    def test_cycle_result_creation(self):
        """SubconsciousCycleResult should be created with all fields."""
        result = SubconsciousCycleResult(
            timestamp="2024-01-01T00:00:00Z",
            operation="sorting",
            input_count=5,
            output={"parsed": {"categories": ["test"]}},
            elapsed_ms=150.5,
            model_used="test-model",
            dry_run=True,
        )

        assert result.operation == "sorting"
        assert result.input_count == 5
        assert result.elapsed_ms == 150.5
        assert result.error is None

    def test_cycle_result_with_error(self):
        """SubconsciousCycleResult should track errors."""
        result = SubconsciousCycleResult(
            timestamp="2024-01-01T00:00:00Z",
            operation="dreaming",
            input_count=0,
            output={},
            elapsed_ms=0.0,
            model_used="test-model",
            dry_run=True,
            error="Connection refused",
        )

        assert result.error == "Connection refused"


# =============================================================================
# Suggestion Tests
# =============================================================================

class TestSuggestion:
    """Test Suggestion dataclass."""

    def test_suggestion_creation(self):
        """Suggestion should be created with all fields."""
        suggestion = Suggestion(
            suggestion_id="suggestion-1",
            category="config",
            confidence=0.8,
            rationale="High gap count detected",
            proposed_change={"action": "review_gaps"},
        )

        assert suggestion.suggestion_id == "suggestion-1"
        assert suggestion.category == "config"
        assert suggestion.confidence == 0.8
        assert suggestion.applied is False
        assert suggestion.error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
