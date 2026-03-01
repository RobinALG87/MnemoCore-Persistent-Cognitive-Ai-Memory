import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# 1. Mock dependencies
mock_engine_cls = MagicMock()
mock_engine_instance = MagicMock()
mock_engine_instance.get_stats = AsyncMock(return_value={"engine_version": "3.5.1", "tiers": {"hot_count": 10}})
mock_engine_instance.get_memory = AsyncMock(return_value=None)
mock_engine_instance.delete_memory = AsyncMock(return_value=True)
mock_engine_instance.initialize = AsyncMock(return_value=None)
mock_engine_instance.close = AsyncMock(return_value=None)
mock_engine_cls.return_value = mock_engine_instance

# Mock container
mock_container = MagicMock()
mock_container.redis_storage = AsyncMock()
mock_container.redis_storage.check_health = AsyncMock(return_value=True)
mock_container.qdrant_store = MagicMock()

# Patch before import
patcher1 = patch("mnemocore.api.main.HAIMEngine", mock_engine_cls)
patcher2 = patch("mnemocore.api.main.build_container", return_value=mock_container)
patcher1.start()
patcher2.start()

from mnemocore.api.main import app, get_api_key

client = TestClient(app)

# Bypass auth for functional tests or provide valid key
API_KEY = "test-key"

@pytest.fixture(autouse=True)
def setup_mocks(monkeypatch):
    from mnemocore.core.config import get_config, reset_config
    reset_config()
    monkeypatch.setenv("HAIM_API_KEY", API_KEY)

    # Mock engine state
    app.state.engine = mock_engine_instance
    app.state.container = mock_container
    yield
    reset_config()

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["engine_ready"] is True

def test_stats():
    mock_engine_instance.get_stats.return_value = {
        "engine_version": "3.5.1",
        "tiers": {"hot_count": 10}
    }

    response = client.get("/stats", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert response.json()["tiers"]["hot_count"] == 10

def test_delete_memory_found():
    mock_memory = MagicMock()
    mock_engine_instance.get_memory.return_value = mock_memory

    response = client.delete("/memory/mem_123", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert response.json()["ok"] is True
    mock_engine_instance.delete_memory.assert_called_with("mem_123")

def test_delete_memory_not_found():
    mock_engine_instance.get_memory.return_value = None

    response = client.delete("/memory/mem_missing", headers={"X-API-Key": API_KEY})
    assert response.status_code == 404
    # MnemoCore exception handler returns {"error": ..., "code": ..., "recoverable": ...}
    json_resp = response.json()
    error_text = json_resp.get("error", json_resp.get("detail", json_resp.get("message", ""))).lower()
    assert "not found" in error_text or "memory" in error_text or "mem_missing" in error_text


# ============================================================================
# Task 15.4: Additional API Functional Tests
# ============================================================================

class TestDreamEndpoint:
    """Tests for /dream endpoint with rate limiting."""

    def test_dream_endpoint_success(self, monkeypatch):
        """Test successful dream endpoint call."""
        # Mock the SubconsciousDaemon
        mock_daemon = MagicMock()
        mock_daemon.extract_concepts = AsyncMock(return_value=[])
        mock_daemon.draw_parallels = AsyncMock(return_value=[])
        mock_daemon.generate_insight = AsyncMock(return_value=None)

        # Mock tier_manager with hot memories
        mock_engine_instance.tier_manager = MagicMock()
        mock_engine_instance.tier_manager.hot = {}

        with patch("mnemocore.subconscious.daemon.SubconsciousDaemon", return_value=mock_daemon):
            response = client.post(
                "/dream",
                headers={"X-API-Key": API_KEY},
                json={"max_cycles": 1}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True

    def test_dream_endpoint_no_memories(self, monkeypatch):
        """Test dream endpoint with no memories to process."""
        mock_engine_instance.tier_manager = MagicMock()
        mock_engine_instance.tier_manager.hot = {}

        response = client.post(
            "/dream",
            headers={"X-API-Key": API_KEY},
            json={"max_cycles": 1}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["cycles_completed"] == 0
        assert "No memories" in data["message"]


class TestExportEndpoint:
    """Tests for /export endpoint with upper bound limit parameter."""

    def test_export_default_limit(self, monkeypatch):
        """Test export endpoint with default limit."""
        mock_engine_instance.tier_manager = MagicMock()
        mock_engine_instance.tier_manager.hot = {}
        mock_engine_instance.tier_manager.qdrant_store = None

        response = client.get(
            "/export",
            headers={"X-API-Key": API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["count"] == 0

    def test_export_with_upper_bound_limit(self, monkeypatch):
        """Test export endpoint respects upper bound limit."""
        # Create mock memories
        mock_node = MagicMock()
        mock_node.id = "mem_123"
        mock_node.content = "Test content"
        mock_node.created_at = MagicMock()
        mock_node.created_at.isoformat = MagicMock(return_value="2024-01-15T10:00:00Z")
        mock_node.ltp_strength = 0.5
        mock_node.tier = "hot"
        mock_node.metadata = {}

        mock_engine_instance.tier_manager = MagicMock()
        mock_engine_instance.tier_manager.hot = {"mem_123": mock_node}
        mock_engine_instance.tier_manager.qdrant_store = None

        # Request with limit of 10
        response = client.get(
            "/export?limit=10",
            headers={"X-API-Key": API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["count"] == 1

    def test_export_invalid_tier(self, monkeypatch):
        """Test export endpoint rejects invalid tier."""
        response = client.get(
            "/export?tier=invalid_tier",
            headers={"X-API-Key": API_KEY}
        )

        assert response.status_code == 400

    def test_export_invalid_format(self, monkeypatch):
        """Test export endpoint rejects invalid format."""
        response = client.get(
            "/export?format=xml",
            headers={"X-API-Key": API_KEY}
        )

        assert response.status_code == 400


class TestConcurrentStore:
    """Tests for concurrent /store requests."""

    def test_concurrent_store_requests(self, monkeypatch):
        """Test that concurrent store requests are handled properly."""
        import threading

        results = []
        errors = []

        def make_store_request(content):
            try:
                response = client.post(
                    "/store",
                    headers={"X-API-Key": API_KEY},
                    json={"content": content}
                )
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads to simulate concurrent requests
        threads = []
        for i in range(5):
            t = threading.Thread(target=make_store_request, args=(f"Concurrent content {i}",))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=10)

        # All requests should succeed (200) or be rate limited (429)
        for status in results:
            assert status in [200, 201, 429]

    def test_store_with_metadata(self, monkeypatch):
        """Test store request with metadata."""
        mock_engine_instance.store = AsyncMock(return_value="mem_new_123")

        response = client.post(
            "/store",
            headers={"X-API-Key": API_KEY},
            json={
                "content": "Test memory with metadata",
                "metadata": {"key": "value", "tags": ["tag1", "tag2"]}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True


class TestInvalidContentTypes:
    """Tests for handling invalid content types."""

    def test_store_invalid_content_type(self, monkeypatch):
        """Test store endpoint with invalid Content-Type."""
        response = client.post(
            "/store",
            headers={"X-API-Key": API_KEY, "Content-Type": "text/plain"},
            content="plain text content"
        )

        # Should return 422 (Unprocessable Entity) or 400
        assert response.status_code in [400, 422]

    def test_query_invalid_content_type(self, monkeypatch):
        """Test query endpoint with invalid Content-Type."""
        response = client.post(
            "/query",
            headers={"X-API-Key": API_KEY, "Content-Type": "text/plain"},
            content="plain text query"
        )

        # Should return 422 (Unprocessable Entity) or 400
        assert response.status_code in [400, 422]

    def test_store_missing_body(self, monkeypatch):
        """Test store endpoint with missing request body."""
        response = client.post(
            "/store",
            headers={"X-API-Key": API_KEY}
        )

        # Should return 422 (Unprocessable Entity)
        assert response.status_code == 422

    def test_store_empty_content(self, monkeypatch):
        """Test store endpoint with empty content string."""
        response = client.post(
            "/store",
            headers={"X-API-Key": API_KEY},
            json={"content": ""}
        )

        # Should return 422 (validation error for empty content)
        assert response.status_code == 422

    def test_query_missing_query_string(self, monkeypatch):
        """Test query endpoint with missing query string."""
        response = client.post(
            "/query",
            headers={"X-API-Key": API_KEY},
            json={"top_k": 10}
        )

        # Should return 422 (validation error for missing query)
        assert response.status_code == 422
