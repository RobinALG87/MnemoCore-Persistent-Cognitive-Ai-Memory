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
patcher1 = patch("src.api.main.HAIMEngine", mock_engine_cls)
patcher2 = patch("src.api.main.build_container", return_value=mock_container)
patcher1.start()
patcher2.start()

from src.api.main import app, get_api_key

client = TestClient(app)

# Bypass auth for functional tests or provide valid key
API_KEY = "test-key"

@pytest.fixture(autouse=True)
def setup_mocks(monkeypatch):
    from src.core.config import get_config, reset_config
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
