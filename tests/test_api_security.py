import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Ensure path is set
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemocore.core.config import reset_config

API_KEY = "test-key"

# Setup mocks before importing app
mock_engine_cls = MagicMock()
mock_engine_instance = MagicMock()
mock_engine_instance.get_stats = AsyncMock(return_value={"status": "ok"})
mock_engine_instance.get_memory = AsyncMock(return_value=None)
mock_engine_instance.delete_memory = AsyncMock(return_value=True)
mock_engine_instance.store = AsyncMock(return_value="mem_id_123")
mock_engine_instance.query = AsyncMock(return_value=[("mem_id_123", 0.9)])
mock_engine_instance.initialize = AsyncMock(return_value=None)
mock_engine_instance.close = AsyncMock(return_value=None)
mock_engine_cls.return_value = mock_engine_instance

# Mock container
mock_container = MagicMock()
mock_container.redis_storage = AsyncMock()
mock_container.redis_storage.check_health = AsyncMock(return_value=True)
mock_container.redis_storage.store_memory = AsyncMock()
mock_container.redis_storage.publish_event = AsyncMock()
mock_container.redis_storage.retrieve_memory = AsyncMock(return_value=None)
mock_container.redis_storage.delete_memory = AsyncMock()
mock_container.redis_storage.close = AsyncMock()
mock_container.qdrant_store = MagicMock()

# Setup pipeline mock
mock_pipeline = MagicMock()
mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
mock_pipeline.__aexit__ = AsyncMock(return_value=None)
mock_pipeline.incr = MagicMock()
mock_pipeline.expire = MagicMock()
mock_pipeline.execute = AsyncMock(return_value=[1, True])

mock_redis_client = MagicMock()
mock_redis_client.pipeline.return_value = mock_pipeline
mock_container.redis_storage.redis_client = mock_redis_client

# Patch before import
patcher1 = patch("mnemocore.api.main.HAIMEngine", mock_engine_cls)
patcher2 = patch("mnemocore.api.main.build_container", return_value=mock_container)
patcher1.start()
patcher2.start()

from mnemocore.api.main import app

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("HAIM_API_KEY", API_KEY)
    reset_config()
    # Mock app state
    app.state.engine = mock_engine_instance
    app.state.container = mock_container
    # Reset rate limiter mock to default (within limit) - just set return_value, don't replace the mock
    mock_pipeline.execute.return_value = [1, True]
    yield
    reset_config()

@pytest.fixture
def client(setup_env):
    with TestClient(app) as c:
        yield c

def test_health_public(client):
    """Health endpoint should be public."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_secure_endpoints(client, monkeypatch):
    """Verify endpoints require X-API-Key."""
    # 1. Store
    response = client.post("/store", json={"content": "test"})
    assert response.status_code == 403

    # 2. Query
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 403

    # 3. Valid key
    mock_memory = MagicMock(
        id="mem_1", content="test", metadata={}, ltp_strength=0.5,
        created_at=MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:00:00"))
    )
    mock_engine_instance.get_memory.return_value = mock_memory
    mock_engine_instance.store.return_value = "mem_1"

    response = client.post(
        "/store",
        json={"content": "test"},
        headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200

# --- Enhanced Security Tests ---

def test_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert "Content-Security-Policy" in response.headers
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

def test_cors_headers(client):
    headers = {"Origin": "https://example.com"}
    response = client.get("/", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"

def test_api_key_missing_enhanced(client):
    response = client.post("/store", json={"content": "test"})
    assert response.status_code == 403

def test_api_key_invalid_enhanced(client):
    response = client.post("/store", json={"content": "test"}, headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 403

def test_query_max_length_validation(client):
    long_query = "a" * 10001
    response = client.post(
        "/query",
        json={"query": long_query},
        headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 422

def test_rate_limiter_within_limit(client):
    # Ensure pipeline execute returns count < limit (default 100)
    mock_pipeline.execute.return_value = [1, True]

    mock_memory = MagicMock(
        id="mem_1", content="test", metadata={}, ltp_strength=0.5,
        created_at=MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:00:00"))
    )
    mock_engine_instance.get_memory.return_value = mock_memory
    mock_engine_instance.store.return_value = "mem_1"

    response = client.post(
        "/store",
        json={"content": "test"},
        headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True

# Note: Rate limiter exceeded tests are in test_api_security_limits.py
# which has more comprehensive rate limit testing with proper isolation
