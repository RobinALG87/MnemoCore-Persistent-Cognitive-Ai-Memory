import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Ensure path is set
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.main import app
from src.core.config import reset_config

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_public(client):
    """Health endpoint should be public."""
    # Mock get_stats on app.state.engine (lifespan initializes it)
    with patch.object(app.state.engine, 'get_stats', return_value={"status": "ok"}):
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

def test_secure_endpoints(client, monkeypatch):
    """Verify endpoints require X-API-Key."""
    monkeypatch.setenv("HAIM_API_KEY", "test-key")
    reset_config()
    
    # 1. Store
    response = client.post("/store", json={"content": "test"})
    assert response.status_code == 403
    
    # 2. Query
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 403
    
    # 3. Valid key
    with patch.object(app.state.engine, 'store', return_value="mem_1"), \
         patch.object(app.state.engine, 'get_memory', return_value=MagicMock()):
        response = client.post(
            "/store", 
            json={"content": "test"},
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200

# --- Enhanced Security Tests ---

@pytest.fixture
def mock_dependencies():
    with patch("src.api.main.HAIMEngine") as MockEngine, \
         patch("src.core.async_storage.AsyncRedisStorage.get_instance") as mock_redis_get:

        # Setup Mock Engine
        mock_engine_instance = MagicMock()
        mock_engine_instance.store = MagicMock(return_value="mem_id_123")
        mock_engine_instance.query = MagicMock(return_value=[("mem_id_123", 0.9)])
        mock_engine_instance.get_memory.return_value = MagicMock(
            id="mem_id_123", content="test", metadata={}, ltp_strength=0.5, created_at=MagicMock()
        )
        mock_engine_instance.close = MagicMock()
        MockEngine.return_value = mock_engine_instance

        # Setup Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.check_health = AsyncMock(return_value=True)
        mock_redis_instance.close = AsyncMock()
        mock_redis_instance.store_memory = AsyncMock()
        mock_redis_instance.publish_event = AsyncMock()

        # Setup Pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.__aenter__.return_value = mock_pipeline
        mock_pipeline.__aexit__.return_value = None
        mock_pipeline.execute = AsyncMock(return_value=[1, True])

        mock_redis_client = MagicMock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        mock_redis_instance.redis_client = mock_redis_client

        mock_redis_get.return_value = mock_redis_instance

        yield {
            "engine": mock_engine_instance,
            "redis": mock_redis_instance,
            "pipeline": mock_pipeline
        }

def test_security_headers(mock_dependencies):
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "Content-Security-Policy" in response.headers
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

def test_cors_headers(mock_dependencies):
    with TestClient(app) as client:
        headers = {"Origin": "https://example.com"}
        response = client.get("/", headers=headers)
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"

def test_api_key_missing_enhanced(mock_dependencies):
    with TestClient(app) as client:
        response = client.post("/store", json={"content": "test"})
        assert response.status_code == 403

def test_api_key_invalid_enhanced(mock_dependencies):
    with TestClient(app) as client:
        response = client.post("/store", json={"content": "test"}, headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403

def test_query_max_length_validation(mock_dependencies):
    with TestClient(app) as client:
        long_query = "a" * 10001
        response = client.post(
            "/query",
            json={"query": long_query},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        assert response.status_code == 422

def test_rate_limiter_within_limit(mock_dependencies):
    mocks = mock_dependencies
    # Ensure pipeline execute returns count < limit (default 100)
    mocks["pipeline"].execute.return_value = [1, True]

    with TestClient(app) as client:
        response = client.post(
            "/store",
            json={"content": "test"},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )

        assert response.status_code == 200
        assert response.json()["ok"] is True

def test_rate_limiter_exceeded(mock_dependencies):
    mocks = mock_dependencies
    # Simulate return value [count=101, expire_result=True] (Limit is 100)
    mocks["pipeline"].execute.return_value = [101, True]

    with TestClient(app) as client:
        response = client.post(
            "/store",
            json={"content": "test"},
            headers={"X-API-Key": "mnemocore-beta-key"}
        )

        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
