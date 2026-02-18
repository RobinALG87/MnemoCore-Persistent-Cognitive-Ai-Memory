import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def mock_deps():
    mock_engine = MagicMock()
    mock_engine.initialize = AsyncMock()
    mock_engine.close = AsyncMock()

    mock_redis = AsyncMock()
    mock_redis.check_health = AsyncMock(return_value=True)
    mock_redis.close = AsyncMock()

    mock_container = MagicMock()
    mock_container.redis_storage = mock_redis
    mock_container.qdrant_store = MagicMock()

    with patch("src.api.main.HAIMEngine", return_value=mock_engine), \
         patch("src.api.main.build_container", return_value=mock_container), \
         patch("src.api.main.TierManager", return_value=MagicMock()):
        yield mock_engine, mock_redis

def test_engine_lifecycle(mock_deps):
    """Test that engine is initialized and closed via lifespan."""
    mock_engine, _ = mock_deps
    
    # We need to import app INSIDE the test or after the patch is active
    from src.api.main import app
    
    with TestClient(app) as client:
        # Check if engine was initialized and stored in app.state
        assert hasattr(app.state, "engine")
        assert app.state.engine == mock_engine
        
    # Check if close was called on exit
    mock_engine.close.assert_called_once()

def test_delete_endpoint_stability(mock_deps):
    """Test that DELETE endpoint uses the new engine.delete_memory method."""
    mock_engine, mock_redis = mock_deps
    from src.api.main import app
    
    mock_engine.delete_memory.return_value = True
    
    with TestClient(app) as client:
        response = client.delete(
            "/memory/test_mem_123",
            headers={"X-API-Key": "mnemocore-beta-key"}
        )
        
        assert response.status_code == 200
        assert response.json() == {"ok": True, "deleted": "test_mem_123"}
        
        # Verify engine.delete_memory was called
        mock_engine.delete_memory.assert_called_with("test_mem_123")
        # Verify redis delete was called
        mock_redis.delete_memory.assert_called_with("test_mem_123")

def test_security_middleware_fallback(mock_deps):
    """Test security middleware with environment variable fallback."""
    _, _ = mock_deps
    from src.api.main import app
    from src.core import config
    
    # Mock config to have NO security section
    mock_conf = MagicMock()
    mock_conf.security = None
    
    with patch("src.api.main.get_config", return_value=mock_conf), \
         patch.dict(os.environ, {"HAIM_API_KEY": "env-secret-key"}):
        
        with TestClient(app) as client:
            # Should fail with wrong key
            response = client.get("/memory/123", headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 403
            
            # Should succeed with env key (Wait, the middleware in main.py uses get_config() internally)
            # Actually, the middleware I wrote uses 'expected_key = security.api_key if security else os.getenv("HAIM_API_KEY", "mnemocore-beta-key")'
            # So if security is None, it uses the env var.
            
            response = client.get("/memory/123", headers={"X-API-Key": "env-secret-key"})
            # It will still reach the handler, which might return 404/200, but not 403.
            assert response.status_code != 403
