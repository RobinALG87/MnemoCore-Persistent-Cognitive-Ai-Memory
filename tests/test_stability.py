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
    mock_engine.get_memory = AsyncMock(return_value=MagicMock())
    mock_engine.delete_memory = AsyncMock(return_value=True)

    mock_redis = AsyncMock()
    mock_redis.check_health = AsyncMock(return_value=True)
    mock_redis.close = AsyncMock()

    mock_container = MagicMock()
    mock_container.redis_storage = mock_redis
    mock_container.qdrant_store = MagicMock()

    # Build a minimal config mock so lifespan's security check passes
    mock_security = MagicMock()
    mock_security.api_key = "test-api-key"
    mock_config = MagicMock()
    mock_config.security = mock_security
    mock_config.dimensionality = 1024

    with patch("mnemocore.api.main.HAIMEngine", return_value=mock_engine), \
         patch("mnemocore.api.main.build_container", return_value=mock_container), \
         patch("mnemocore.api.main.get_config", return_value=mock_config):
        yield mock_engine, mock_redis

def test_engine_lifecycle(mock_deps):
    """Test that engine is initialized and closed via lifespan."""
    mock_engine, _ = mock_deps

    # We need to import app INSIDE the test or after the patch is active
    from mnemocore.api.main import app

    with TestClient(app) as client:
        # Check if engine was initialized and stored in app.state
        assert hasattr(app.state, "engine")
        assert app.state.engine == mock_engine

    # Check if close was called on exit
    mock_engine.close.assert_called_once()

def test_delete_endpoint_stability(mock_deps):
    """Test that DELETE endpoint uses the new engine.delete_memory method."""
    mock_engine, mock_redis = mock_deps
    from mnemocore.api.main import app

    with TestClient(app) as client:
        response = client.delete(
            "/memory/test_mem_123",
            headers={"X-API-Key": "test-api-key"}
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True, "deleted": "test_mem_123"}

        # Verify engine.delete_memory was called
        mock_engine.delete_memory.assert_called_with("test_mem_123")

def test_security_middleware_fallback(mock_deps):
    """Test security middleware with environment variable fallback."""
    _, _ = mock_deps

    # Build full mocks needed for lifespan startup
    mock_engine2 = MagicMock()
    mock_engine2.initialize = AsyncMock()
    mock_engine2.close = AsyncMock()
    mock_engine2.get_memory = AsyncMock(return_value=None)  # 404 is fine here
    mock_redis2 = AsyncMock()
    mock_redis2.check_health = AsyncMock(return_value=True)
    mock_redis2.retrieve_memory = AsyncMock(return_value=None)  # no cache hit
    mock_container2 = MagicMock()
    mock_container2.redis_storage = mock_redis2
    mock_container2.qdrant_store = MagicMock()

    # Config with security section using the env-var key (for lifespan startup check)
    # but get_api_key() will fall back to env var when security.api_key is falsy.
    mock_conf_no_sec = MagicMock()
    mock_security_no_key = MagicMock()
    mock_security_no_key.api_key = ""   # falsy → triggers env-var fallback in get_api_key
    mock_conf_no_sec.security = mock_security_no_key
    mock_conf_no_sec.dimensionality = 1024

    with patch("mnemocore.api.main.get_config", return_value=mock_conf_no_sec), \
         patch("mnemocore.api.main.HAIMEngine", return_value=mock_engine2), \
         patch("mnemocore.api.main.build_container", return_value=mock_container2), \
         patch.dict(os.environ, {"HAIM_API_KEY": "env-secret-key"}, clear=False):

        # Re-import to get fresh app with new patches applied
        import importlib
        import mnemocore.api.main as main_module
        importlib.reload(main_module)

        with TestClient(main_module.app) as client:
            # Wrong key -> 403
            response = client.get("/memory/123", headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 403

            # Correct env key -> not 403 (could be 404 for missing memory)
            response = client.get("/memory/123", headers={"X-API-Key": "env-secret-key"})
            assert response.status_code != 403
