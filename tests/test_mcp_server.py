import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We need to patch HAIMEngine before importing src.mcp_server
# so it doesn't try to initialize the real engine (which reads files).

@pytest.fixture(scope="module", autouse=True)
def mock_haim_engine_class():
    with patch("src.core.engine.HAIMEngine") as MockEngineClass:
        # Configure the mock instance
        mock_instance = MockEngineClass.return_value
        mock_instance.store.return_value = "mem_test_123"

        # Mock query return
        mock_node = MagicMock()
        mock_node.id = "mem_123"
        mock_node.content = "Test content"
        mock_node.metadata = {"tag": "test"}
        mock_node.created_at.isoformat.return_value = "2023-01-01T00:00:00"

        mock_instance.query.return_value = [("mem_123", 0.95)]
        mock_instance.get_memory.return_value = mock_node

        # Mock tier manager
        mock_instance.tier_manager.hot = {"mem_123": mock_node}

        # Mock get_recent_memories
        mock_instance.get_recent_memories.return_value = [mock_node]

        mock_instance.get_stats.return_value = {"status": "ok"}

        yield MockEngineClass

def test_store_memory_tool(mock_haim_engine_class):
    # Import inside test to ensure mock is active
    if "src.mcp_server" in sys.modules:
        del sys.modules["src.mcp_server"]
    from src.mcp_server import store_memory

    result = store_memory("Test content", metadata={"tag": "test"})
    assert "mem_test_123" in result

    mock_haim_engine_class.return_value.store.assert_called()

def test_query_memories_tool(mock_haim_engine_class):
    if "src.mcp_server" in sys.modules:
        del sys.modules["src.mcp_server"]
    from src.mcp_server import query_memories

    result = query_memories("Test query")
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["id"] == "mem_123"

    mock_haim_engine_class.return_value.query.assert_called()

def test_resources(mock_haim_engine_class):
    if "src.mcp_server" in sys.modules:
        del sys.modules["src.mcp_server"]
    from src.mcp_server import get_recent_memories, get_stats

    recent = get_recent_memories()
    assert "mem_123" in recent

    stats = get_stats()
    assert "status" in stats
