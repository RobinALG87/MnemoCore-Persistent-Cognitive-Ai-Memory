import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.memory_model import WorkingMemoryState, WorkingMemoryItem, Episode, EpisodeEvent

@pytest.fixture
def mock_engine():
    engine = HAIMEngine(persist_path=":memory:", tier_manager=AsyncMock())
    engine.working_memory = MagicMock()
    engine.episodic_store = MagicMock()
    
    # Mock node retrieval
    mock_node = MagicMock()
    mock_node.content = "Related Concept Content"
    mock_node.tier = "hot"
    engine.get_memory = AsyncMock(return_value=mock_node)
    
    # Mock query
    engine.query = AsyncMock(return_value=[("node_1", 0.85), ("node_2", 0.65), ("node_3", 0.10)])
    
    return engine

@pytest.mark.asyncio
async def test_generate_subtle_thoughts(mock_engine):
    agent_id = "agent_42"
    
    # Setup working memory mock
    wm_item = WorkingMemoryItem(
        id="wm_1", agent_id=agent_id, created_at=datetime.now(timezone.utc),
        ttl_seconds=3600, content="I need to solve a math problem.", kind="thought",
        importance=0.9, tags=[]
    )
    mock_engine.working_memory.get_state.return_value = WorkingMemoryState(
        agent_id=agent_id, max_items=10, items=[wm_item]
    )
    
    # Setup episodic mock
    ep_event = EpisodeEvent(
        timestamp=datetime.now(timezone.utc), kind="observation",
        content="The user asked about calculus.", metadata={}
    )
    episode = Episode(
        id="ep_1", agent_id=agent_id, started_at=datetime.now(timezone.utc),
        ended_at=None, goal="Help user with math", context="academic",
        events=[ep_event], outcome="in_progress", reward=None,
        links_prev=[], links_next=[], ltp_strength=0.5, reliability=0.8
    )
    mock_engine.episodic_store.get_recent.return_value = [episode]
    
    # Run
    associations = await mock_engine.generate_subtle_thoughts(agent_id, limit=2)
    
    # Assert
    assert len(associations) == 2  # Limits to 2 items above threshold
    assert associations[0]["agent_id"] == "agent_42"
    assert associations[0]["related_concept_ids"] == ["node_1"]
    assert associations[0]["suggestion_text"] == "Related Concept Content"
    assert associations[0]["confidence"] == 0.85
    
    # Verify query was built properly
    mock_engine.query.assert_called_once()
    query_str = mock_engine.query.call_args[0][0]
    assert "I need to solve a math problem." in query_str
    assert "Goal: Help user with math" in query_str
    assert "The user asked about calculus." in query_str

@pytest.mark.asyncio
async def test_empty_subtle_thoughts(mock_engine):
    # Setup empty WM / EP
    mock_engine.working_memory.get_state.return_value = None
    mock_engine.episodic_store.get_recent.return_value = []
    
    associations = await mock_engine.generate_subtle_thoughts("agent_42")
    assert associations == []
    mock_engine.query.assert_not_called()
