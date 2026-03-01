import pytest
import asyncio
from unittest.mock import MagicMock
from mnemocore.agent_interface import CognitiveMemoryClient
from mnemocore.core.engine import HAIMEngine
from mnemocore.core.working_memory import WorkingMemoryService
from mnemocore.core.episodic_store import EpisodicStoreService
from mnemocore.core.semantic_store import SemanticStoreService
from mnemocore.core.procedural_store import ProceduralStoreService
from mnemocore.core.meta_memory import MetaMemoryService

@pytest.fixture
def mock_engine():
    engine = MagicMock(spec=HAIMEngine)
    engine.encoder = MagicMock()
    return engine

@pytest.mark.asyncio
async def test_cognitive_client_observe_and_context(mock_engine):
    wm = WorkingMemoryService()
    client = CognitiveMemoryClient(
        engine=mock_engine,
        wm=wm,
        episodic=MagicMock(),
        semantic=MagicMock(),
        procedural=MagicMock(),
        meta=MagicMock()
    )
    
    agent_id = "agent-alpha"
    await client.observe(agent_id, content="User said hi", importance=0.9)
    await client.observe(agent_id, content="User asked about weather", importance=0.7)
    
    ctx = await client.get_working_context(agent_id)
    assert len(ctx) == 2
    assert ctx[0].content == "User said hi"

@pytest.mark.asyncio
async def test_cognitive_client_episodic(mock_engine):
    episodic = EpisodicStoreService()
    client = CognitiveMemoryClient(
        engine=mock_engine,
        wm=MagicMock(),
        episodic=episodic,
        semantic=MagicMock(),
        procedural=MagicMock(),
        meta=MagicMock()
    )
    
    agent_id = "agent-beta"
    ep_id = await client.start_episode(agent_id, goal="Greet user")
    await client.append_event(ep_id, kind="action", content="Said hello")
    await client.end_episode(ep_id, outcome="Success")
    
    recent = episodic.get_recent(agent_id)
    assert len(recent) == 1
    assert recent[0].goal == "Greet user"

@pytest.mark.asyncio
async def test_cognitive_client_recall(mock_engine):
    episodic = EpisodicStoreService()
    client = CognitiveMemoryClient(
        engine=mock_engine,
        wm=MagicMock(),
        episodic=episodic,
        semantic=MagicMock(),
        procedural=MagicMock(),
        meta=MagicMock()
    )
    
    agent_id = "agent-gamma"
    ep_id = await client.start_episode(agent_id, goal="Buy milk")
    await client.end_episode(ep_id, outcome="Success")
    
    # Mock engine query
    mock_engine.query.return_value = [("mem-1", 0.9)]
    
    mock_node = MagicMock()
    mock_node.content = "Semantic info about milk"
    
    async def mock_get_memory(mem_id):
        return mock_node
        
    mock_engine.tier_manager = MagicMock()
    mock_engine.tier_manager.get_memory = mock_get_memory
    
    results = await client.recall(agent_id, query="milk", modes=("episodic", "semantic"))
    assert len(results) == 2
    sources = [r["source"] for r in results]
    assert "episodic" in sources
    assert "semantic/engine" in sources
