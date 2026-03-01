import pytest
from datetime import datetime, timedelta, timezone
from mnemocore.core.working_memory import WorkingMemoryService
from mnemocore.core.memory_model import WorkingMemoryItem

@pytest.mark.asyncio
async def test_working_memory_push_and_get():
    wm = WorkingMemoryService(max_items_per_agent=5)
    
    item = WorkingMemoryItem(
        id="wm_123",
        agent_id="agent1",
        created_at=datetime.now(timezone.utc),
        ttl_seconds=3600,
        content="Testing working memory",
        importance=0.8,
        tags=["test"],
        kind="thought"
    )
    
    await wm.push_item("agent1", item)
    state = await wm.get_state("agent1")
    
    assert state is not None
    assert len(state.items) == 1
    assert state.items[0].id == "wm_123"

@pytest.mark.asyncio
async def test_working_memory_eviction():
    wm = WorkingMemoryService(max_items_per_agent=2)
    
    for i in range(3):
        item = WorkingMemoryItem(
            id=f"wm_{i}",
            agent_id="agent1",
            created_at=datetime.now(timezone.utc) + timedelta(milliseconds=i*50),
            ttl_seconds=3600,
            content=f"Test {i}",
            importance=0.5,
            kind="thought",
            tags=[]
        )
        await wm.push_item("agent1", item)
        
    state = await wm.get_state("agent1")
    assert state is not None
    assert len(state.items) == 2
    # The oldest one (wm_0) should be evicted.
    assert state.items[0].id == "wm_1"
    assert state.items[1].id == "wm_2"

@pytest.mark.asyncio
async def test_working_memory_clear():
    wm = WorkingMemoryService()
    
    item = WorkingMemoryItem(
        id="wm_123",
        agent_id="agent1",
        created_at=datetime.now(timezone.utc),
        ttl_seconds=3600,
        content="Testing working memory",
        importance=0.8,
        kind="thought",
        tags=[]
    )
    
    await wm.push_item("agent1", item)
    await wm.clear("agent1")
    
    state = await wm.get_state("agent1")
    assert state is None or len(state.items) == 0

@pytest.mark.asyncio
async def test_working_memory_prune():
    wm = WorkingMemoryService()
    
    # Push an item that is immediately expired
    item = WorkingMemoryItem(
        id="wm_expired",
        agent_id="agent1",
        created_at=datetime.now(timezone.utc),
        ttl_seconds=-1, # Expired
        content="Expired memory",
        importance=0.1,
        kind="thought",
        tags=[]
    )
    await wm.push_item("agent1", item)
    
    await wm.prune_all()
    state = await wm.get_state("agent1")
    assert state is None or len(state.items) == 0
