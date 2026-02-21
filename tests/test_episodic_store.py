import pytest
from datetime import datetime
from mnemocore.core.episodic_store import EpisodicStoreService

def test_episodic_store_flow():
    store = EpisodicStoreService()
    agent_id = "agent-x"
    
    # Start episode
    ep_id = store.start_episode(agent_id, goal="Find the keys", context="Living room")
    assert ep_id is not None
    
    # Append events
    store.append_event(ep_id, kind="action", content="Looked under the sofa", metadata={"location": "sofa"})
    store.append_event(ep_id, kind="observation", content="Found nothing")
    
    # End episode
    store.end_episode(ep_id, outcome="Failed", reward=-1.0)
    
    # Retrieve recent
    recent = store.get_recent(agent_id, limit=2)
    assert len(recent) == 1
    
    ep = recent[0]
    assert ep.id == ep_id
    assert ep.agent_id == agent_id
    assert ep.goal == "Find the keys"
    assert ep.outcome == "Failed"
    assert ep.reward == -1.0
    assert len(ep.events) == 2
    assert ep.events[0].kind == "action"
    assert ep.events[0].content == "Looked under the sofa"

def test_episodic_store_context_filtering():
    store = EpisodicStoreService()
    agent_id = "agent-x"
    
    ep1 = store.start_episode(agent_id, goal="Task A", context="ctx1")
    store.end_episode(ep1, outcome="Success")
    
    ep2 = store.start_episode(agent_id, goal="Task B", context="ctx2")
    store.end_episode(ep2, outcome="Success")
    
    recent_ctx1 = store.get_recent(agent_id, context="ctx1")
    assert len(recent_ctx1) == 1
    assert recent_ctx1[0].goal == "Task A"
    
def test_episodic_eviction():
    store = EpisodicStoreService()
    agent_id = "agent-x"
    
    for i in range(3):
        ep_id = store.start_episode(agent_id, goal=f"Goal {i}")
        store.end_episode(ep_id, outcome="Done")
        
    recent = store.get_recent(agent_id, limit=2)
    assert len(recent) == 2
    # Should contain Goal 2 and Goal 1, Goal 0 evicted visually in get_recent limit
    assert recent[0].goal == "Goal 2"
    assert recent[1].goal == "Goal 1"
