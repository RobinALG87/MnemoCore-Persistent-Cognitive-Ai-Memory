import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from mnemocore.core.meta_memory import MetaMemoryService
from mnemocore.core.memory_model import SelfMetric

@pytest.fixture
def mock_engine_with_llm():
    engine = MagicMock()
    engine.subconscious = MagicMock()
    # Mock the LLM returning a valid JSON-embedded response
    engine.subconscious.analyze_dream = AsyncMock(
        return_value='Here is my analysis: {"title": "Reduce Batch Size", "rationale": "High memory consumption", "expected_effect": "Lower latency"}'
    )
    return engine

@pytest.mark.asyncio
async def test_meta_proposal_generation_on_anomaly(mock_engine_with_llm):
    meta = MetaMemoryService()
    
    # Inject healthy metrics
    meta.record_metric("query_hit_rate", 0.95, "5m")
    meta.record_metric("query_latency", 150.0, "5m")
    
    # Inject an anomaly (CPU-bound trigger)
    meta.record_metric("api_failure_rate", 0.15, "5m")  # > 0.1 trigger
    
    # Trigger generation
    proposal_id = await meta.generate_proposals_from_metrics(mock_engine_with_llm)
    
    # Assert LLM was called due to the anomaly
    assert proposal_id is not None
    mock_engine_with_llm.subconscious.analyze_dream.assert_called_once()
    
    # Verify the proposal was stored
    proposals = meta.list_proposals()
    assert len(proposals) == 1
    assert proposals[0].id == proposal_id
    assert proposals[0].title == "Reduce Batch Size"
    assert proposals[0].status == "pending"

@pytest.mark.asyncio
async def test_meta_proposal_skips_when_healthy(mock_engine_with_llm):
    meta = MetaMemoryService()
    
    # Inject ONLY healthy metrics
    meta.record_metric("query_hit_rate", 0.95, "5m")
    meta.record_metric("query_latency", 150.0, "5m")
    meta.record_metric("api_failure_rate", 0.01, "5m")  # < 0.1 trigger
    
    # Trigger generation
    proposal_id = await meta.generate_proposals_from_metrics(mock_engine_with_llm)
    
    # Assert LLM was NOT called (CPU optimization test)
    assert proposal_id is None
    mock_engine_with_llm.subconscious.analyze_dream.assert_not_called()
    assert len(meta.list_proposals()) == 0
