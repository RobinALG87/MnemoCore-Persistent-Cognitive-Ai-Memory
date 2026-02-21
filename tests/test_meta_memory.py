import pytest
from mnemocore.core.meta_memory import MetaMemoryService

def test_meta_memory_metrics():
    meta = MetaMemoryService()
    
    meta.record_metric("inference_time_ms", 120.5, window="1m")
    meta.record_metric("inference_time_ms", 130.0, window="1m")
    meta.record_metric("token_count", 500, window="1h")
    
    assert len([m for m in meta.list_metrics() if m.name == "inference_time_ms"]) == 2
    assert len([m for m in meta.list_metrics() if m.name == "token_count"]) == 1

import pytest
from datetime import datetime
from mnemocore.core.meta_memory import MetaMemoryService
from mnemocore.core.memory_model import SelfImprovementProposal

def test_meta_memory_proposals():
    meta = MetaMemoryService()
    
    proposal = SelfImprovementProposal(
        id="prop1",
        created_at=datetime.utcnow(),
        author="system",
        title="Reduce Temp",
        description="Agent repeats tools too often",
        rationale="Reduce temperature or augment prompt with history",
        expected_effect="Less tool looping",
        status="pending",
        metadata={"confidence": 0.85}
    )
    
    meta.create_proposal(proposal)
    
    proposals = meta.list_proposals()
    assert len(proposals) == 1
    
    p = proposals[0]
    assert p.status == "pending"
    assert "reduce temperature" in p.rationale.lower()
    
    # Update status
    meta.update_proposal_status(p.id, "approved")
    
    proposals2 = meta.list_proposals()
    assert proposals2[0].status == "approved"
