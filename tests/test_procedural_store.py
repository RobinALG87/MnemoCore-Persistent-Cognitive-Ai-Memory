import pytest
from datetime import datetime, timezone
from mnemocore.core.procedural_store import ProceduralStoreService
from mnemocore.core.memory_model import Procedure

@pytest.mark.asyncio
async def test_procedural_store_add_and_get():
    store = ProceduralStoreService()
    
    proc = Procedure(
        id="proc-1",
        name="extract_information",
        description="Extracts names and dates from text",
        created_by_agent="system",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        steps=["Read text", "Regex match", "Return JSON"],
        trigger_pattern=".*",
        success_count=0,
        failure_count=0,
        reliability=1.0,
        tags=[]
    )
    
    await store.store_procedure(proc)
    
    p2 = await store.get_procedure("proc-1")
    assert p2 is not None
    assert p2.name == "extract_information"

@pytest.mark.asyncio
async def test_procedural_store_outcome():
    store = ProceduralStoreService()
    proc = Procedure(
        id="proc-2",
        name="API_Call",
        description="GET request",
        created_by_agent="system",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        steps=["Requests.get"],
        trigger_pattern=".*",
        success_count=0,
        failure_count=0,
        reliability=1.0,
        tags=[]
    )
    await store.store_procedure(proc)
    
    # Success
    await store.record_procedure_outcome("proc-2", success=True)
    p2 = await store.get_procedure("proc-2")
    assert p2.success_count == 1
    assert p2.failure_count == 0
    
    # Failure
    await store.record_procedure_outcome("proc-2", success=False)
    p3 = await store.get_procedure("proc-2")
    assert p3.success_count == 1
    assert p3.failure_count == 1
    assert p3.reliability == 0.9

@pytest.mark.asyncio
async def test_procedural_store_find():
    store = ProceduralStoreService()
    proc1 = Procedure(
        id="p1", name="search_web", description="Find info online", steps=[],
        created_by_agent="system", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc), trigger_pattern="search", success_count=0, failure_count=0, reliability=1.0, tags=[]
    )
    proc2 = Procedure(
        id="p2", name="calculate_math", description="Evaluate math expression", steps=[],
        created_by_agent="system", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc), trigger_pattern="math", success_count=0, failure_count=0, reliability=1.0, tags=[]
    )
    
    await store.store_procedure(proc1)
    await store.store_procedure(proc2)
    
    results = await store.find_applicable_procedures("math expression")
    assert len(results) > 0
    # The basic regex search should catch math
    assert any(p.id == "p2" for p in results)
