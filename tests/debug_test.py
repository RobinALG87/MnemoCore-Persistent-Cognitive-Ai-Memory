
import pytest
import asyncio
from src.core.engine import HAIMEngine

@pytest.mark.asyncio
async def test_simple():
    engine = HAIMEngine()
    print("Engine created")
    # store something
    mid = await engine.store("debug content")
    print(f"Stored {mid}")
    assert mid is not None
