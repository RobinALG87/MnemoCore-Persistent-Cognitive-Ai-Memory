
import asyncio
import uuid
from src.core.engine import HAIMEngine
from src.core.config import get_config

async def verify():
    print("Initializing Engine...")
    engine = HAIMEngine()
    
    content = "This is a test memory to verify UUID format."
    print(f"Storing memory: '{content}'")
    
    node_id = await engine.store(content)
    print(f"Generated Node ID: {node_id}")
    
    # Check if it's a valid UUID
    try:
        val = uuid.UUID(node_id, version=4)
        print(f"SUCCESS: {node_id} is a valid UUIDv4")
    except ValueError:
        print(f"FAILURE: {node_id} is NOT a valid UUIDv4")
        return

    # Verify retrieval
    node = await engine.get_memory(node_id)
    if node and node.id == node_id:
        print(f"SUCCESS: Retrieved node with ID {node.id}")
    else:
        print("FAILURE: Could not retrieve node or ID mismatch")

if __name__ == "__main__":
    asyncio.run(verify())
