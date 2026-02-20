import asyncio
import json
import logging
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemocore.core.async_storage import AsyncRedisStorage
from mnemocore.core.config import get_config
from mnemocore.subconscious.daemon import SubconsciousDaemon

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repro_sync")


async def test_receive_event():
    # 1. Initialize Redis
    storage = AsyncRedisStorage.get_instance()
    if not await storage.check_health():
        logger.error("Redis not available. Cannot run reproduction.")
        return

    # 2. Initialize Daemon (Mocking run loop to just check state)
    daemon = SubconsciousDaemon()
    daemon.storage = storage  # Manually inject storage as it's done in run()

    # 3. Simulate API publishing a memory
    test_id = f"mem_test_{int(datetime.now().timestamp())}"
    test_payload = {
        "id": test_id,
        "content": "Test memory for synchronization",
        "metadata": {"source": "repro_script"},
        "ltp_strength": 0.5,
        "created_at": datetime.now().isoformat(),
    }

    logger.info(f"Simulating API: Publishing memory.created for {test_id}")
    await storage.store_memory(test_id, test_payload)
    await storage.publish_event("memory.created", {"id": test_id})

    # 4. Run Daemon's consumption logic (which doesn't exist yet, or verify it fails)
    # We need to expose the consumer if we want to test it specifically, or run the daemon briefly.
    # For now, we will verify that the daemon DOES NOT have the memory in its engine.

    # Wait a bit for async processing (if it were happening)
    await asyncio.sleep(2)

    if test_id in daemon.engine.tier_manager.hot:
        logger.info("SUCCESS: Daemon received the memory!")
    else:
        logger.error("FAILURE: Daemon did NOT receive the memory.")

    # Clean up
    await storage.delete_memory(test_id)
    await storage.close()


if __name__ == "__main__":
    asyncio.run(test_receive_event())
