import asyncio
import sys
import logging
from unittest.mock import AsyncMock

from src.core.async_storage import AsyncRedisStorage

logging.basicConfig(level=logging.DEBUG)

async def main():
    print("Starting debug_async.py...")
    try:
        mock_client = AsyncMock()
        print("Mock client created.")

        storage = AsyncRedisStorage(client=mock_client)
        print("AsyncRedisStorage initialized.")

        node_id = "mem_debug"
        data = {"content": "debug"}

        print("Calling store_memory...")
        await storage.store_memory(node_id, data)
        print("store_memory returned.")

        mock_client.set.assert_called_once()
        print("Assertion passed.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
