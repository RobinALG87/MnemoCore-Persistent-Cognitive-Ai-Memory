
import asyncio
import sys
import os
from unittest.mock import patch

# Add src to path
sys.path.append(os.getcwd())

from src.core.qdrant_store import QdrantStore

async def main():
    print("Starting main", flush=True)
    with patch("src.core.qdrant_store.AsyncQdrantClient") as MockClass:
        print("Patched AsyncQdrantClient", flush=True)
        try:
            store = QdrantStore()
            print("Instantiated QdrantStore", flush=True)
            await store.ensure_collections()
            print("Called ensure_collections", flush=True)
        except Exception as e:
            print(f"Error: {e}", flush=True)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
