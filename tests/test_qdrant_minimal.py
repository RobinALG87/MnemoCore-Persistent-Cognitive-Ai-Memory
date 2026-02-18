
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Add src to path just in case
sys.path.append(os.getcwd())

from mnemocore.core.qdrant_store import QdrantStore

class TestMinimal(unittest.IsolatedAsyncioTestCase):
    async def test_minimal(self):
        print("Starting test_minimal", flush=True)
        with patch("src.core.qdrant_store.AsyncQdrantClient") as MockClass:
            # Use AsyncMock so async methods (ensure_collections) can be awaited
            MockClass.return_value = AsyncMock()
            print("Patched AsyncQdrantClient", flush=True)
            try:
                # Bypass singleton and global patch
                store = QdrantStore(url="http://localhost:6333", api_key=None, dimensionality=1024)
                print("Instantiated QdrantStore", flush=True)
            except Exception as e:
                with open("error_log.txt", "w") as f:
                    f.write(f"Failed to instantiate QdrantStore: {e}\nType: {type(e)}\n")
                    import traceback
                    traceback.print_exc(file=f)
                print(f"Failed to instantiate QdrantStore: {e}", flush=True)
                raise
            
            await store.ensure_collections()
            print("Called ensure_collections", flush=True)
            self.assertTrue(True)

if __name__ == "__main__":
    print("Running main", flush=True)
    unittest.main()
