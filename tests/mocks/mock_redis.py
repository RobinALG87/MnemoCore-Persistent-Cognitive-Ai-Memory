"""
Mock Async Redis Storage
========================
In-memory mock implementation of AsyncRedisStorage for offline testing.

Uses fakeredis for Redis-compatible behavior when available,
falls back to pure Python in-memory implementation.

Implements all public methods from src.core.async_storage.AsyncRedisStorage
"""

from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from loguru import logger

# Try to import fakeredis for realistic Redis behavior
try:
    import fakeredis.aioredis as fakeredis
    HAS_FAKEREDIS = True
except ImportError:
    HAS_FAKEREDIS = False
    logger.info("[MockRedis] fakeredis not available, using in-memory dict storage")


@dataclass
class StreamEntry:
    """Mock Redis Stream entry."""
    id: str
    data: Dict[str, str]


class InMemoryRedisClient:
    """
    Pure Python in-memory Redis client mock.

    Implements the subset of Redis commands used by AsyncRedisStorage:
    - get/set/setex/delete
    - mget
    - zadd/zrange/zrem
    - xadd
    - ping
    - pipeline
    """

    def __init__(self, decode_responses: bool = True):
        self._data: Dict[str, Any] = {}
        self._ttls: Dict[str, int] = {}
        self._sorted_sets: Dict[str, Dict[str, float]] = {}
        self._streams: Dict[str, List[StreamEntry]] = {}
        self._decode_responses = decode_responses
        self._id_counter = 0

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> Optional[str]:
        # Check TTL
        if key in self._ttls:
            import time
            if time.time() > self._ttls[key]:
                del self._data[key]
                del self._ttls[key]
                return None
        return self._data.get(key)

    async def set(self, key: str, value: str) -> bool:
        self._data[key] = value
        return True

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        import time
        self._data[key] = value
        self._ttls[key] = int(time.time()) + ttl
        return True

    async def delete(self, key: str) -> int:
        if key in self._data:
            del self._data[key]
            self._ttls.pop(key, None)
            return 1
        return 0

    async def mget(self, keys: List[str]) -> List[Optional[str]]:
        import time
        results = []
        for key in keys:
            # Check TTL
            if key in self._ttls:
                if time.time() > self._ttls[key]:
                    del self._data[key]
                    del self._ttls[key]
                    results.append(None)
                    continue
            results.append(self._data.get(key))
        return results

    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        if key not in self._sorted_sets:
            self._sorted_sets[key] = {}
        added = 0
        for member, score in mapping.items():
            if member not in self._sorted_sets[key]:
                added += 1
            self._sorted_sets[key][member] = score
        return added

    async def zrange(self, key: str, start: int, stop: int) -> List[str]:
        if key not in self._sorted_sets:
            return []
        # Sort by score
        sorted_items = sorted(self._sorted_sets[key].items(), key=lambda x: x[1])
        # Handle Python-style slicing (stop is inclusive in Redis ZRANGE)
        if stop >= 0:
            stop += 1
        return [item[0] for item in sorted_items[start:stop]]

    async def zrem(self, key: str, member: str) -> int:
        if key not in self._sorted_sets:
            return 0
        if member in self._sorted_sets[key]:
            del self._sorted_sets[key][member]
            return 1
        return 0

    async def xadd(self, stream: str, fields: Dict[str, str]) -> str:
        if stream not in self._streams:
            self._streams[stream] = []

        # Generate unique ID
        import time
        timestamp = int(time.time() * 1000)
        self._id_counter += 1
        entry_id = f"{timestamp}-{self._id_counter}"

        self._streams[stream].append(StreamEntry(id=entry_id, data=fields))
        return entry_id

    def pipeline(self):
        return MockPipeline(self)


class MockPipeline:
    """Mock Redis pipeline for batch operations."""

    def __init__(self, client: InMemoryRedisClient):
        self._client = client
        self._commands: List[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    def incr(self, key: str):
        self._commands.append(('incr', key))

    def expire(self, key: str, seconds: int):
        self._commands.append(('expire', key, seconds))

    async def execute(self) -> List[Any]:
        results = []
        for cmd in self._commands:
            if cmd[0] == 'incr':
                key = cmd[1]
                current = self._client._data.get(key, '0')
                new_val = int(current) + 1
                self._client._data[key] = str(new_val)
                results.append(new_val)
            elif cmd[0] == 'expire':
                import time
                key, seconds = cmd[1], cmd[2]
                self._client._ttls[key] = int(time.time()) + seconds
                results.append(1)
        self._commands.clear()
        return results

    async def aclose(self):
        pass


class MockAsyncRedisStorage:
    """
    Mock implementation of AsyncRedisStorage.

    Provides full implementation of all public methods:
    - store_memory()
    - retrieve_memory()
    - batch_retrieve()
    - delete_memory()
    - get_eviction_candidates()
    - update_ltp()
    - publish_event()
    - check_health()
    - close()

    Uses in-memory dictionaries for storage, enabling full test isolation.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        stream_key: str = "haim:subconscious",
        max_connections: int = 10,
        socket_timeout: int = 5,
        password: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        """Initialize mock storage with configuration matching real AsyncRedisStorage."""
        self.stream_key = stream_key
        self.url = url

        # Use provided client or create in-memory client
        if client:
            self.redis_client = client
        else:
            self.redis_client = InMemoryRedisClient(decode_responses=True)

    async def close(self):
        """Close the client connection."""
        if hasattr(self.redis_client, 'aclose'):
            await self.redis_client.aclose()
        elif hasattr(self.redis_client, 'close'):
            await self.redis_client.close()

    # --- CRUD Operations ---

    async def store_memory(self, node_id: str, data: Dict[str, Any], ttl: Optional[int] = None):
        """
        Store memory metadata in Redis (Key-Value) + Update LTP Index.
        """
        key = f"haim:memory:{node_id}"
        payload = json.dumps(data, default=str)

        if ttl:
            await self.redis_client.setex(key, ttl, payload)
        else:
            await self.redis_client.set(key, payload)

        # Update LTP Index (Sorted Set)
        ltp = float(data.get("ltp_strength", 0.0))
        await self.redis_client.zadd("haim:ltp_index", {node_id: ltp})

    async def retrieve_memory(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve memory metadata by ID.

        Returns:
            Dict with memory data if found, None if not found.
        """
        key = f"haim:memory:{node_id}"
        data = await self.redis_client.get(key)

        if data:
            return json.loads(data)
        return None

    async def batch_retrieve(self, node_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        Batch retrieve multiple memories using MGET.

        Returns:
            List of dicts (or None for not found entries).
        """
        if not node_ids:
            return []

        keys = [f"haim:memory:{mid}" for mid in node_ids]
        results = await self.redis_client.mget(keys)

        parsed = []
        for r in results:
            if r:
                try:
                    parsed.append(json.loads(r))
                except json.JSONDecodeError:
                    parsed.append(None)
            else:
                parsed.append(None)
        return parsed

    async def delete_memory(self, node_id: str):
        """
        Delete memory from storage and index.
        """
        key = f"haim:memory:{node_id}"
        await self.redis_client.delete(key)
        await self.redis_client.zrem("haim:ltp_index", node_id)

    # --- Index/LTP Operations ---

    async def get_eviction_candidates(self, count: int = 10) -> List[str]:
        """
        Get IDs of memories with the lowest LTP scores.

        Returns:
            List of node IDs (empty list if none found).
        """
        # ZRANGE 0 (count-1) returns lowest scores
        members = await self.redis_client.zrange("haim:ltp_index", 0, count - 1)
        return members

    async def update_ltp(self, node_id: str, new_ltp: float):
        """
        Update just the LTP score in the index.
        """
        await self.redis_client.zadd("haim:ltp_index", {node_id: new_ltp})

    # --- Streaming (Subconscious Bus) ---

    async def publish_event(self, event_type: str, payload: Dict[str, Any]):
        """
        Publish an event to the Subconscious Bus (Redis Stream).
        """
        # XADD expects flat dict of strings
        msg = {"type": event_type}
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                msg[k] = json.dumps(v)
            else:
                msg[k] = str(v)

        await self.redis_client.xadd(self.stream_key, msg)

    async def check_health(self) -> bool:
        """Ping Redis to check connectivity."""
        try:
            return await self.redis_client.ping()
        except Exception:
            return False

    # --- Utility methods for testing ---

    def _get_stored_keys(self) -> List[str]:
        """Get all stored memory keys (for testing assertions)."""
        if isinstance(self.redis_client, InMemoryRedisClient):
            return [k for k in self.redis_client._data.keys() if k.startswith("haim:memory:")]
        return []

    def _get_ltp_index(self) -> Dict[str, float]:
        """Get LTP index contents (for testing assertions)."""
        if isinstance(self.redis_client, InMemoryRedisClient):
            return dict(self.redis_client._sorted_sets.get("haim:ltp_index", {}))
        return {}

    def _get_stream_events(self) -> List[Dict[str, Any]]:
        """Get all stream events (for testing assertions)."""
        if isinstance(self.redis_client, InMemoryRedisClient):
            events = self.redis_client._streams.get(self.stream_key, [])
            return [{"id": e.id, "data": e.data} for e in events]
        return []

    def _clear_all(self):
        """Clear all data (for test cleanup)."""
        if isinstance(self.redis_client, InMemoryRedisClient):
            self.redis_client._data.clear()
            self.redis_client._ttls.clear()
            self.redis_client._sorted_sets.clear()
            self.redis_client._streams.clear()


# Factory function to create appropriate mock based on available dependencies
def create_mock_redis_storage(
    url: str = "redis://localhost:6379/0",
    stream_key: str = "haim:subconscious",
    **kwargs
) -> MockAsyncRedisStorage:
    """
    Create a mock Redis storage instance.

    Uses fakeredis if available, otherwise falls back to in-memory dict.
    """
    if HAS_FAKEREDIS:
        try:
            fake_client = fakeredis.FakeRedis(decode_responses=True)
            return MockAsyncRedisStorage(
                url=url,
                stream_key=stream_key,
                client=fake_client,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"[MockRedis] Failed to create fakeredis client: {e}, using in-memory")

    return MockAsyncRedisStorage(
        url=url,
        stream_key=stream_key,
        **kwargs
    )
