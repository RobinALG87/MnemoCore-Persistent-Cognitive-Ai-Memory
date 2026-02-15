"""
Async Redis Storage Layer
=========================
Provides non-blocking access to Redis for high-performance memory metadata storage,
LTP indexing, and event streaming (Subconscious Bus).

Uses `redis.asyncio` for native asyncio support.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from .config import get_config

logger = logging.getLogger(__name__)


class AsyncRedisStorage:
    """
    Wrapper for Async Redis client with connection pooling.
    Can be used as a Singleton via `get_instance()`.
    """
    _instance = None
    _pool: Optional[ConnectionPool] = None

    def __init__(self, client: Optional[redis.Redis] = None):
        """
        Initialize with optional explicit client (for testing/DI).
        If client is None, uses the shared connection pool.
        """
        self.config = get_config()
        if client:
            self.redis_client = client
        else:
            self._initialize_from_pool()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _initialize_from_pool(self):
        """Initialize Redis client from connection pool."""
        # Use class-level pool to share connections if multiple instances are created accidentally
        if AsyncRedisStorage._pool is None:
            logger.info(f"Initializing Async Redis Pool: {self.config.redis.url}")
            AsyncRedisStorage._pool = ConnectionPool.from_url(
                self.config.redis.url,
                max_connections=self.config.redis.max_connections,
                socket_timeout=self.config.redis.socket_timeout,
                decode_responses=True  # Auto-decode bytes to str
            )
        
        self.redis_client = redis.Redis(connection_pool=AsyncRedisStorage._pool)

    async def close(self):
        """Close the client connection."""
        if self.redis_client:
            await self.redis_client.aclose()
            
    # --- CRUD Operations ---

    async def store_memory(self, node_id: str, data: Dict[str, Any], ttl: Optional[int] = None):
        """
        Store memory metadata in Redis (Key-Value) + Update LTP Index.
        """
        try:
            key = f"haim:memory:{node_id}"
            # Serialize
            payload = json.dumps(data, default=str)
            
            if ttl:
                await self.redis_client.setex(key, ttl, payload)
            else:
                await self.redis_client.set(key, payload)

            # Update LTP Index (Sorted Set)
            ltp = float(data.get("ltp_strength", 0.0))
            await self.redis_client.zadd("haim:ltp_index", {node_id: ltp})

        except Exception as e:
            logger.error(f"AsyncRedis store failed for {node_id}: {e}")
            raise

    async def retrieve_memory(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory metadata by ID."""
        try:
            key = f"haim:memory:{node_id}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"AsyncRedis retrieve failed for {node_id}: {e}")
            return None

    async def batch_retrieve(self, node_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Batch retrieve multiple memories using MGET."""
        if not node_ids:
            return []
            
        keys = [f"haim:memory:{mid}" for mid in node_ids]
        try:
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
        except Exception as e:
            logger.error(f"AsyncRedis batch retrieve failed: {e}")
            return [None] * len(node_ids)

    async def delete_memory(self, node_id: str):
        """Delete memory from storage and index."""
        try:
            key = f"haim:memory:{node_id}"
            await self.redis_client.delete(key)
            await self.redis_client.zrem("haim:ltp_index", node_id)
        except Exception as e:
            logger.error(f"AsyncRedis delete failed for {node_id}: {e}")

    # --- Index/LTP Operations ---

    async def get_eviction_candidates(self, count: int = 10) -> List[str]:
        """
        Get IDs of memories with the lowest LTP scores.
        Usage: Consolidation worker calling this to find what to move to COLD.
        """
        try:
            # ZRANGE 0 (count-1) returns lowest scores
            members = await self.redis_client.zrange("haim:ltp_index", 0, count - 1)
            return members
        except Exception as e:
            logger.error(f"AsyncRedis eviction scan failed: {e}")
            return []
            
    async def update_ltp(self, node_id: str, new_ltp: float):
        """Update just the LTP score in the index."""
        try:
             await self.redis_client.zadd("haim:ltp_index", {node_id: new_ltp})
        except Exception as e:
            logger.error(f"AsyncRedis LTP update failed: {e}")

    # --- Streaming (Subconscious Bus) ---

    async def publish_event(self, event_type: str, payload: Dict[str, Any]):
        """
        Publish an event to the Subconscious Bus (Redis Stream).
        Phase 3.5.3 will consume these.
        """
        stream_key = self.config.redis.stream_key
        try:
            # XADD expects flat dict of strings
            msg = {"type": event_type}
            for k, v in payload.items():
                if isinstance(v, (dict, list)):
                    msg[k] = json.dumps(v)
                else:
                    msg[k] = str(v)
            
            await self.redis_client.xadd(stream_key, msg)
        except Exception as e:
            logger.error(f"AsyncRedis publish failed: {e}")

    async def check_health(self) -> bool:
        """Ping Redis to check connectivity."""
        try:
            return await self.redis_client.ping()
        except Exception:
            return False
