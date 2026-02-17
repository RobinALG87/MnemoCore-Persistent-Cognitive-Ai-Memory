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
from .reliability import StorageCircuitBreaker

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

            kwargs = {
                "max_connections": self.config.redis.max_connections,
                "socket_timeout": self.config.redis.socket_timeout,
                "decode_responses": True,
            }
            if self.config.redis.password:
                kwargs["password"] = self.config.redis.password

            AsyncRedisStorage._pool = ConnectionPool.from_url(
                self.config.redis.url,
                **kwargs
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
        breaker = StorageCircuitBreaker.get_redis_breaker()
        try:
            await breaker.call(self._store_memory, node_id, data, ttl)
        except Exception:
            logger.error(f"AsyncRedis store blocked or failed for {node_id}")
            raise

    async def _store_memory(self, node_id: str, data: Dict[str, Any], ttl: Optional[int] = None):
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

    async def retrieve_memory(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory metadata by ID."""
        breaker = StorageCircuitBreaker.get_redis_breaker()
        try:
            key = f"haim:memory:{node_id}"
            data = await breaker.call(self.redis_client.get, key)
            if data:
                return json.loads(data)
            return None
        except Exception:
            logger.error(f"AsyncRedis retrieve blocked or failed for {node_id}")
            return None

    async def batch_retrieve(self, node_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Batch retrieve multiple memories using MGET."""
        if not node_ids:
            return []
            
        breaker = StorageCircuitBreaker.get_redis_breaker()
        keys = [f"haim:memory:{mid}" for mid in node_ids]
        try:
            results = await breaker.call(self.redis_client.mget, keys)
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
        except Exception:
            logger.error("AsyncRedis batch retrieve blocked or failed")
            return [None] * len(node_ids)

    async def delete_memory(self, node_id: str):
        """Delete memory from storage and index."""
        breaker = StorageCircuitBreaker.get_redis_breaker()
        try:
            key = f"haim:memory:{node_id}"
            await breaker.call(self.redis_client.delete, key)
            await breaker.call(self.redis_client.zrem, "haim:ltp_index", node_id)
        except Exception:
            logger.error(f"AsyncRedis delete blocked or failed for {node_id}")

    # --- Index/LTP Operations ---

    async def get_eviction_candidates(self, count: int = 10) -> List[str]:
        """
        Get IDs of memories with the lowest LTP scores.
        Usage: Consolidation worker calling this to find what to move to COLD.
        """
        breaker = StorageCircuitBreaker.get_redis_breaker()
        try:
            # ZRANGE 0 (count-1) returns lowest scores
            members = await breaker.call(self.redis_client.zrange, "haim:ltp_index", 0, count - 1)
            return members
        except Exception:
            logger.error("AsyncRedis eviction scan blocked or failed")
            return []
            
    async def update_ltp(self, node_id: str, new_ltp: float):
        """Update just the LTP score in the index."""
        breaker = StorageCircuitBreaker.get_redis_breaker()
        try:
             await breaker.call(self.redis_client.zadd, "haim:ltp_index", {node_id: new_ltp})
        except Exception:
            logger.error("AsyncRedis LTP update blocked or failed")

    # --- Streaming (Subconscious Bus) ---

    async def publish_event(self, event_type: str, payload: Dict[str, Any]):
        """
        Publish an event to the Subconscious Bus (Redis Stream).
        Phase 3.5.3 will consume these.
        """
        breaker = StorageCircuitBreaker.get_redis_breaker()
        stream_key = self.config.redis.stream_key
        try:
            # XADD expects flat dict of strings
            msg = {"type": event_type}
            for k, v in payload.items():
                if isinstance(v, (dict, list)):
                    msg[k] = json.dumps(v)
                else:
                    msg[k] = str(v)
            
            await breaker.call(self.redis_client.xadd, stream_key, msg)
        except Exception:
            logger.error("AsyncRedis publish blocked or failed")

    async def check_health(self) -> bool:
        """Ping Redis to check connectivity."""
        try:
            return await self.redis_client.ping()
        except Exception:
            return False
