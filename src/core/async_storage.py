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
from .exceptions import (
    StorageError,
    StorageConnectionError,
    DataCorruptionError,
    CircuitOpenError,
    wrap_storage_exception,
)

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
        """
        Retrieve memory metadata by ID.

        Returns:
            Dict with memory data if found, None if not found.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            StorageConnectionError: If Redis connection fails.
            DataCorruptionError: If stored data cannot be deserialized.
        """
        breaker = StorageCircuitBreaker.get_redis_breaker()
        key = f"haim:memory:{node_id}"

        try:
            data = await breaker.call(self.redis_client.get, key)
            if data:
                try:
                    return json.loads(data)
                except json.JSONDecodeError as e:
                    raise DataCorruptionError(
                        resource_id=node_id,
                        reason=f"Invalid JSON data: {e}",
                        context={"key": key}
                    )
            return None  # Not found - expected case, not an error
        except CircuitOpenError:
            # Re-raise circuit breaker errors directly
            logger.error(f"AsyncRedis retrieve blocked for {node_id}: circuit breaker open")
            raise
        except DataCorruptionError:
            # Re-raise data corruption errors directly
            raise
        except Exception as e:
            # Wrap other exceptions in StorageConnectionError
            logger.error(f"AsyncRedis retrieve failed for {node_id}: {e}")
            raise wrap_storage_exception("redis", "retrieve", e)

    async def batch_retrieve(self, node_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        Batch retrieve multiple memories using MGET.

        Returns:
            List of dicts (or None for not found/corrupt entries).

        Raises:
            CircuitOpenError: If circuit breaker is open.
            StorageConnectionError: If Redis connection fails.
        """
        if not node_ids:
            return []

        breaker = StorageCircuitBreaker.get_redis_breaker()
        keys = [f"haim:memory:{mid}" for mid in node_ids]

        try:
            results = await breaker.call(self.redis_client.mget, keys)
            parsed = []
            for i, r in enumerate(results):
                if r:
                    try:
                        parsed.append(json.loads(r))
                    except json.JSONDecodeError as e:
                        # Log corruption but don't fail the whole batch
                        logger.warning(f"Corrupt JSON for {node_ids[i]}: {e}")
                        parsed.append(None)
                else:
                    parsed.append(None)
            return parsed
        except CircuitOpenError:
            logger.error("AsyncRedis batch retrieve blocked: circuit breaker open")
            raise
        except Exception as e:
            logger.error(f"AsyncRedis batch retrieve failed: {e}")
            raise wrap_storage_exception("redis", "batch_retrieve", e)

    async def delete_memory(self, node_id: str):
        """
        Delete memory from storage and index.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            StorageConnectionError: If Redis connection fails.
        """
        breaker = StorageCircuitBreaker.get_redis_breaker()
        try:
            key = f"haim:memory:{node_id}"
            await breaker.call(self.redis_client.delete, key)
            await breaker.call(self.redis_client.zrem, "haim:ltp_index", node_id)
        except CircuitOpenError:
            logger.error(f"AsyncRedis delete blocked for {node_id}: circuit breaker open")
            raise
        except Exception as e:
            logger.error(f"AsyncRedis delete failed for {node_id}: {e}")
            raise wrap_storage_exception("redis", "delete", e)

    # --- Index/LTP Operations ---

    async def get_eviction_candidates(self, count: int = 10) -> List[str]:
        """
        Get IDs of memories with the lowest LTP scores.
        Usage: Consolidation worker calling this to find what to move to COLD.

        Returns:
            List of node IDs (empty list if none found or on non-critical errors).

        Note:
            This method returns an empty list on errors rather than raising,
            as eviction is a background operation that should not crash the system.
        """
        breaker = StorageCircuitBreaker.get_redis_breaker()
        try:
            # ZRANGE 0 (count-1) returns lowest scores
            members = await breaker.call(self.redis_client.zrange, "haim:ltp_index", 0, count - 1)
            return members
        except CircuitOpenError:
            logger.warning("AsyncRedis eviction scan blocked: circuit breaker open")
            return []
        except Exception as e:
            logger.error(f"AsyncRedis eviction scan failed: {e}")
            return []

    async def update_ltp(self, node_id: str, new_ltp: float):
        """
        Update just the LTP score in the index.

        Raises:
            CircuitOpenError: If circuit breaker is open.
            StorageConnectionError: If Redis connection fails.
        """
        breaker = StorageCircuitBreaker.get_redis_breaker()
        try:
            await breaker.call(self.redis_client.zadd, "haim:ltp_index", {node_id: new_ltp})
        except CircuitOpenError:
            logger.error(f"AsyncRedis LTP update blocked for {node_id}: circuit breaker open")
            raise
        except Exception as e:
            logger.error(f"AsyncRedis LTP update failed for {node_id}: {e}")
            raise wrap_storage_exception("redis", "update_ltp", e)

    # --- Streaming (Subconscious Bus) ---

    async def publish_event(self, event_type: str, payload: Dict[str, Any]):
        """
        Publish an event to the Subconscious Bus (Redis Stream).
        Phase 3.5.3 will consume these.

        Note:
            This method logs errors but does not raise, as event publishing
            is a fire-and-forget operation that should not block the caller.
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
        except CircuitOpenError:
            logger.warning(f"AsyncRedis publish blocked for {event_type}: circuit breaker open")
        except Exception as e:
            logger.error(f"AsyncRedis publish failed for {event_type}: {e}")

    async def check_health(self) -> bool:
        """Ping Redis to check connectivity."""
        try:
            return await self.redis_client.ping()
        except Exception:
            return False
