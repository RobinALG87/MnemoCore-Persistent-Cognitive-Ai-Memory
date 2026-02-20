"""
Tests for AsyncRedisStorage (Phase 3.5.1)
=========================================
Uses unittest.IsolatedAsyncioTestCase for robust async support without plugins.
"""

import json
import unittest
from unittest.mock import AsyncMock

from mnemocore.core.async_storage import AsyncRedisStorage


class TestAsyncStorage(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.mock_client = AsyncMock()
        self.storage = AsyncRedisStorage(client=self.mock_client)

    async def test_store_memory(self):
        node_id = "mem_123"
        data = {"content": "test", "ltp_strength": 0.5}

        await self.storage.store_memory(node_id, data)

        # Verify set
        self.mock_client.set.assert_called_once()
        args, _ = self.mock_client.set.call_args
        self.assertEqual(args[0], f"haim:memory:{node_id}")
        self.assertEqual(json.loads(args[1])["content"], "test")

        # Verify zadd
        self.mock_client.zadd.assert_called_once_with("haim:ltp_index", {node_id: 0.5})

    async def test_retrieve_memory(self):
        node_id = "mem_456"
        mock_data = {"id": node_id, "content": "retrieved"}
        self.mock_client.get.return_value = json.dumps(mock_data)

        result = await self.storage.retrieve_memory(node_id)

        self.assertEqual(result, mock_data)
        self.mock_client.get.assert_called_once_with(f"haim:memory:{node_id}")

    async def test_batch_retrieve(self):
        self.mock_client.mget.return_value = [
            json.dumps({"id": "1"}),
            None,
            json.dumps({"id": "3"}),
        ]

        results = await self.storage.batch_retrieve(["1", "2", "3"])

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], "1")
        self.assertIsNone(results[1])
        self.assertEqual(results[2]["id"], "3")

    async def test_publish_event(self):
        event_type = "test.event"
        payload = {"foo": "bar"}

        await self.storage.publish_event(event_type, payload)

        self.mock_client.xadd.assert_called_once()
        args, _ = self.mock_client.xadd.call_args
        self.assertEqual(args[0], "haim:subconscious")
        self.assertEqual(args[1]["type"], event_type)

    async def test_eviction_candidates(self):
        self.mock_client.zrange.return_value = ["mem_A"]

        result = await self.storage.get_eviction_candidates(count=5)

        self.assertEqual(result, ["mem_A"])
