"""
Tests for Consolidation Worker (Phase 3.5.3)
===========================================
Verify event consumption and consolidation logic using unittest.mock.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from mnemocore.core.consolidation_worker import ConsolidationWorker


class TestConsolidationWorker(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Patch dependencies
        self.storage_patcher = patch(
            "mnemocore.core.consolidation_worker.AsyncRedisStorage"
        )
        self.tier_manager_patcher = patch(
            "mnemocore.core.consolidation_worker.TierManager"
        )
        self.config_patcher = patch("mnemocore.core.consolidation_worker.get_config")

        self.MockStorage = self.storage_patcher.start()
        self.MockTierManager = self.tier_manager_patcher.start()
        self.mock_config = self.config_patcher.start()

        # Setup mock storage instance
        self.mock_storage_instance = MagicMock()
        self.mock_storage_instance.redis_client = AsyncMock()
        self.MockStorage.return_value = self.mock_storage_instance

        self.worker = ConsolidationWorker(storage=self.mock_storage_instance)

    async def asyncTearDown(self):
        self.storage_patcher.stop()
        self.tier_manager_patcher.stop()
        self.config_patcher.stop()

    async def test_setup_stream(self):
        await self.worker.setup_stream()
        self.mock_storage_instance.redis_client.xgroup_create.assert_called_once()

    async def test_process_event_created(self):
        event_data = {"type": "memory.created", "id": "mem_1"}
        await self.worker.process_event("evt_1", event_data)
        # Currently just logs, verify no exceptions

    async def test_run_consolidation_cycle(self):
        await self.worker.run_consolidation_cycle()
        # Should call tier_manager.consolidate_warm_to_cold in a thread
        # Verify TierManager instance called
        self.worker.tier_manager.consolidate_warm_to_cold.assert_called_once()

    async def test_consume_loop_logic(self):
        # Make xreadgroup return one event then block indefinitely (return empty)
        call_count = 0

        async def mock_xreadgroup(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    (
                        "stream_key",
                        [("evt_1", {"type": "memory.created", "id": "mem_1"})],
                    )
                ]
            # Second call: signal stop
            self.worker.running = False
            return []

        self.mock_storage_instance.redis_client.xreadgroup = mock_xreadgroup
        self.mock_storage_instance.redis_client.xack = AsyncMock()

        self.worker.running = True
        try:
            await asyncio.wait_for(self.worker.consume_loop(), timeout=2.0)
        except asyncio.TimeoutError:
            self.worker.running = False

        self.assertGreaterEqual(call_count, 1)


if __name__ == "__main__":
    unittest.main()
