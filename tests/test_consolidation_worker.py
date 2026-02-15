"""
Tests for Consolidation Worker (Phase 3.5.3)
===========================================
Verify event consumption and consolidation logic using unittest.mock.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from src.core.consolidation_worker import ConsolidationWorker

class TestConsolidationWorker(unittest.IsolatedAsyncioTestCase):
    
    async def asyncSetUp(self):
        # Patch dependencies
        self.storage_patcher = patch('src.core.consolidation_worker.AsyncRedisStorage')
        self.tier_manager_patcher = patch('src.core.consolidation_worker.TierManager')
        self.config_patcher = patch('src.core.consolidation_worker.get_config')
        
        self.MockStorage = self.storage_patcher.start()
        self.MockTierManager = self.tier_manager_patcher.start()
        self.mock_config = self.config_patcher.start()
        
        # Setup mock storage instance
        self.mock_storage_instance = MagicMock()
        self.mock_storage_instance.redis_client = AsyncMock()
        self.MockStorage.get_instance.return_value = self.mock_storage_instance
        
        self.worker = ConsolidationWorker()

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
        # Difficult to test infinite loop, but can check xreadgroup call logic
        # Mock xreadgroup to return one event then raise CancelledError or stop
        self.mock_storage_instance.redis_client.xreadgroup.return_value = [
            ("stream_key", [("evt_1", {"type": "memory.created", "id": "mem_1"})])
        ]
        
        # Run one iteration manually logic-wise?
        # Or patch running=False after one iter
        
        # Simulating one pass:
        self.worker.running = True
        task = asyncio.create_task(self.worker.consume_loop())
        await asyncio.sleep(0.1)
        self.worker.running = False
        await task
        
        self.mock_storage_instance.redis_client.xreadgroup.assert_called()
        self.mock_storage_instance.redis_client.xack.assert_called_with(
            self.worker.config.redis.stream_key, 
            self.worker.consumer_group, 
            "evt_1"
        )


if __name__ == '__main__':
    unittest.main()
