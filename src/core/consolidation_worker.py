"""
Consolidation Worker (Phase 3.5.3)
==================================
Subconscious bus consumer that:
1. Listens to 'memory.created' events for reactive processing.
2. Periodically triggers WARM -> COLD consolidation.
3. Generates insights via small LLM (System 2) - placeholder logic.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List

from .async_storage import AsyncRedisStorage
from .config import get_config
from .tier_manager import TierManager

logger = logging.getLogger(__name__)

class ConsolidationWorker:
    def __init__(self):
        self.config = get_config()
        self.storage = AsyncRedisStorage.get_instance()
        self.tier_manager = TierManager()
        self.running = False
        self.consumer_group = "haim_workers"
        self.consumer_name = f"worker_{int(time.time())}"

    async def setup_stream(self):
        """Ensure consumer group exists."""
        stream_key = self.config.redis.stream_key
        try:
            await self.storage.redis_client.xgroup_create(
                stream_key, self.consumer_group, id="0", mkstream=True
            )
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Failed to create consumer group: {e}")

    async def process_event(self, event_id: str, data: Dict[str, Any]):
        """Handle a single event from the bus."""
        event_type = data.get("type")
        logger.info(f"Processing event {event_id}: {event_type}")

        if event_type == "memory.created":
            # Reactive Logic: Check if memory needs immediate attention
            # For now, just log and ack
            mem_id = data.get("id")
            logger.info(f"New memory registered: {mem_id}")
            # Placeholder for future "System 2" triggers
        
        elif event_type == "memory.accessed":
            # Update access patterns, maybe promote if needed (handled by TierManager logic mostly)
            pass

    async def run_consolidation_cycle(self):
        """Execute periodic WARM -> COLD consolidation."""
        logger.info("Running WARM -> COLD consolidation cycle...")
        try:
            # Sync call in thread executor if blocking
            # But TierManager.consolidate_warm_to_cold is sync (file I/O + Qdrant sync)
            # asyncio.to_thread is good here
            await asyncio.to_thread(self.tier_manager.consolidate_warm_to_cold)
        except Exception as e:
            logger.error(f"Consolidation cycle failed: {e}")

    async def consume_loop(self):
        """Main event loop."""
        stream_key = self.config.redis.stream_key
        last_consolidation = time.time()
        consolidation_interval = 300  # 5 minutes

        while self.running:
            try:
                # 1. Read from stream
                streams = {stream_key: ">"}
                messages = await self.storage.redis_client.xreadgroup(
                    self.consumer_group, self.consumer_name, streams, count=10, block=1000
                )
                
                if messages:
                    for stream, event_list in messages:
                        for event_id, event_data in event_list:
                            # event_data is dict of bytes/strings
                            # Decode if necessary (redis-py handles decoding if decode_responses=True)
                            await self.process_event(event_id, event_data)
                            await self.storage.redis_client.xack(
                                stream_key, self.consumer_group, event_id
                            )

                # 2. Check periodic tasks
                if time.time() - last_consolidation > consolidation_interval:
                    await self.run_consolidation_cycle()
                    last_consolidation = time.time()

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(5) # Backoff logic placeholder

    async def start(self):
        self.running = True
        logger.info(f"Starting Consolidation Worker ({self.consumer_name})...")
        await self.setup_stream()
        await self.consume_loop()

    def stop(self):
        self.running = False
        logger.info("Stopping worker...")

if __name__ == "__main__":
    # Standalone entry point
    logging.basicConfig(level=logging.INFO)
    worker = ConsolidationWorker()
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        worker.stop()
