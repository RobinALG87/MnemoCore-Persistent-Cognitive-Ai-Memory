
import asyncio
import json
import os
from mnemocore.core.engine import HAIMEngine
from mnemocore.core.config import get_config
from mnemocore.core.container import build_container
from mnemocore.core.tier_manager import TierManager

async def sync_qdrant():
    config = get_config()
    container = build_container(config)
    
    # Initialize TierManager with Qdrant
    tier_manager = TierManager(config=config, qdrant_store=container.qdrant_store)
    
    engine = HAIMEngine(
        config=config,
        tier_manager=tier_manager,
        working_memory=container.working_memory,
        episodic_store=container.episodic_store,
        semantic_store=container.semantic_store
    )
    await engine.initialize()
    
    print(f"Engine initialized. Memories in HOT: {len(engine.tier_manager.hot)}")
    
    # Force sync from memory.jsonl if HOT is empty
    if len(engine.tier_manager.hot) == 0:
        print("HOT is empty, reloading from legacy log...")
        await engine._load_legacy_if_needed()
        print(f"Memories after reload: {len(engine.tier_manager.hot)}")
    
    # Consolidation will move them to WARM (Qdrant)
    # But we can also just call _save_to_warm manually for all nodes
    print("Syncing nodes to Qdrant...")
    count = 0
    for node_id, node in list(engine.tier_manager.hot.items()):
        await engine.tier_manager._save_to_warm(node)
        count += 1
        if count % 100 == 0:
            print(f"Synced {count} nodes...")
            
    print(f"Total synced: {count}")
    await engine.close()

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.getcwd(), "src"))
    asyncio.run(sync_qdrant())
