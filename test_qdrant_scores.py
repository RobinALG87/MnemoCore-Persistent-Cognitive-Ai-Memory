
import asyncio
import numpy as np
from mnemocore.core.qdrant_store import QdrantStore
from mnemocore.core.config import get_config

async def test_qdrant_scores():
    config = get_config()
    store = QdrantStore(
        url=config.qdrant.url,
        api_key=None,
        dimensionality=config.dimensionality
    )
    
    print(f"Ensuring collections (Migration Check)...")
    await store.ensure_collections()
    
    print(f"Searching {config.qdrant.collection_warm}...")
    try:
        info = await store.get_collection_info(config.qdrant.collection_warm)
        print(f"Collection Info: {info}")
        
        # Get one point first to have a valid vector
        scroll_res = await store.scroll(config.qdrant.collection_warm, limit=1, with_vectors=True)
        points = scroll_res[0]
        
        if not points:
            print("No points found in collection.")
            return
            
        target_vec = points[0].vector
        target_id = points[0].id
        print(f"Target Point: ID={target_id}")
        
        # Test basic search without search_params
        response = await store.client.query_points(
            collection_name=config.qdrant.collection_warm,
            query=target_vec,
            limit=3
        )
        hits = response.points
        print(f"Basic Search Hits count: {len(hits)}")
        for i, hit in enumerate(hits):
            print(f"Hit {i}: ID={hit.id}, Score={hit.score}")
            
        hits = await store.search(config.qdrant.collection_warm, target_vec, limit=3)
        print(f"Store Search Hits count: {len(hits)}")
        for i, hit in enumerate(hits):
            print(f"Hit {i}: ID={hit.id}, Score={hit.score}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        await store.close()

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.getcwd(), "src"))
    asyncio.run(test_qdrant_scores())
