import json
import torch
import numpy as np
import faiss
import os
import sys

# Ensure we can import core.encoder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core.encoder import WorldEncoder
except ImportError:
    # Fallback if structure is different
    sys.path.insert(0, "./vector_core/dream_stream")
    from core.encoder import WorldEncoder

from datetime import datetime

def load_and_sort_memories(path):
    memories = []
    print(f"Reading from {path}...")
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    mem = json.loads(line)
                    # Parse TS for sorting
                    ts = mem.get('created_at') or mem.get('metadata', {}).get('ts')
                    if ts:
                        mem['sort_ts'] = ts
                        memories.append(mem)
                except:
                    continue
    
    # Sort by time
    memories.sort(key=lambda x: x['sort_ts'])
    return memories

def ingest():
    print("--- Reality Ingestion Engine v2.0 ---")
    print(f"Time: {datetime.now().isoformat()}")
    
    print("Initializing WorldEncoder...")
    encoder = WorldEncoder()
    
    mem_path = "./data/memory.jsonl"
    if not os.path.exists(mem_path):
        raise FileNotFoundError(f"Critical Error: {mem_path} not found.")
        
    memories = load_and_sort_memories(mem_path)
    print(f"Found {len(memories)} sequential memories.")
    
    if not memories:
        print("No memories to process. Exiting.")
        return

    texts = [m['content'] for m in memories]
    # Use metadata 'type' as the 'Action' context
    actions = [m.get('metadata', {}).get('type', 'unknown') for m in memories]
    
    print(f"Encoding {len(texts)} memories (this may take a moment)...")
    embeddings = encoder.encode_batch(texts) # [N, 384]
    action_embeddings = encoder.encode_batch(actions) # [N, 384]
    
    # Build FAISS Index
    print("Building FAISS Index...")
    index = faiss.IndexFlatL2(384)
    index.add(embeddings.numpy())
    
    # Save Index - Use absolute path
    output_path = "./vector_core/dream_stream/reality.faiss"
    faiss.write_index(index, output_path)
    print(f"SUCCESS: Saved {output_path}")
    
    # Create JEPA Training Dataset (State, Action, Next State)
    states = embeddings[:-1]
    next_states = embeddings[1:]
    action_vecs = action_embeddings[1:]
    
    dataset = {
        'states': states,
        'actions': action_vecs,
        'targets': next_states
    }
    
    dataset_path = "./vector_core/dream_stream/reality_sequences.pt"
    torch.save(dataset, dataset_path)
    print(f"SUCCESS: Saved {dataset_path} with {len(states)} transitions.")

if __name__ == "__main__":
    try:
        ingest()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)
