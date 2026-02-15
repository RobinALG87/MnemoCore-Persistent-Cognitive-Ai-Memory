import json
import torch
import numpy as np
import faiss
from core.encoder import WorldEncoder
from datetime import datetime

def load_and_sort_memories(path):
    memories = []
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
    print("Initializing WorldEncoder...")
    encoder = WorldEncoder()
    
    print("Loading Memories...")
    import os
    # Dynamically find the project root (assuming we run from workspace root usually, or haim root)
    # The script is in haim/vector_core/dream_stream/
    # memory.jsonl is in haim/data/
    
    # Let's try absolute path relative to workspace root
    possible_paths = [
        "haim/data/memory.jsonl",
        "../data/memory.jsonl",
        "../../data/memory.jsonl",
        "../../../data/memory.jsonl",
        "./data/memory.jsonl"
    ]
    
    mem_path = None
    for p in possible_paths:
        if os.path.exists(p):
            mem_path = p
            break
            
    if not mem_path:
        raise FileNotFoundError("Could not find memory.jsonl")
        
    memories = load_and_sort_memories(mem_path)
    print(f"Found {len(memories)} sequential memories.")
    
    texts = [m['content'] for m in memories]
    # Use metadata 'type' as the 'Action' context
    actions = [m.get('metadata', {}).get('type', 'unknown') for m in memories]
    
    print("Encoding Reality (this may take a moment)...")
    embeddings = encoder.encode_batch(texts) # [N, 384]
    action_embeddings = encoder.encode_batch(actions) # [N, 384]
    
    # Build FAISS Index
    print("Building FAISS Index...")
    index = faiss.IndexFlatL2(384)
    index.add(embeddings.numpy())
    
    # Save Index
    faiss.write_index(index, "reality.faiss")
    print("Saved reality.faiss")
    
    # Create JEPA Training Dataset (State, Action, Next State)
    # S_t = memories[i]
    # a_t = actions[i+1] (The action that LED to the next state? Or the type of the next state?)
    # Let's say: State_t + Action_Context -> State_t+1
    
    states = embeddings[:-1]
    next_states = embeddings[1:]
    # Action is the 'type' of the *next* memory (what happened)
    # or better: the transition. Let's use the 'type' of the next memory as the "Action" that occurred.
    action_vecs = action_embeddings[1:]
    
    dataset = {
        'states': states,
        'actions': action_vecs,
        'targets': next_states
    }
    
    torch.save(dataset, "reality_sequences.pt")
    print(f"Saved reality_sequences.pt with {len(states)} transitions.")

if __name__ == "__main__":
    ingest()
