import json
import time
import sys
import os

# Placeholder for FAISS/SentenceTransformer (to be installed)
# This script prepares the data for vectorization

def load_memories(memory_path):
    data = []
    # Handle JSONL format
    with open(memory_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(data)} memories.")
    return data

def prepare_corpus(memories):
    corpus = []
    ids = []
    for m in memories:
        # Combine relevant fields for embedding
        text = f"{m.get('content', '')} {m.get('context', '')} {' '.join(m.get('tags', []))}"
        corpus.append(text)
        ids.append(m.get('id'))
    return ids, corpus

if __name__ == "__main__":
    memory_file = "haim/data/memory.jsonl"
    if not os.path.exists(memory_file):
        print(f"Error: {memory_file} not found.")
        sys.exit(1)
        
    ids, corpus = prepare_corpus(load_memories(memory_file))
    print(f"Prepared {len(corpus)} text chunks for embedding.")
    
    # Save prepared corpus for the actual vectorization step
    with open("haim/vector_core/corpus_ready.json", "w") as f:
        json.dump({"ids": ids, "corpus": corpus}, f)
    print("Corpus saved to haim/vector_core/corpus_ready.json")
