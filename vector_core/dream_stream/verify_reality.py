import faiss
import torch
import os

def check():
    faiss_path = "./vector_core/dream_stream/reality.faiss"
    pt_path = "./vector_core/dream_stream/reality_sequences.pt"
    
    print(f"Checking {faiss_path}...")
    index = faiss.read_index(faiss_path)
    print(f"Index loaded. Total vectors: {index.ntotal}")
    
    print(f"Checking {pt_path}...")
    dataset = torch.load(pt_path)
    print(f"Dataset loaded. Keys: {list(dataset.keys())}")
    print(f"State tensor shape: {dataset['states'].shape}")
    
    if index.ntotal > 0 and 'states' in dataset:
        print("VERIFICATION SUCCESSFUL")
    else:
        print("VERIFICATION FAILED")

if __name__ == "__main__":
    check()
