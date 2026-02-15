import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ConceptSequenceDataset(Dataset):
    """
    Generates synthetic 'Concept Sequences' for training Omega-JEPA.
    A sequence consists of (State_t, Action_t, State_t+1).
    The transition follows a hidden causal rule.
    """
    def __init__(
        self, 
        num_samples: int = 10000, 
        embedding_dim: int = 256, 
        action_dim: int = 64,
        noise_level: float = 0.05
    ):
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.noise_level = noise_level
        
        # Hidden Causal Rule Parameters
        # We'll use a fixed random projection to simulate a complex but deterministic rule
        self.W_state = torch.randn(embedding_dim, embedding_dim) * 0.1
        self.W_action = torch.randn(action_dim, embedding_dim) * 0.1
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 1. Random initial state
        state_t = torch.randn(self.embedding_dim)
        
        # 2. Random action
        action_t = torch.randn(self.action_dim)
        
        # 3. Apply hidden causal rule: state_{t+1} = Tanh(state_t @ W_s + action_t @ W_a) + noise
        with torch.no_grad():
            # Synthetic transition logic
            transition = torch.tanh(
                state_t @ self.W_state + action_t @ self.W_action
            )
            noise = torch.randn(self.embedding_dim) * self.noise_level
            state_t1 = transition + noise
            
        return state_t, action_t, state_t1

def get_dataloader(batch_size=64, num_samples=10000, embedding_dim=256, action_dim=64):
    dataset = ConceptSequenceDataset(
        num_samples=num_samples, 
        embedding_dim=embedding_dim, 
        action_dim=action_dim
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Test the generator
    loader = get_dataloader(batch_size=4)
    s_t, a_t, s_t1 = next(iter(loader))
    print(f"State shape: {s_t.shape}")
    print(f"Action shape: {a_t.shape}")
    print(f"Next State shape: {s_t1.shape}")
