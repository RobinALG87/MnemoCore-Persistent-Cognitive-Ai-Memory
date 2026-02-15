import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from core.predictor import OmegaJEPA_Predictor
from core.omega_metrics import OmegaMetrics
import os

class RealityDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.states = data['states']
        self.actions = data['actions']
        self.targets = data['targets']
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.targets[idx]

def train_reality():
    # Config
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 5 # Rapid adaptation for testing
    
    # Check data
    if not os.path.exists("reality_sequences.pt"):
        print("Error: reality_sequences.pt not found. Run ingest_reality.py first.")
        return

    # Load Data
    dataset = RealityDataset("reality_sequences.pt")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded Reality: {len(dataset)} transitions.")

    # Init Model (384 dim for MiniLM)
    # Corrected args based on predictor.py definition: embedding_dim, action_dim, latent_dim
    model = OmegaJEPA_Predictor(embedding_dim=384, action_dim=384, latent_dim=384)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        total_tra = 0
        
        for s, a, target in dataloader:
            optimizer.zero_grad()
            
            # Forward
            pred = model(s, a)
            
            # Loss: MSE + TRA (Auxiliary)
            mse = nn.functional.mse_loss(pred, target)
            
            # TRA: We want High TRA (Irreversibility) for real events?
            # Actually, standard physics is reversible, but entropy/complexity is not.
            # In Omega V4, we look for TRA.
            # Here we just monitor it.
            
            loss = mse
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                # Fix: compute_tra needs (state_t, state_t1)
                # Create an instance of OmegaMetrics first
                metrics = OmegaMetrics()
                total_tra += metrics.compute_tra(s, pred).mean().item()
        
        avg_loss = total_loss / len(dataloader)
        avg_tra = total_tra / len(dataloader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f} | TRA={avg_tra:.4f}")

    # Save
    torch.save(model.state_dict(), "checkpoints/omega_jepa_reality.pt")
    print("Reality Model Saved: checkpoints/omega_jepa_reality.pt")

if __name__ == "__main__":
    train_reality()
