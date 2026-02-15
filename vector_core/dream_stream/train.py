import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add core to path to import components
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from predictor import OmegaJEPA_Predictor
from omega_metrics import OmegaMetrics
from mock_data import get_dataloader

def train():
    # 1. Hyperparameters
    embedding_dim = 256
    action_dim = 64
    latent_dim = 64
    batch_size = 64
    epochs = 10
    lr = 1e-4
    alpha = 0.1 # Weight for auxiliary anomaly loss (TRA)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 2. Initialize Model, Metrics, and Data
    model = OmegaJEPA_Predictor(
        embedding_dim=embedding_dim,
        action_dim=action_dim,
        latent_dim=latent_dim
    ).to(device)
    
    metrics_auditor = OmegaMetrics()
    
    dataloader = get_dataloader(
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        action_dim=action_dim
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        epoch_tras = []
        
        for batch_idx, (s_t, a_t, s_t1) in enumerate(dataloader):
            s_t, a_t, s_t1 = s_t.to(device), a_t.to(device), s_t1.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            # In training, we can either sample z or use deterministic (zero)
            # For simplicity, we use z=None (deterministic) to learn the mean transition
            pred_s_t1 = model(s_t, a_t)
            
            # Loss Components
            # a) L2 Distance (Primary Prediction Loss)
            mse_loss = torch.mean((pred_s_t1 - s_t1)**2)
            
            # b) Auxiliary Loss: Omega Anomaly Score (TRA)
            # We want to minimize TRA to encourage organic transitions that respect energy flow
            tra_loss = metrics_auditor.compute_tra(s_t, pred_s_t1).mean()
            
            # Total Loss
            total_loss = mse_loss + (alpha * tra_loss)
            
            # Backward Pass
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(mse_loss.item())
            epoch_tras.append(tra_loss.item())
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"MSE: {mse_loss.item():.6f} | TRA: {tra_loss.item():.6f}")
                
        avg_mse = sum(epoch_losses) / len(epoch_losses)
        avg_tra = sum(epoch_tras) / len(epoch_tras)
        print(f"==> Epoch {epoch} Complete | Avg MSE: {avg_mse:.6f} | Avg TRA: {avg_tra:.6f}")

    # 4. Save Model
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "omega_jepa_latest.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Training finished and model saved to {save_path}.")

if __name__ == "__main__":
    train()
