import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union

class OmegaMetrics:
    """
    The Auditor: Implements Protocol Omega validation metrics.
    Focuses on Time-Reversal Asymmetry (TRA) and statistical anomaly detection (Z-Score).
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the scalar 'energy' of a state representation.
        In this implementation, we use the L2 norm magnitude as a proxy for energy magnitude.
        
        Args:
            state: [Batch, Dim]
            
        Returns:
            Energy: [Batch]
        """
        return torch.norm(state, p=2, dim=-1)

    def compute_tra(self, state_t: torch.Tensor, state_t1: torch.Tensor) -> torch.Tensor:
        """
        Computes Time-Reversal Asymmetry (TRA).
        
        Hypothesis: In a causal flow, the energy transition E(t) -> E(t+1) 
        should preserve directional structure. TRA measures the violation of reversibility.
        
        TRA = |E(t+1) - E(t)|^2 / (E(t) + epsilon)
        
        (This is a simplified scalar metric for immediate feedback).
        """
        e_t = self.compute_energy(state_t)
        e_t1 = self.compute_energy(state_t1)
        
        diff = torch.abs(e_t1 - e_t)
        tra = (diff ** 2) / (e_t + self.epsilon)
        
        return tra

    def compute_z_score(self, error_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes Z-Score of the prediction errors within the current batch.
        High Z-scores indicate outliers or 'surprise' events that the JEPA failed to predict.
        
        Args:
            error_tensor: [Batch] scalar errors (e.g. MSE per sample)
        """
        mean = torch.mean(error_tensor)
        std = torch.std(error_tensor)
        
        z_scores = (error_tensor - mean) / (std + self.epsilon)
        return z_scores

    def validate_batch(
        self, 
        pred_state: torch.Tensor, 
        target_state: torch.Tensor, 
        prev_state: torch.Tensor
    ) -> Dict[str, float]:
        """
        Full validation suite for a training step.
        
        Args:
            pred_state: Output from Omega-JEPA [Batch, Dim]
            target_state: Ground truth representation [Batch, Dim]
            prev_state: The input context state [Batch, Dim] (for TRA calculation)
            
        Returns:
            Dictionary of aggregated metrics.
        """
        # 1. Reconstruction Loss (L2) - Proxy for 'prediction error' in latent space
        # Note: In pure JEPA, we maximize similarity, here we measure distance.
        mse_loss = F.mse_loss(pred_state, target_state, reduction='none').mean(dim=-1)
        avg_mse = mse_loss.mean().item()
        
        # 2. Time-Reversal Asymmetry (TRA)
        # Did the predicted transition violate energy conservation logic significantly?
        tra_scores = self.compute_tra(prev_state, pred_state)
        avg_tra = tra_scores.mean().item()
        
        # 3. Z-Score Analysis of the batch errors
        z_scores = self.compute_z_score(mse_loss)
        max_z = z_scores.max().item() # The biggest anomaly in the batch
        
        # 4. Omega Integrity Score
        # A synthetic score: Lower is better.
        # Penalizes high MSE and high TRA violation.
        omega_score = avg_mse * (1.0 + avg_tra)
        
        return {
            "loss_mse": avg_mse,
            "tra_index": avg_tra,
            "max_anomaly_z": max_z,
            "omega_integrity": omega_score
        }
