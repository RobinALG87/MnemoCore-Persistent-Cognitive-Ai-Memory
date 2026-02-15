import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class OmegaJEPA_Predictor(nn.Module):
    """
    Omega-JEPA Predictor Network (Dream Stream Implementation).
    
    A deterministic-compatible predictor that maps (Context, Action, Latent) -> Predicted State.
    Designed for clean-room adherence to JEPA principles while allowing Protocol Omega
    metric integration.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 256, 
        action_dim: int = 64, 
        latent_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            embedding_dim: Dimension of state representations (Sx, Sy).
            action_dim: Dimension of action vector (a).
            latent_dim: Dimension of latent variable (z).
            hidden_dim: Width of internal MLP layers.
            num_layers: Number of hidden residual blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Input projection: Concatenates [Sx, a, z] -> Hidden
        self.input_dim = embedding_dim + action_dim + latent_dim
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        
        # Core Predictor Body (Residual MLP)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Output projection: Hidden -> Predicted Sy
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # Layer Norms for stability
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.ln_out = nn.LayerNorm(embedding_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        context: torch.Tensor, 
        action: torch.Tensor, 
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predicts the next state representation.
        
        Args:
            context (Sx): [Batch, Embedding_Dim]
            action (a): [Batch, Action_Dim]
            z (z): [Batch, Latent_Dim] - if None, a zero-tensor is used (deterministic mode)
            
        Returns:
            pred_state (Sy): [Batch, Embedding_Dim]
        """
        batch_size = context.shape[0]
        
        # Handle optional latent z
        if z is None:
            device = context.device
            z = torch.zeros(batch_size, self.latent_dim, device=device)
            
        # 1. Fuse Inputs
        # x = Concat(Sx, a, z)
        x = torch.cat([context, action, z], dim=-1)
        
        # 2. Project & Normalize
        x = self.input_proj(x)
        x = self.ln_in(x)
        x = F.gelu(x)
        
        # 3. Residual Processing
        for block in self.blocks:
            x = block(x)
            
        # 4. Project to Output Space
        x = self.output_proj(x)
        
        # Final Norm (optional, but helps keep representations bounded)
        x = self.ln_out(x)
        
        return x

class ResidualBlock(nn.Module):
    """Simple Residual Block with GeLU and LayerNorm."""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x + self.net(x))
