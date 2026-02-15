# Omega-JEPA (Dream Stream) - Technical Specification

## 1. Overview
This document outlines the architecture for the Omega-JEPA Predictor Network, a clean-room implementation inspired by Joint Embedding Predictive Architectures (JEPA) but specialized for the "Dream Stream" environment. The core innovation is the integration of Protocol Omega metrics (Time-Reversal Asymmetry and Z-Scores) directly into the validation loop to ensure causal fidelity in state predictions.

## 2. Theoretical Foundation
### 2.1. The JEPA Paradigm
Unlike generative models that predict pixels (x), JEPA predicts representations (y) in an abstract space.
- **Context ($S_x$):** The current state representation.
- **Action ($a$):** The control or transition vector.
- **Latent ($z$):** A stochastic variable capturing uncertainty in the transition.
- **Prediction ($S_y$):** The predicted future state representation.

Equation: $S_y = Predictor(S_x, a, z)$

### 2.2. Protocol Omega Integration
To prevent the model from learning "easy" but non-causal shortcuts (representation collapse), we enforce Time-Reversal Asymmetry (TRA).
- **TRA Hypothesis:** A valid causal transition $A \to B$ should be distinguishable from $B \to A$ in the latent energy landscape.
- **Omega Score:** A composite metric combining prediction error (L2) with TRA violation penalties.

## 3. Architecture Design

### 3.1. `OmegaJEPA_Predictor` (The Brain)
- **Type:** PyTorch `nn.Module`.
- **Structure:** Multi-Layer Perceptron (MLP) with residual connections.
- **Inputs:**
    - `context_embedding`: Tensor [Batch, Dim]
    - `action_vector`: Tensor [Batch, ActionDim]
    - `latent_z`: Tensor [Batch, LatentDim] (Optional/sampled)
- **Mechanism:**
    1. Concatenate $S_x$, $a$, and $z$.
    2. Pass through a high-dimensional projection layer.
    3. Apply LayerNorm and GeLU activations.
    4. Output projected state $\hat{S}_y$.

### 3.2. `OmegaMetrics` (The Auditor)
- **Purpose:** Stateless validator class to compute physics-inspired metrics.
- **Key Metrics:**
    - `compute_tra(state_t, state_t1)`: Measures asymmetry magnitude.
    - `compute_z_score(residuals)`: Standard deviation based anomaly detection.
    - `energy_function(state)`: Helper for TRA computation (e.g., magnitude or learned energy).

## 4. Implementation Constraints
- **Framework:** PyTorch (v2.x).
- **Style:** Strict typing, modular design.
- **Clean Room:** No usage of Meta/Facebook source code; strictly first-principles implementation based on LeCun's 2022 paper and Protocol Omega specs.

## 5. Usage
```python
model = OmegaJEPA_Predictor(dim=256, action_dim=64)
metrics = OmegaMetrics()

# Forward pass
s_next_pred = model(s_curr, action, z)

# Validation
tra_score = metrics.compute_tra(s_curr, s_next_pred)
```
