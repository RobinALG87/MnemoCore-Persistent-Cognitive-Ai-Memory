import torch
import faiss
import time
from core.encoder import WorldEncoder
from core.predictor import OmegaJEPA_Predictor
from core.omega_metrics import OmegaMetrics

class DreamLoop:
    def __init__(self):
        print("Initializing Dream Loop...")
        self.encoder = WorldEncoder()
        
        # Load FAISS
        self.index = faiss.read_index("reality.faiss")
        
        # Load Model
        self.model = OmegaJEPA_Predictor(384, 384, 384)
        try:
            self.model.load_state_dict(torch.load("checkpoints/omega_jepa_reality.pt"))
            print("Loaded Reality Model.")
        except:
            print("Warning: No Reality Model found. Using untrained brain.")
            
        self.model.eval()
        
    def dream(self, context_text):
        print(f"\n💭 Dreaming about: '{context_text}'")
        s_now = self.encoder.encode(context_text)
        
        # Candidate Actions
        actions = [
            "Research extensively",
            "Build code immediately",
            "Wait and observe",
            "Consolidate memories",
            "Expand infrastructure"
        ]
        
        best_action = None
        best_score = -999
        
        for action in actions:
            a_vec = self.encoder.encode(action)
            
            # Predict Future
            with torch.no_grad():
                s_future = self.model(s_now, a_vec)
            
            # Evaluate (Omega Protocol)
            # 1. Z-Score (Is this a hallucination?)
            # Since we don't have a ground truth, we compare against the *Index*.
            # Distance to nearest real memory = "Plausibility"
            D, I = self.index.search(s_future.numpy(), 1)
            plausibility = -D[0][0] # Lower distance is better (higher plausibility)
            
            # 2. TRA (Is it causal?)
            # Fix: Create instance first
            metrics = OmegaMetrics()
            tra = metrics.compute_tra(s_now, s_future).item()
            
            # 3. Growth (Magnitude)
            growth = torch.norm(s_future).item() - torch.norm(s_now).item()
            
            # Composite Score
            score = (plausibility * 0.5) + (tra * 2.0) + (growth * 1.0)
            
            print(f"  👉 Action: '{action}' | Plausibility={plausibility:.2f} | TRA={tra:.2f} | Growth={growth:.2f} | Score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_action = action
                
        print(f"✨ Chosen Path: {best_action}")
        return best_action

if __name__ == "__main__":
    dreamer = DreamLoop()
    dreamer.dream("The system is stable but stagnant.")
