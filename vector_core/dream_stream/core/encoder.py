import torch
from sentence_transformers import SentenceTransformer

class WorldEncoder:
    """
    Translates the Real World (Text) into Omega-JEPA Embeddings (Vectors).
    Uses 'all-MiniLM-L6-v2' for efficiency and high semantic density.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        print(f"Loading WorldEncoder: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = 384 # Output dimension of MiniLM
        
    def encode(self, text):
        """
        Input: "User needs coffee"
        Output: Tensor [1, 384]
        """
        if isinstance(text, str):
            text = [text]
            
        embeddings = self.model.encode(text, convert_to_tensor=True)
        return embeddings

    def encode_batch(self, texts):
        return self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

if __name__ == "__main__":
    # Test
    encoder = WorldEncoder()
    vec = encoder.encode("Omega Protocol Active")
    print(f"Encoded shape: {vec.shape}")
