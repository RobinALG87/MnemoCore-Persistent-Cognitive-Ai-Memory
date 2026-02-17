import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class HDV:
    """Holographic Distributed Representation"""

    vector: Optional[np.ndarray] = None  # 10,000-dimensional vector
    dimension: int = 10000
    id: str = None

    def __post_init__(self):
        if self.vector is None:
            # Initialize with random bipolar vector
            self.vector = np.random.choice(
                [-1, 1],
                size=self.dimension
            )
        elif self.vector.shape[0] != self.dimension:
             raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {self.vector.shape[0]}")


    def __add__(self, other: 'HDV') -> 'HDV':
        """Superposition: v_A + v_B contains both"""
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for superposition")
        return HDV(
            vector=self.vector + other.vector,
            dimension=self.dimension
        )

    def __xor__(self, other: 'HDV') -> 'HDV':
        """Binding: v_A ⊗ v_B (HRR circular convolution) (Deprecated: Use .bind() instead)"""
        return self.bind(other)

    def bind(self, other: 'HDV') -> 'HDV':
        """Binding: v_A ⊗ v_B (HRR circular convolution)"""
        if self.dimension != other.dimension:
             raise ValueError("Dimensions must match for binding")
        return HDV(
            vector=self.fft_convolution(self.vector, other.vector),
            dimension=self.dimension
        )

    def unbind(self, other: 'HDV') -> 'HDV':
        """Unbinding: v_AB ⊗ v_A* (Approximate inverse)"""
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for unbinding")
        # Unbinding is convolution with involution
        inv = self.involution(other.vector)
        return HDV(
            vector=self.fft_convolution(self.vector, inv),
            dimension=self.dimension
        ).normalize()

    def involution(self, a: np.ndarray) -> np.ndarray:
        """Involution for HRR: a_i* = a_{(-i mod N)}"""
        res = np.zeros_like(a)
        res[0] = a[0]
        res[1:] = a[:0:-1]
        return res

    def permute(self, shift: int = 1) -> 'HDV':
        """Permutation for sequence/role representation"""
        return HDV(
            vector=np.roll(self.vector, shift),
            dimension=self.dimension
        )

    def cosine_similarity(self, other: 'HDV') -> float:
        """Measure semantic similarity"""
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return np.dot(self.vector, other.vector) / (norm_a * norm_b)

    def normalize(self) -> 'HDV':
        """Binarize for cleaner superposition"""
        # np.sign returns 0 for 0, we want to avoid 0s in bipolar vectors generally, 
        # but for superposition result it's standard to threshold.
        # If 0, we can map to 1 or -1, or keep 0 (tertiary). 
        # For strict bipolar, we usually map >=0 to 1, <0 to -1.
        
        v = np.sign(self.vector)
        v[v == 0] = 1 # Handle zero case deterministically
        
        return HDV(
            vector=v.astype(int),
            dimension=self.dimension
        )

    @staticmethod
    def fft_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular convolution via FFT (HRR binding)"""
        fft_a = np.fft.fft(a)
        fft_b = np.fft.fft(b)
        fft_result = fft_a * fft_b
        return np.real(np.fft.ifft(fft_result))
