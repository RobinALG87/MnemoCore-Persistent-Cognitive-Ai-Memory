import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from .hdv import HDV

class ConceptualMemory:
    """
    Conceptual/Structural Memory Layer for HAIM.
    Implements VSA-based knowledge graphs.
    """

    def __init__(self, dimension: int = 10000, storage_dir: str = "./data"):
        self.dimension = dimension
        self.storage_dir = storage_dir
        self.codebook_path = os.path.join(storage_dir, "codebook.json")
        self.concepts_path = os.path.join(storage_dir, "concepts.json")
        
        # Symbol table (symbol string -> HDV)
        self.symbols: Dict[str, HDV] = {}
        # Concept table (concept name -> HDV)
        self.concepts: Dict[str, HDV] = {}
        
        self.load()

    def get_symbol(self, name: str) -> HDV:
        """Atomic symbol lookup or creation."""
        if name not in self.symbols:
            self.symbols[name] = HDV(dimension=self.dimension)
        return self.symbols[name]

    def store_concept(self, name: str, attributes: Dict[str, str]):
        """
        Store a complex concept using structural binding.
        Concept = Bundled(Attribute * Value)
        """
        bundled_vector = np.zeros(self.dimension)
        
        for attr, val in attributes.items():
            attr_hdv = self.get_symbol(attr)
            val_hdv = self.get_symbol(val)
            # Bind attribute to value
            pair_hdv = attr_hdv.bind(val_hdv)
            bundled_vector += pair_hdv.vector
            
        concept_hdv = HDV(vector=bundled_vector, dimension=self.dimension).normalize()
        self.concepts[name] = concept_hdv
        self.save()

    def append_to_concept(self, name: str, attribute: str, value: str):
        """
        v1.7: Add a new attribute-value pair to an existing concept bundle.
        Used for building growing hierarchies (e.g. Tag -> [Member1, Member2...])
        """
        attr_hdv = self.get_symbol(attribute)
        val_hdv = self.get_symbol(value)
        pair_hdv = attr_hdv.bind(val_hdv)
        
        if name in self.concepts:
            # Superposition: Add to existing vector
            existing = self.concepts[name]
            new_vector = existing.vector + pair_hdv.vector
            self.concepts[name] = HDV(vector=new_vector, dimension=self.dimension).normalize()
        else:
            # Create new
            self.concepts[name] = pair_hdv.normalize()
            
        self.save()

    def query(self, query_hdv: HDV, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """Query for similar concepts."""
        results = []
        for name, hdv in self.concepts.items():
            sim = query_hdv.cosine_similarity(hdv)
            if sim >= threshold:
                results.append((name, float(sim)))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def solve_analogy(self, A_name: str, B_val: str, C_name: str) -> List[Tuple[str, float]]:
        """
        Solves A:B :: C:?
        A_name: source concept (e.g. 'arbitrage')
        B_val: source value (e.g. 'finance')
        C_name: target concept (e.g. 'bio_hacking')
        """
        if A_name not in self.concepts or C_name not in self.concepts:
            return []
            
        A = self.concepts[A_name]
        C = self.concepts[C_name]
        B = self.get_symbol(B_val)
        
        # M = C * inverse(A)
        mapping = C.unbind(A)
        # D = M * B
        D_hat = mapping.bind(B)
        
        # Search in symbols for the nearest value
        matches = []
        for name, hdv in self.symbols.items():
            sim = D_hat.cosine_similarity(hdv)
            matches.append((name, float(sim)))
            
        return sorted(matches, key=lambda x: x[1], reverse=True)

    def extract_attribute(self, concept_name: str, attribute_name: str) -> List[Tuple[str, float]]:
        """What is the value of [attribute] for [concept]?"""
        if concept_name not in self.concepts:
            return []
            
        concept_hdv = self.concepts[concept_name]
        attr_hdv = self.get_symbol(attribute_name)
        
        val_hat = concept_hdv.unbind(attr_hdv)
        
        matches = []
        for name, hdv in self.symbols.items():
            sim = val_hat.cosine_similarity(hdv)
            matches.append((name, float(sim)))
            
        return sorted(matches, key=lambda x: x[1], reverse=True)

    def save(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        # Save Codebook
        codebook_data = {k: v.vector.tolist() for k, v in self.symbols.items()}
        with open(self.codebook_path, 'w') as f:
            json.dump(codebook_data, f)
        # Save Concepts
        concepts_data = {k: v.vector.tolist() for k, v in self.concepts.items()}
        with open(self.concepts_path, 'w') as f:
            json.dump(concepts_data, f)

    def load(self):
        """Load persisted symbols and concepts.

        Robustness: If the persisted vectors were saved with a different dimension
        than the currently configured ConceptualMemory.dimension, we skip them.
        This prevents hard failures when running tests with reduced dimensions.
        """
        if os.path.exists(self.codebook_path):
            with open(self.codebook_path, 'r') as f:
                data = json.load(f)
                loaded_symbols: Dict[str, HDV] = {}
                for k, v in data.items():
                    vec = np.array(v)
                    if vec.shape[0] != self.dimension:
                        # Dimension mismatch: skip legacy vectors
                        continue
                    loaded_symbols[k] = HDV(vector=vec, dimension=self.dimension)
                self.symbols = loaded_symbols

        if os.path.exists(self.concepts_path):
            with open(self.concepts_path, 'r') as f:
                data = json.load(f)
                loaded_concepts: Dict[str, HDV] = {}
                for k, v in data.items():
                    vec = np.array(v)
                    if vec.shape[0] != self.dimension:
                        continue
                    loaded_concepts[k] = HDV(vector=vec, dimension=self.dimension)
                self.concepts = loaded_concepts
