"""
Conceptual/Structural Memory Layer for HAIM.
Implements VSA-based knowledge graphs using Binary HDV.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any

from .binary_hdv import BinaryHDV, majority_bundle
from .config import get_config


class ConceptualMemory:
    """
    Conceptual/Structural Memory Layer for HAIM.
    Implements VSA-based knowledge graphs using BinaryHDV.

    Note: XOR binding is self-inverse, so bind() and unbind() are the same operation.
    This simplifies analogy solving compared to HRR binding.
    """

    def __init__(self, dimension: int = 16384, storage_dir: Optional[str] = None):
        self.dimension = dimension
        # Use config for default storage directory
        if storage_dir is None:
            config = get_config()
            storage_dir = config.paths.data_dir
        self.storage_dir = storage_dir
        self.codebook_path = os.path.join(storage_dir, "codebook.json")
        self.concepts_path = os.path.join(storage_dir, "concepts.json")

        # Symbol table (symbol string -> BinaryHDV)
        self.symbols: Dict[str, BinaryHDV] = {}
        # Concept table (concept name -> BinaryHDV)
        self.concepts: Dict[str, BinaryHDV] = {}

        self.load()

    def get_symbol(self, name: str) -> BinaryHDV:
        """Atomic symbol lookup or creation using deterministic seeding."""
        if name not in self.symbols:
            self.symbols[name] = BinaryHDV.from_seed(name, self.dimension)
        return self.symbols[name]

    def store_concept(self, name: str, attributes: Dict[str, str]):
        """
        Store a complex concept using structural binding.
        Concept = Bundled(Attribute XOR Value)

        Uses majority bundling for clean superposition with binary vectors.
        """
        bound_vectors = []

        for attr, val in attributes.items():
            attr_hdv = self.get_symbol(attr)
            val_hdv = self.get_symbol(val)
            # Bind attribute to value via XOR
            pair_hdv = attr_hdv.xor_bind(val_hdv)
            bound_vectors.append(pair_hdv)

        if bound_vectors:
            concept_hdv = majority_bundle(bound_vectors)
        else:
            concept_hdv = BinaryHDV.random(self.dimension)

        self.concepts[name] = concept_hdv
        self.save()

    def append_to_concept(self, name: str, attribute: str, value: str):
        """
        Add a new attribute-value pair to an existing concept bundle.
        Used for building growing hierarchies (e.g. Tag -> [Member1, Member2...])
        """
        attr_hdv = self.get_symbol(attribute)
        val_hdv = self.get_symbol(value)
        pair_hdv = attr_hdv.xor_bind(val_hdv)

        if name in self.concepts:
            # Superposition via majority bundling
            existing = self.concepts[name]
            concept_hdv = majority_bundle([existing, pair_hdv])
        else:
            concept_hdv = pair_hdv

        self.concepts[name] = concept_hdv
        self.save()

    def query(self, query_hdv: BinaryHDV, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Query for similar concepts.

        Note: Binary HDV similarity is in [0.0, 1.0] where:
        - 1.0 = identical
        - 0.5 = orthogonal/random
        - 0.0 = maximally different

        Default threshold raised to 0.5 (was 0.1 for cosine similarity).
        """
        results = []
        for name, hdv in self.concepts.items():
            sim = query_hdv.similarity(hdv)
            if sim >= threshold:
                results.append((name, float(sim)))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def solve_analogy(self, A_name: str, B_val: str, C_name: str) -> List[Tuple[str, float]]:
        """
        Solves A:B :: C:?
        A_name: source concept (e.g. 'arbitrage')
        B_val: source value (e.g. 'finance')
        C_name: target concept (e.g. 'bio_hacking')

        With XOR binding: D = (A XOR B) XOR C = A XOR B XOR C
        This is simpler than HRR but has different properties.
        """
        if A_name not in self.concepts or C_name not in self.concepts:
            return []

        A = self.concepts[A_name]
        C = self.concepts[C_name]
        B = self.get_symbol(B_val)

        # With XOR: A XOR B gives the relationship
        # We want D such that C XOR D has the same relationship
        # D = (A XOR B) XOR C
        relationship = A.xor_bind(B)
        D_hat = relationship.xor_bind(C)

        # Search in symbols for the nearest value
        matches = []
        for name, hdv in self.symbols.items():
            sim = D_hat.similarity(hdv)
            matches.append((name, float(sim)))

        return sorted(matches, key=lambda x: x[1], reverse=True)

    def extract_attribute(self, concept_name: str, attribute_name: str) -> List[Tuple[str, float]]:
        """
        What is the value of [attribute] for [concept]?

        With XOR binding: Concept contains (Attribute XOR Value)
        So: Concept XOR Attribute = Value
        """
        if concept_name not in self.concepts:
            return []

        concept_hdv = self.concepts[concept_name]
        attr_hdv = self.get_symbol(attribute_name)

        # XOR is self-inverse
        val_hat = concept_hdv.xor_bind(attr_hdv)

        matches = []
        for name, hdv in self.symbols.items():
            sim = val_hat.similarity(hdv)
            matches.append((name, float(sim)))

        return sorted(matches, key=lambda x: x[1], reverse=True)

    def save(self):
        """Persist symbols and concepts to disk."""
        os.makedirs(self.storage_dir, exist_ok=True)

        # Save Codebook - store as base64 for compactness
        codebook_data = {}
        for k, v in self.symbols.items():
            codebook_data[k] = {
                "data": list(v.data),  # Store as list of uint8
                "dimension": v.dimension
            }
        with open(self.codebook_path, 'w') as f:
            json.dump(codebook_data, f)

        # Save Concepts
        concepts_data = {}
        for k, v in self.concepts.items():
            concepts_data[k] = {
                "data": list(v.data),
                "dimension": v.dimension
            }
        with open(self.concepts_path, 'w') as f:
            json.dump(concepts_data, f)

    def load(self):
        """
        Load persisted symbols and concepts.

        Robustness: If the persisted vectors were saved with a different dimension
        than the currently configured ConceptualMemory.dimension, we skip them.
        This prevents hard failures when running tests with reduced dimensions.
        """
        if os.path.exists(self.codebook_path):
            with open(self.codebook_path, 'r') as f:
                data = json.load(f)
                loaded_symbols: Dict[str, BinaryHDV] = {}
                for k, v in data.items():
                    # Handle both new format (dict with data/dimension) and legacy format (list)
                    if isinstance(v, dict):
                        dim = v.get("dimension", self.dimension)
                        if dim != self.dimension:
                            continue
                        arr = np.array(v["data"], dtype=np.uint8)
                    else:
                        # Legacy format: float vector (skip - incompatible)
                        continue
                    loaded_symbols[k] = BinaryHDV(data=arr, dimension=dim)
                self.symbols = loaded_symbols

        if os.path.exists(self.concepts_path):
            with open(self.concepts_path, 'r') as f:
                data = json.load(f)
                loaded_concepts: Dict[str, BinaryHDV] = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        dim = v.get("dimension", self.dimension)
                        if dim != self.dimension:
                            continue
                        arr = np.array(v["data"], dtype=np.uint8)
                    else:
                        # Legacy format: float vector (skip - incompatible)
                        continue
                    loaded_concepts[k] = BinaryHDV(data=arr, dimension=dim)
                self.concepts = loaded_concepts
