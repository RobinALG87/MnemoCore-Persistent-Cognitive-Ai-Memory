import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from mnemocore.core.semantic_store import SemanticStoreService
from mnemocore.core.memory_model import SemanticConcept
from mnemocore.core.binary_hdv import BinaryHDV

@pytest.fixture
def mock_qdrant():
    return MagicMock()

def test_semantic_store_upsert_and_get(mock_qdrant):
    store = SemanticStoreService(qdrant_store=mock_qdrant)
    
    mock_hdv = MagicMock(spec=BinaryHDV)
    
    concept = SemanticConcept(
        id="Dog",
        label="Dog",
        description="A domesticated carnivorous mammal",
        tags=["animal"],
        prototype_hdv=mock_hdv,
        support_episode_ids=[],
        reliability=1.0,
        last_updated_at=datetime.now(timezone.utc),
        metadata={"legs": 4, "sound": "bark"}
    )
    
    store.upsert_concept(concept)
    
    retrieved = store.get_concept("Dog")
    assert retrieved is not None
    assert retrieved.label == "Dog"
    assert retrieved.description == "A domesticated carnivorous mammal"
    assert retrieved.metadata["legs"] == 4
    
    # Upsert existing
    concept.metadata["friendly"] = True
    store.upsert_concept(concept)
    
    concept2 = store.get_concept("Dog")
    assert concept2.metadata["legs"] == 4
    assert concept2.metadata["friendly"] is True

def test_semantic_store_find_nearby(mock_qdrant):
    store = SemanticStoreService(qdrant_store=mock_qdrant)
    
    # We just test the empty return when Qdrant is mocked and we haven't wired 
    # the advanced embedding layer in the test
    results = store.find_nearby_concepts("Puppy")
    assert isinstance(results, list)
    assert len(results) == 0
