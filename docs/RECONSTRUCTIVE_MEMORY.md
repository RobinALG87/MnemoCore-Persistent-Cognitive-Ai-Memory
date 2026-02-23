# Reconstructive Memory Module - Usage Guide

## Overview

The Reconstructive Memory module implements cognitive science principles for synthesizing memories from fragments when exact matches are unavailable. This mimics how human memory works - we reconstruct memories rather than playing back stored recordings.

## Key Concepts

### Memory Fragment
A piece of retrieved memory with similarity score and metadata:
- `node_id`: Source memory identifier
- `content`: Text content
- `similarity`: Semantic similarity [0.0, 1.0]
- `source_tier`: Storage tier (HOT/WARM/COLD)

### Reconstructed Memory
A synthesized memory created from fragments:
- `content`: Synthesized text
- `fragments`: Source fragments used
- `confidence`: Confidence score [0.0, 1.0]
- `is_reconstructed`: Always true (vs stored memories)

### Reconstruction Methods
1. **Extraction**: Use best fragment directly (high similarity match)
2. **Interpolation**: Blend between two related fragments
3. **Synthesis**: Combine multiple fragments into new representation

## Basic Usage

```python
from mnemocore.cognitive import (
    ReconstructiveRecall,
    ReconstructionConfig,
)
from mnemocore.core.engine import HAIMEngine

# Initialize engine
engine = HAIMEngine()
await engine.initialize()

# Create reconstructor with optional config
config = ReconstructionConfig(
    synthesis_threshold=0.55,      # Below this -> reconstruct
    max_fragments=7,               # Max fragments to use
    enable_gap_detection=True,     # Detect knowledge gaps
    enable_persistent_storage=False, # Don't auto-store reconstructions
)

reconstructor = ReconstructiveRecall(
    engine=engine,
    config=config,
    gap_detector=engine.gap_detector,
)

# Perform reconstructive recall
result = await reconstructor.recall("What did we discuss about the project timeline?")

# Check result type
if result.is_reconstructed:
    print(f"Reconstructed (confidence: {result.confidence_breakdown['overall_confidence']:.2f})")
    print(f"Content: {result.reconstructed.content}")

    # View fragments used
    for frag in result.reconstructed.fragments:
        print(f"  - {frag.content[:50]}... (similarity: {frag.similarity:.2f})")
else:
    print("Direct match found!")
    for node_id, similarity in result.direct_matches:
        print(f"  {node_id}: {similarity:.2f}")
```

## Integration with HAIMEngine

The module is integrated into HAIMEngine via the `reconstructive_recall()` method:

```python
# Using the engine's built-in method
result = await engine.reconstructive_recall(
    query="What were the meeting outcomes?",
    top_k=10,
    enable_synthesis=True,
    project_id="my_project",
)

print(f"Content: {result['content']}")
print(f"Is reconstructed: {result['is_reconstructed']}")
print(f"Confidence: {result['confidence']}")
print(f"Fragments: {len(result['fragments'])}")
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_similarity_threshold` | 0.35 | Minimum similarity for fragment inclusion |
| `max_fragments` | 7 | Maximum fragments to use in synthesis |
| `synthesis_threshold` | 0.55 | Below this avg similarity, trigger reconstruction |
| `confidence_weight_fragment` | 0.5 | Weight for fragment similarity in confidence |
| `confidence_weight_count` | 0.2 | Weight for fragment count in confidence |
| `confidence_weight_coherence` | 0.3 | Weight for semantic coherence |
| `enable_gap_detection` | True | Whether to detect knowledge gaps |
| `enable_persistent_storage` | False | Whether to store reconstructed memories |
| `max_synthesis_length` | 500 | Maximum character length for synthesized content |

## Validation and Feedback

Provide feedback on reconstructions to improve future accuracy:

```python
# Validate a reconstruction
await reconstructor.validate_reconstruction(
    reconstruction=result.reconstructed,
    is_helpful=True,  # or False if not useful
    feedback="The synthesis captured the key points accurately.",
)
```

## Checking if a Memory is Reconstructed

```python
from mnemocore.cognitive import is_reconstructed_memory, get_reconstruction_metadata

node = await engine.get_memory(node_id)

if is_reconstructed_memory(node):
    metadata = get_reconstruction_metadata(node)
    print(f"Method: {metadata['method']}")
    print(f"Fragments used: {metadata['fragment_count']}")
    print(f"Query ID: {metadata['query_id']}")
```

## Statistics

Track reconstructive recall operations:

```python
stats = reconstructor.stats
print(f"Total recalls: {stats['total_recalls']}")
print(f"Reconstructed: {stats['reconstructed_count']}")
print(f"Direct matches: {stats['direct_match_count']}")
print(f"Gaps detected: {stats['gaps_detected']}")
```

## API Reference

### ReconstructiveRecall

**Methods:**
- `async recall(query, top_k=10, enable_synthesis=True, project_id=None)` - Main recall method
- `async recall_with_fragments(query, fragment_texts, query_id=None)` - Reconstruct from provided fragments
- `async validate_reconstruction(reconstruction, is_helpful, feedback=None)` - Record feedback
- `stats` - Get statistics dictionary

### ReconstructionResult

**Attributes:**
- `reconstructed: Optional[ReconstructedMemory]` - Synthesized memory if created
- `direct_matches: List[Tuple[str, float]]` - Direct (node_id, similarity) matches
- `fragments: List[MemoryFragment]` - All retrieved fragments
- `confidence_breakdown: Dict[str, float]` - Detailed confidence scoring
- `is_reconstructed: bool` - Whether result was synthesized
- `gap_records: List[GapRecord]` - Any gaps detected

### ReconstructedMemory

**Attributes:**
- `content: str` - Synthesized text content
- `fragments: List[MemoryFragment]` - Fragments used in synthesis
- `confidence: float` - Overall confidence score [0.0, 1.0]
- `is_reconstructed: bool` - Always True
- `reconstruction_method: str` - Method used (extraction/interpolation/synthesis)
- `created_at: datetime` - Timestamp of reconstruction
- `query_id: str` - Hash of original query
- `gap_detected: bool` - Whether a knowledge gap was identified
