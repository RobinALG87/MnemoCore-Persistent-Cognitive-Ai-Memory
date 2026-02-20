"""
API Request/Response Models
===========================
Pydantic models with comprehensive input validation and Field validators.
"""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class StoreRequest(BaseModel):
    """Request model for storing a memory."""

    content: str = Field(
        ...,
        max_length=100_000,
        description="The content to store as a memory",
        examples=["This is a sample memory content"],
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata associated with the memory"
    )
    agent_id: Optional[str] = Field(
        default=None, max_length=256, description="Optional agent identifier"
    )
    ttl: Optional[int] = Field(
        default=None,
        ge=1,
        le=86400 * 365,  # Max 1 year TTL
        description="Time-to-live in seconds (1 to 31536000)",
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v

    @field_validator("metadata")
    @classmethod
    def check_metadata_size(
        cls, v: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Validate metadata constraints."""
        if v is None:
            return v
        if len(v) > 50:
            raise ValueError("Too many metadata keys (max 50)")
        for key, value in v.items():
            if len(key) > 64:
                raise ValueError(
                    f'Metadata key "{key[:20]}..." too long (max 64 chars)'
                )
            if not re.match(r"^[a-zA-Z0-9_\-\.]+$", key):
                raise ValueError(
                    f'Metadata key "{key}" contains invalid characters (only alphanumeric, underscore, hyphen, dot allowed)'
                )
            # Metadata values can be Any, but limit strings
            if isinstance(value, str) and len(value) > 1000:
                raise ValueError(
                    f'Metadata value for "{key}" too long (max 1000 chars)'
                )
            # Limit nested structures
            if isinstance(value, (dict, list)):
                raise ValueError(
                    f'Metadata value for "{key}" must be a primitive type (str, int, float, bool, null)'
                )
        return v

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate agent_id format."""
        if v is None:
            return v
        if not re.match(r"^[a-zA-Z0-9_\-\:]+$", v):
            raise ValueError("Agent ID contains invalid characters")
        return v


class QueryRequest(BaseModel):
    """Request model for querying memories."""

    query: str = Field(
        ...,
        max_length=10000,
        description="The search query string",
        examples=["sample search query"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of results to return (1-100)",
    )
    agent_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional agent identifier to filter by",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v


class ConceptRequest(BaseModel):
    """Request model for defining a concept."""

    name: str = Field(
        ..., max_length=256, description="Name of the concept", examples=["animal"]
    )
    attributes: Dict[str, str] = Field(
        ..., description="Key-value attributes for the concept"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate concept name."""
        if not v or not v.strip():
            raise ValueError("Concept name cannot be empty")
        if not re.match(r"^[a-zA-Z0-9_\-\s]+$", v):
            raise ValueError("Concept name contains invalid characters")
        return v.strip()

    @field_validator("attributes")
    @classmethod
    def check_attributes_size(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate attributes constraints."""
        if len(v) == 0:
            raise ValueError("At least one attribute is required")
        if len(v) > 50:
            raise ValueError("Too many attributes (max 50)")
        for key, value in v.items():
            if len(key) > 64:
                raise ValueError(
                    f'Attribute key "{key[:20]}..." too long (max 64 chars)'
                )
            if not re.match(r"^[a-zA-Z0-9_\-\.]+$", key):
                raise ValueError(f'Attribute key "{key}" contains invalid characters')
            if len(value) > 1000:
                raise ValueError(
                    f'Attribute value for "{key}" too long (max 1000 chars)'
                )
        return v


class AnalogyRequest(BaseModel):
    """Request model for solving analogies."""

    source_concept: str = Field(
        ..., max_length=256, description="The source concept in the analogy"
    )
    source_value: str = Field(
        ..., max_length=1000, description="The value associated with the source concept"
    )
    target_concept: str = Field(
        ..., max_length=256, description="The target concept in the analogy"
    )

    @field_validator("source_concept", "target_concept")
    @classmethod
    def validate_concept(cls, v: str) -> str:
        """Validate concept names."""
        if not v or not v.strip():
            raise ValueError("Concept cannot be empty")
        return v.strip()

    @field_validator("source_value")
    @classmethod
    def validate_value(cls, v: str) -> str:
        """Validate source value."""
        if not v or not v.strip():
            raise ValueError("Source value cannot be empty")
        return v.strip()


class MemoryResponse(BaseModel):
    """Response model for memory retrieval."""

    id: str
    content: str
    metadata: Dict[str, Any]
    created_at: str
    epistemic_value: float = 0.0
    ltp_strength: float = 0.0
    tier: str = "unknown"


class QueryResult(BaseModel):
    """Single result from a query."""

    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    tier: str


class QueryResponse(BaseModel):
    """Response model for query results."""

    ok: bool = True
    query: str
    results: List[QueryResult]


class StoreResponse(BaseModel):
    """Response model for store operation."""

    ok: bool = True
    memory_id: str
    message: str


class DeleteResponse(BaseModel):
    """Response model for delete operation."""

    ok: bool = True
    deleted: str


class ConceptResponse(BaseModel):
    """Response model for concept definition."""

    ok: bool = True
    concept: str


class AnalogyResult(BaseModel):
    """Single result from an analogy query."""

    value: str
    score: float


class AnalogyResponse(BaseModel):
    """Response model for analogy query."""

    ok: bool = True
    analogy: str
    results: List[AnalogyResult]


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str
    error_type: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    redis_connected: bool
    storage_circuit_breaker: str
    qdrant_circuit_breaker: str
    engine_ready: bool
    timestamp: str


class RootResponse(BaseModel):
    """Root endpoint response model."""

    status: str
    service: str
    version: str
    phase: str
    timestamp: str
