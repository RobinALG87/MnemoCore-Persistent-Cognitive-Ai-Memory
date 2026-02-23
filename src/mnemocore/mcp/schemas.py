from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from mnemocore.core.exceptions import ValidationError


class StoreToolInput(BaseModel):
    content: str = Field(..., min_length=1, max_length=100_000)
    metadata: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = Field(default=None, max_length=256)
    ttl: Optional[int] = Field(default=None, gt=0)


class QueryToolInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=100)
    agent_id: Optional[str] = Field(default=None, max_length=256)


class MemoryIdInput(BaseModel):
    memory_id: str = Field(..., min_length=1, max_length=256)


# --- Phase 5: Cognitive Client Schemas ---

class ObserveToolInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=256)
    content: str = Field(..., min_length=1, max_length=100_000)
    kind: str = Field(default="observation", max_length=64)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: Optional[list[str]] = None

class ContextToolInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=256)
    limit: int = Field(default=16, ge=1, le=100)

class EpisodeToolInput(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=256)
    goal: str = Field(..., min_length=1, max_length=10_000)
    context: Optional[str] = None


# --- Phase 4.5 & 5.0: Advanced Synthesis & Export Schemas ---

class SynthesizeToolInput(BaseModel):
    """Input for memory_synthesize tool (Phase 4.5 Recursive Synthesis)."""
    query: str = Field(..., min_length=3, max_length=4096, description="Complex query to synthesize")
    top_k: int = Field(default=10, ge=1, le=50, description="Final results to return")
    max_depth: int = Field(default=3, ge=0, le=5, description="Maximum recursion depth")
    context_text: Optional[str] = Field(None, max_length=500_000, description="Optional external context")
    project_id: Optional[str] = Field(None, max_length=128, description="Optional project scope")


class DreamToolInput(BaseModel):
    """Input for memory_dream tool (SubconsciousDaemon cycle)."""
    max_cycles: int = Field(default=1, ge=1, le=10, description="Number of dream cycles")
    force_insight: bool = Field(default=False, description="Force meta-insight generation")


class ExportToolInput(BaseModel):
    """Input for memory_export tool."""
    agent_id: Optional[str] = Field(None, max_length=256, description="Filter by agent_id")
    tier: Optional[str] = Field(None, pattern="^(hot|warm|cold|soul)$", description="Filter by tier")
    limit: int = Field(default=100, ge=1, le=1000, description="Max memories to export")
    include_metadata: bool = Field(default=True, description="Include full metadata")
    format: str = Field(default="json", pattern="^(json|jsonl)$", description="Export format")


class ToolResult(BaseModel):
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @field_validator("error")
    @classmethod
    def validate_error(cls, value: Optional[str], info):
        if info.data.get("ok") and value:
            raise ValidationError(
                field="error",
                reason="error must be empty when ok is true",
                value=value
            )
        return value
