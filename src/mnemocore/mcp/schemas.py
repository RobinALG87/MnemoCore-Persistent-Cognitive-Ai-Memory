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
