"""
Event Integration Helper for MnemoCore Components
===================================================

Provides helper functions for integrating EventBus and WebhookManager
with existing MnemoCore modules (engine, dream, consolidation, etc.).

Usage:
    ```python
    from mnemocore.events.integration import emit_event, emit_memory_created

    # Emit specific event
    await emit_memory_created(
        event_bus=engine.event_bus,
        memory_id="abc123",
        content="Hello world",
        tier="hot"
    )

    # Emit generic event
    await emit_event(
        event_bus=engine.event_bus,
        event_type="custom.event",
        data={"key": "value"}
    )
    ```
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from loguru import logger

from .event_bus import EventBus


# =============================================================================
# Generic Event Emission
# =============================================================================

async def emit_event(
    event_bus: Optional[EventBus],
    event_type: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit an event to the EventBus if available.

    Args:
        event_bus: EventBus instance (may be None)
        event_type: Type of event to emit
        data: Event payload data
        metadata: Optional metadata
    """
    if event_bus is None:
        return

    try:
        await event_bus.publish(
            event_type=event_type,
            data=data,
            metadata=metadata or {},
        )
    except Exception as e:
        logger.warning(f"[EventIntegration] Failed to emit {event_type}: {e}")


# =============================================================================
# Memory Events
# =============================================================================

async def emit_memory_created(
    event_bus: Optional[EventBus],
    memory_id: str,
    content: str,
    tier: str,
    ltp_strength: Optional[float] = None,
    eig: Optional[float] = None,
    tags: Optional[List[str]] = None,
    category: Optional[str] = None,
    agent_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    previous_id: Optional[str] = None,
) -> None:
    """Emit memory.created event."""
    await emit_event(
        event_bus=event_bus,
        event_type="memory.created",
        data={
            "memory_id": memory_id,
            "content": content,
            "content_preview": content[:200] if len(content) > 200 else content,
            "tier": tier,
            "ltp_strength": ltp_strength,
            "eig": eig,
            "tags": tags or [],
            "category": category,
            "agent_id": agent_id,
            "episode_id": episode_id,
            "previous_id": previous_id,
        },
    )


async def emit_memory_accessed(
    event_bus: Optional[EventBus],
    memory_id: str,
    query: str,
    similarity_score: float,
    rank: int,
    tier: str,
) -> None:
    """Emit memory.accessed event."""
    await emit_event(
        event_bus=event_bus,
        event_type="memory.accessed",
        data={
            "memory_id": memory_id,
            "query": query,
            "similarity_score": similarity_score,
            "rank": rank,
            "tier": tier,
        },
    )


async def emit_memory_deleted(
    event_bus: Optional[EventBus],
    memory_id: str,
    tier: str,
    reason: str = "manual",
) -> None:
    """Emit memory.deleted event."""
    await emit_event(
        event_bus=event_bus,
        event_type="memory.deleted",
        data={
            "memory_id": memory_id,
            "tier": tier,
            "reason": reason,
        },
    )


async def emit_memory_consolidated(
    event_bus: Optional[EventBus],
    memory_id: str,
    from_tier: str,
    to_tier: str,
    consolidation_type: str = "promotion",
    ltp_strength: Optional[float] = None,
) -> None:
    """Emit memory.consolidated event."""
    await emit_event(
        event_bus=event_bus,
        event_type="memory.consolidated",
        data={
            "memory_id": memory_id,
            "from_tier": from_tier,
            "to_tier": to_tier,
            "consolidation_type": consolidation_type,
            "ltp_strength": ltp_strength,
        },
    )


# =============================================================================
# Consolidation Events
# =============================================================================

async def emit_consolidation_completed(
    event_bus: Optional[EventBus],
    cycle_id: str,
    duration_seconds: float,
    memories_processed: int,
    memories_consolidated: int,
    hot_to_warm: int = 0,
    warm_to_cold: int = 0,
) -> None:
    """Emit consolidation.completed event."""
    await emit_event(
        event_bus=event_bus,
        event_type="consolidation.completed",
        data={
            "cycle_id": cycle_id,
            "duration_seconds": duration_seconds,
            "memories_processed": memories_processed,
            "memories_consolidated": memories_consolidated,
            "hot_to_warm": hot_to_warm,
            "warm_to_cold": warm_to_cold,
        },
    )


# =============================================================================
# Contradiction Events
# =============================================================================

async def emit_contradiction_detected(
    event_bus: Optional[EventBus],
    group_id: str,
    memory_a_id: str,
    memory_b_id: str,
    similarity_score: float,
    llm_confirmed: bool = False,
    llm_confidence: Optional[float] = None,
) -> None:
    """Emit contradiction.detected event."""
    await emit_event(
        event_bus=event_bus,
        event_type="contradiction.detected",
        data={
            "group_id": group_id,
            "memory_a_id": memory_a_id,
            "memory_b_id": memory_b_id,
            "similarity_score": similarity_score,
            "llm_confirmed": llm_confirmed,
            "llm_confidence": llm_confidence,
        },
    )


async def emit_contradiction_resolved(
    event_bus: Optional[EventBus],
    group_id: str,
    resolution_type: str = "manual",
    resolution_note: Optional[str] = None,
    resolved_by: Optional[str] = None,
) -> None:
    """Emit contradiction.resolved event."""
    await emit_event(
        event_bus=event_bus,
        event_type="contradiction.resolved",
        data={
            "group_id": group_id,
            "resolution_type": resolution_type,
            "resolution_note": resolution_note,
            "resolved_by": resolved_by,
        },
    )


# =============================================================================
# Dream Events
# =============================================================================

async def emit_dream_started(
    event_bus: Optional[EventBus],
    session_id: str,
    trigger: str,
    schedule_name: Optional[str] = None,
    max_memories: Optional[int] = None,
    stages_enabled: Optional[List[str]] = None,
) -> None:
    """Emit dream.started event."""
    await emit_event(
        event_bus=event_bus,
        event_type="dream.started",
        data={
            "session_id": session_id,
            "trigger": trigger,
            "schedule_name": schedule_name,
            "max_memories": max_memories,
            "stages_enabled": stages_enabled or [],
        },
    )


async def emit_dream_completed(
    event_bus: Optional[EventBus],
    session_id: str,
    duration_seconds: float,
    memories_processed: int,
    clusters_found: int = 0,
    patterns_extracted: int = 0,
    contradictions_resolved: int = 0,
    memories_promoted: int = 0,
    dream_report_path: Optional[str] = None,
) -> None:
    """Emit dream.completed event."""
    await emit_event(
        event_bus=event_bus,
        event_type="dream.completed",
        data={
            "session_id": session_id,
            "duration_seconds": duration_seconds,
            "memories_processed": memories_processed,
            "clusters_found": clusters_found,
            "patterns_extracted": patterns_extracted,
            "contradictions_resolved": contradictions_resolved,
            "memories_promoted": memories_promoted,
            "dream_report_path": dream_report_path,
        },
    )


async def emit_dream_failed(
    event_bus: Optional[EventBus],
    session_id: str,
    error: str,
    stage: Optional[str] = None,
    duration_seconds: float = 0.0,
    memories_processed: int = 0,
) -> None:
    """Emit dream.failed event."""
    await emit_event(
        event_bus=event_bus,
        event_type="dream.failed",
        data={
            "session_id": session_id,
            "error": error,
            "stage": stage,
            "duration_seconds": duration_seconds,
            "memories_processed": memories_processed,
        },
    )


# =============================================================================
# Synapse Events
# =============================================================================

async def emit_synapse_formed(
    event_bus: Optional[EventBus],
    synapse_id: str,
    neuron_a_id: str,
    neuron_b_id: str,
    initial_strength: float = 1.0,
    formation_reason: str = "auto_bind",
    similarity: Optional[float] = None,
) -> None:
    """Emit synapse.formed event."""
    await emit_event(
        event_bus=event_bus,
        event_type="synapse.formed",
        data={
            "synapse_id": synapse_id,
            "neuron_a_id": neuron_a_id,
            "neuron_b_id": neuron_b_id,
            "initial_strength": initial_strength,
            "formation_reason": formation_reason,
            "similarity": similarity,
        },
    )


async def emit_synapse_fired(
    event_bus: Optional[EventBus],
    synapse_id: str,
    source_id: str,
    target_id: str,
    strength: float,
    was_successful: bool = True,
    context: Optional[str] = None,
) -> None:
    """Emit synapse.fired event."""
    await emit_event(
        event_bus=event_bus,
        event_type="synapse.fired",
        data={
            "synapse_id": synapse_id,
            "source_id": source_id,
            "target_id": target_id,
            "strength": strength,
            "was_successful": was_successful,
            "context": context,
        },
    )


# =============================================================================
# Gap Events
# =============================================================================

async def emit_gap_detected(
    event_bus: Optional[EventBus],
    gap_id: str,
    query: str,
    gap_type: str = "missing_info",
    related_memory_ids: Optional[List[str]] = None,
    severity: float = 0.5,
) -> None:
    """Emit gap.detected event."""
    await emit_event(
        event_bus=event_bus,
        event_type="gap.detected",
        data={
            "gap_id": gap_id,
            "query": query,
            "gap_type": gap_type,
            "related_memory_ids": related_memory_ids or [],
            "severity": severity,
        },
    )


async def emit_gap_filled(
    event_bus: Optional[EventBus],
    gap_id: str,
    memory_id: str,
    fill_method: str = "llm_generation",
    confidence: Optional[float] = None,
) -> None:
    """Emit gap.filled event."""
    await emit_event(
        event_bus=event_bus,
        event_type="gap.filled",
        data={
            "gap_id": gap_id,
            "memory_id": memory_id,
            "fill_method": fill_method,
            "confidence": confidence,
        },
    )
