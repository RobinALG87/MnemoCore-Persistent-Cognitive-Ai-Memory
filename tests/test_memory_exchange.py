"""
Tests for Multi-Agent Memory Exchange Protocol (memory_exchange.py)
====================================================================
Covers MemoryExchangeProtocol, SharedMemory, AgentPermission,
AccessLevel, MemoryTierAccess.

Research basis: SAMEP (arXiv 2507) — 73% reduction in redundant
computations, 89% improved context relevance.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from mnemocore.core.memory_exchange import (
    AccessLevel,
    AgentPermission,
    MemoryExchangeProtocol,
    MemoryTierAccess,
    SharedMemory,
)


# ═══════════════════════════════════════════════════════════════════════
# AccessLevel / AgentPermission
# ═══════════════════════════════════════════════════════════════════════

class TestAccessLevel:

    def test_ordering(self):
        assert AccessLevel.NONE.value < AccessLevel.READ.value
        assert AccessLevel.READ.value < AccessLevel.ANNOTATE.value
        assert AccessLevel.ANNOTATE.value < AccessLevel.FORK.value
        assert AccessLevel.FORK.value < AccessLevel.FULL.value


class TestAgentPermission:

    def test_default_permission(self):
        perm = AgentPermission(agent_id="agent-1")
        assert perm.access_level == AccessLevel.READ
        assert perm.is_expired is False
        assert perm.can_access("hot") is True
        assert perm.can_access("cold") is False

    def test_expired_permission(self):
        perm = AgentPermission(
            agent_id="agent-1",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert perm.is_expired is True
        assert perm.can_access("hot") is False

    def test_to_dict_roundtrip(self):
        perm = AgentPermission(
            agent_id="agent-1",
            access_level=AccessLevel.FORK,
            allowed_tiers=["hot", "warm", "cold"],
        )
        d = perm.to_dict()
        perm2 = AgentPermission.from_dict(d)
        assert perm2.agent_id == "agent-1"
        assert perm2.access_level == AccessLevel.FORK
        assert "cold" in perm2.allowed_tiers


# ═══════════════════════════════════════════════════════════════════════
# SharedMemory
# ═══════════════════════════════════════════════════════════════════════

class TestSharedMemory:

    def test_compute_signature(self):
        sm = SharedMemory(
            source_agent_id="agent-1",
            source_memory_id="mem-1",
            content="test content",
        )
        sig = sm.compute_signature()
        assert len(sig) == 64  # SHA-256 hex
        sm.signature = sig
        assert sm.verify_signature() is True

    def test_tampered_content_fails_verification(self):
        sm = SharedMemory(
            source_agent_id="agent-1",
            source_memory_id="mem-1",
            content="original content",
        )
        sm.signature = sm.compute_signature()
        sm.content = "tampered content"
        assert sm.verify_signature() is False

    def test_to_dict_roundtrip(self):
        sm = SharedMemory(
            source_agent_id="agent-1",
            source_memory_id="mem-1",
            content="test",
            tier="warm",
            tags=["tag1", "tag2"],
        )
        sm.signature = sm.compute_signature()
        d = sm.to_dict()
        sm2 = SharedMemory.from_dict(d)
        assert sm2.source_agent_id == "agent-1"
        assert sm2.tier == "warm"
        assert sm2.tags == ["tag1", "tag2"]
        assert sm2.signature == sm.signature


# ═══════════════════════════════════════════════════════════════════════
# MemoryExchangeProtocol
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryExchangeProtocol:

    def test_share_and_request(self):
        protocol = MemoryExchangeProtocol()

        # Agent A shares a memory
        shared = protocol.share(
            source_agent_id="agent-a",
            source_memory_id="mem-1",
            content="how to deploy kubernetes",
            tags=["devops", "k8s"],
        )
        assert shared.id
        assert shared.verify_signature() is True

        # Agent B needs permission first
        protocol.grant_permission("agent-b", granted_by="agent-a")

        # Agent B requests the shared memory
        result = protocol.request("agent-b", shared.id)
        assert result is not None
        assert result.content == "how to deploy kubernetes"

    def test_request_without_permission(self):
        protocol = MemoryExchangeProtocol()
        shared = protocol.share(
            source_agent_id="agent-a",
            source_memory_id="mem-1",
            content="secret info",
        )
        result = protocol.request("unauthorized-agent", shared.id)
        assert result is None

    def test_discover(self):
        protocol = MemoryExchangeProtocol()

        # Share multiple memories
        protocol.share("agent-a", "m1", "python debugging techniques", tags=["python"])
        protocol.share("agent-a", "m2", "java debugging", tags=["java"])
        protocol.share("agent-a", "m3", "cooking recipes", tags=["food"])

        # Grant access to agent-b
        protocol.grant_permission("agent-b", granted_by="agent-a")

        # Discover relevant memories
        results = protocol.discover("agent-b", "debugging")
        assert len(results) >= 1
        assert any("debug" in r.content.lower() for r in results)

    def test_discover_excludes_own(self):
        protocol = MemoryExchangeProtocol()
        protocol.share("agent-a", "m1", "my own memory")
        protocol.grant_permission("agent-a", granted_by="system")
        results = protocol.discover("agent-a", "memory", exclude_own=True)
        assert len(results) == 0

    def test_discover_includes_own_when_not_excluded(self):
        protocol = MemoryExchangeProtocol()
        protocol.share("agent-a", "m1", "my own memory")
        protocol.grant_permission("agent-a", granted_by="system")
        results = protocol.discover("agent-a", "memory", exclude_own=False)
        assert len(results) >= 1

    def test_annotate(self):
        protocol = MemoryExchangeProtocol()
        shared = protocol.share("agent-a", "m1", "test content")
        protocol.grant_permission("agent-b", granted_by="agent-a", access_level=AccessLevel.ANNOTATE)

        success = protocol.annotate("agent-b", shared.id, "great memory!", rating=0.9)
        assert success is True
        assert len(shared.annotations) == 1
        assert shared.annotations[0]["rating"] == 0.9

    def test_annotate_denied_read_only(self):
        protocol = MemoryExchangeProtocol()
        shared = protocol.share("agent-a", "m1", "test content")
        protocol.grant_permission("agent-b", granted_by="agent-a", access_level=AccessLevel.READ)

        success = protocol.annotate("agent-b", shared.id, "not allowed")
        assert success is False

    def test_revoke_share(self):
        protocol = MemoryExchangeProtocol()
        shared = protocol.share("agent-a", "m1", "ephemeral data")
        protocol.grant_permission("agent-b", granted_by="agent-a")

        # Revoke by source agent
        result = protocol.revoke_share("agent-a", shared.id)
        assert result is True
        assert shared.is_revoked is True

        # Agent B can no longer access
        accessed = protocol.request("agent-b", shared.id)
        assert accessed is None

    def test_revoke_by_wrong_agent(self):
        protocol = MemoryExchangeProtocol()
        shared = protocol.share("agent-a", "m1", "data")
        result = protocol.revoke_share("agent-b", shared.id)
        assert result is False

    def test_permission_management(self):
        protocol = MemoryExchangeProtocol()

        # Grant
        perm = protocol.grant_permission(
            "agent-b",
            granted_by="admin",
            access_level=AccessLevel.FULL,
            allowed_tiers=["hot", "warm", "cold"],
        )
        assert perm.access_level == AccessLevel.FULL

        # Check effective access
        level = protocol.get_access_level("agent-b", "cold")
        assert level == AccessLevel.FULL

        # Revoke
        revoked = protocol.revoke_permission("agent-b", revoked_by="admin")
        assert revoked is True

        # Check again
        level = protocol.get_access_level("agent-b", "cold")
        assert level == AccessLevel.NONE

    def test_tier_based_access_control(self):
        protocol = MemoryExchangeProtocol()

        # Grant access only to hot tier
        protocol.grant_permission(
            "agent-b",
            granted_by="system",
            allowed_tiers=["hot"],
        )

        shared_hot = protocol.share("agent-a", "m1", "hot data", tier="hot")
        shared_cold = protocol.share("agent-a", "m2", "cold data", tier="cold")

        # Can access hot
        result_hot = protocol.request("agent-b", shared_hot.id)
        assert result_hot is not None

        # Cannot access cold
        result_cold = protocol.request("agent-b", shared_cold.id)
        assert result_cold is None

    def test_expired_permission_denied(self):
        protocol = MemoryExchangeProtocol()
        protocol.grant_permission(
            "agent-b",
            granted_by="system",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        shared = protocol.share("agent-a", "m1", "data")
        result = protocol.request("agent-b", shared.id)
        assert result is None

    def test_get_stats(self):
        protocol = MemoryExchangeProtocol()
        protocol.share("agent-a", "m1", "data1")
        protocol.share("agent-b", "m2", "data2")
        protocol.grant_permission("agent-c", granted_by="system")
        protocol.discover("agent-c", "data")

        stats = protocol.get_stats()
        assert stats["total_shared"] == 2
        assert stats["active_shared"] == 2
        assert stats["total_discoveries"] >= 1

    def test_persistence(self, tmp_path):
        path = tmp_path / "exchange.json"
        protocol = MemoryExchangeProtocol(config=MagicMock(
            persistence_path=str(path),
            auto_persist=True,
            max_shared_memories=50000,
            max_annotations_per_memory=50,
            default_access_level=1,
        ))
        protocol.share("agent-a", "m1", "persistent data")
        protocol.grant_permission("agent-b", granted_by="system")
        assert path.exists()

        # Load from disk
        protocol2 = MemoryExchangeProtocol(config=MagicMock(
            persistence_path=str(path),
            auto_persist=True,
            max_shared_memories=50000,
            max_annotations_per_memory=50,
            default_access_level=1,
        ))
        assert len(protocol2._shared) == 1

    def test_discover_with_tag_filter(self):
        protocol = MemoryExchangeProtocol()
        protocol.share("agent-a", "m1", "python tips", tags=["python", "coding"])
        protocol.share("agent-a", "m2", "cooking tips", tags=["food", "recipes"])
        protocol.grant_permission("agent-b", granted_by="system")

        results = protocol.discover("agent-b", "tips", tags=["python"])
        assert len(results) >= 1
        assert all("python" in r.tags for r in results)
