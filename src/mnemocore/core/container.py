"""
Dependency Injection Container
==============================
Builds and wires all application dependencies.
Replaces singleton pattern with explicit dependency injection.
"""

from dataclasses import dataclass, field
from typing import Optional

from .config import HAIMConfig
from .async_storage import AsyncRedisStorage
from .qdrant_store import QdrantStore

# Phase 5 AGI Services
from .working_memory import WorkingMemoryService
from .episodic_store import EpisodicStoreService
from .semantic_store import SemanticStoreService
from .procedural_store import ProceduralStoreService
from .meta_memory import MetaMemoryService
from .agent_profile import AgentProfileService


@dataclass
class Container:
    """
    Container holding all wired application dependencies.
    """
    config: HAIMConfig
    redis_storage: Optional[AsyncRedisStorage] = None
    qdrant_store: Optional[QdrantStore] = None
    
    # Phase 5 Services
    working_memory: Optional[WorkingMemoryService] = None
    episodic_store: Optional[EpisodicStoreService] = None
    semantic_store: Optional[SemanticStoreService] = None
    procedural_store: Optional[ProceduralStoreService] = None
    meta_memory: Optional[MetaMemoryService] = None
    agent_profiles: Optional[AgentProfileService] = None


def build_container(config: HAIMConfig) -> Container:
    """
    Build and wire all application dependencies.

    Args:
        config: Validated HAIMConfig instance.

    Returns:
        Container with all dependencies initialized.
    """
    container = Container(config=config)

    # Initialize Redis storage
    container.redis_storage = AsyncRedisStorage(
        url=config.redis.url,
        stream_key=config.redis.stream_key,
        max_connections=config.redis.max_connections,
        socket_timeout=config.redis.socket_timeout,
        password=config.redis.password,
    )

    # Initialize Qdrant store
    container.qdrant_store = QdrantStore(
        url=config.qdrant.url,
        api_key=config.qdrant.api_key,
        dimensionality=config.dimensionality,
        collection_hot=config.qdrant.collection_hot,
        collection_warm=config.qdrant.collection_warm,
        binary_quantization=config.qdrant.binary_quantization,
        always_ram=config.qdrant.always_ram,
        hnsw_m=config.qdrant.hnsw_m,
        hnsw_ef_construct=config.qdrant.hnsw_ef_construct,
    )

    # Initialize Phase 5 AGI Services
    container.working_memory = WorkingMemoryService()
    container.episodic_store = EpisodicStoreService()
    container.semantic_store = SemanticStoreService(qdrant_store=container.qdrant_store)
    container.procedural_store = ProceduralStoreService()
    container.meta_memory = MetaMemoryService()
    container.agent_profiles = AgentProfileService()

    return container


def build_test_container(config: Optional[HAIMConfig] = None) -> Container:
    """
    Build a container for testing with mock/fake dependencies.

    Args:
        config: Optional test config. If None, uses default config.

    Returns:
        Container suitable for testing.
    """
    if config is None:
        from .config import load_config
        config = load_config()

    return build_container(config)
