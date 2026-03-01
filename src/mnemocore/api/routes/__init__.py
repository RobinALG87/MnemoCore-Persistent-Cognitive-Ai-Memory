"""
API Route Modules
=================
Modular route organization for MnemoCore API.

Routes are organized by functional area:
- memories: Core memory CRUD operations
- episodes: Episodic memory management
- observations: Working memory observations
- dreams: Dream loop triggers
- health: Health checks and stats
- export: Memory export
- procedures: Procedural memory operations
- predictions: Prediction management
"""

from .memories import router as memories_router
from .episodes import router as episodes_router
from .observations import router as observations_router
from .dreams import router as dreams_router
from .health import router as health_router
from .export import router as export_router
from .procedures import router as procedures_router
from .predictions import router as predictions_router

__all__ = [
    "memories_router",
    "episodes_router",
    "observations_router",
    "dreams_router",
    "health_router",
    "export_router",
    "procedures_router",
    "predictions_router",
]
