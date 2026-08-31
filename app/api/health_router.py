from fastapi import APIRouter
from qdrant_client import QdrantClient

from app.config import settings
from app.db import db
from app.models import HealthResponse

# No prefix: mounts /health and / directly on the app, per blueprint's repo layout
router = APIRouter(tags=["health"])


def _check_qdrant() -> bool:
    """Probe Qdrant with a lightweight collections call; short client-side timeout so /health stays fast."""
    try:
        client = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port, timeout=2
        )
        client.get_collections()
        return True
    except Exception:
        return False  # unreachable, wrong port, or Qdrant not up yet


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — actually probes Postgres and Qdrant instead of hardcoding status."""
    db_ok = await db.is_healthy()
    qdrant_ok = _check_qdrant()

    # Overall status only "ok" when both dependencies are reachable; api itself is always "running"
    # if this handler executes at all.
    overall_status = "ok" if (db_ok and qdrant_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        services={
            "api": "running",
            "database": "connected" if db_ok else "not connected",
            "qdrant": "connected" if qdrant_ok else "not connected",
        },
    )


@router.get("/")
async def root():
    """Root endpoint — quick liveness/identity check without probing dependencies."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }
