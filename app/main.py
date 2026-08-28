import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.ask_router import router as ask_router
from app.api.retrieve_router import router as retrieve_router
from app.config import settings
from app.db import db
from app.extraction.router import router as extraction_router
from app.ingestion.router import router as ingestion_router
from app.models import HealthResponse

# Add app to path for imports
sys.path.insert(0, str(__file__).parent)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Privacy-first document Q&A for regulated industries",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingestion_router)
app.include_router(retrieve_router)
app.include_router(ask_router)
app.include_router(extraction_router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        services={
            "api": "running",
            "database": "not connected",
            "qdrant": "not connected",
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await db.connect()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await db.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.host, port=settings.port, reload=settings.debug
    )
