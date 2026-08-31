from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.ask_router import router as ask_router
from app.api.health_router import router as health_router
from app.api.retrieve_router import router as retrieve_router
from app.config import settings
from app.db import db
from app.extraction.router import router as extraction_router
from app.ingestion.router import router as ingestion_router

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
app.include_router(health_router)  # GET /health, GET / — no prefix
app.include_router(ingestion_router)
app.include_router(retrieve_router)
app.include_router(ask_router)
app.include_router(extraction_router)


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
