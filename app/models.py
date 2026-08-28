from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Document Models
class DocumentUploadResponse(BaseModel):
    """Response after successful document upload."""

    doc_id: str
    filename: str
    chunk_count: int
    page_count: int
    uploaded_at: datetime


class DocumentMetadata(BaseModel):
    """Document metadata from database."""

    doc_id: str
    filename: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    content_hash: Optional[str] = None
    uploaded_at: datetime


# Chunk Models
class Chunk(BaseModel):
    """A text chunk with metadata."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    page_number: int
    text: str
    token_count: Optional[int] = None
    filename: Optional[str] = None  # Added during retrieval


# Retrieval Models
class RetrievalRequest(BaseModel):
    """Request for document retrieval."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)


class RetrievalResult(BaseModel):
    """Result from retrieval pipeline."""

    chunks: List[Chunk]
    latency_ms: float
    stage_latencies: Dict[str, float]


# Generation Models
class AskRequest(BaseModel):
    """Request for Q&A."""

    question: str = Field(..., min_length=1, max_length=1000)
    mode: Optional[str] = Field(default="offline", pattern="^(cloud|offline)$")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    """Source citation."""

    filename: str
    page_number: int
    chunk_id: str


class AskResponse(BaseModel):
    """Response from Q&A endpoint."""

    answer: str
    sources: List[Source]
    mode: str
    model_used: str
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float


# Extraction Models
class ExtractionRequest(BaseModel):
    """Request for structured extraction."""

    text: str = Field(..., min_length=1)
    schema_name: str = Field(..., description="Name of extraction schema to use")


class ExtractionResponse(BaseModel):
    """Response from extraction endpoint."""

    extracted_data: Dict[str, Any]
    model_used: str
    latency_ms: float


# Health Models
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    services: Dict[str, str]


# Error Models
class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
