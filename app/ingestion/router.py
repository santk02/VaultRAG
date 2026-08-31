import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.ingestion.chunker import create_chunks
from app.ingestion.indexer import get_indexer
from app.ingestion.parser import ParserError, parse_document
from app.models import DocumentUploadResponse

router = APIRouter(prefix="/v1/documents", tags=["ingestion"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a document for RAG.

    Supports PDF and DOCX files. Extracts text, chunks it, generates embeddings,
    and stores in Qdrant + PostgreSQL with page numbers for citation accuracy.
    """
    # Validate file size by streaming through the underlying file object (avoids loading
    # the whole upload into memory just to check its size)
    file_size = 0
    for chunk in file.file:
        file_size += len(chunk)
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum of {settings.max_file_size} bytes",
            )

    # Reset file pointer so the subsequent .read() below gets the full content again
    await file.seek(0)

    # Validate file extension
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = os.path.splitext(filename)[1].lower()
    allowed_extensions = settings.allowed_extensions.split(",")
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {settings.allowed_extensions}",
        )

    # Save temporary file
    temp_filename = f"temp_{uuid.uuid4()}_{filename}"
    temp_path = os.path.join("temp", temp_filename)

    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)

    try:
        # Write uploaded file to disk
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Parse document
        try:
            pages, content_hash = parse_document(temp_path, file_ext)
        except ParserError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Create chunks
        doc_id = str(uuid.uuid4())
        chunks = create_chunks(pages, doc_id)

        # Index document
        indexer = get_indexer()
        await indexer.ensure_collection()
        doc_id, chunk_count = await indexer.index_document(
            chunks, filename, content_hash
        )

        # Calculate page count
        page_count = max(page.page_number for page in pages) if pages else 0

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=filename,
            chunk_count=chunk_count,
            page_count=page_count,
            uploaded_at=datetime.now(timezone.utc),
        )

    finally:
        # Clean up temporary file regardless of success/failure above
        if os.path.exists(temp_path):
            os.remove(temp_path)
