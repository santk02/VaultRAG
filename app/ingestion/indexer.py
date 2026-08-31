import pickle
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from app.config import settings
from app.db import db
from app.ingestion.chunker import Chunk
from app.ingestion.embedder import embed_texts


class Indexer:
    """
    Document indexer that handles:
    - Qdrant vector upserts
    - BM25 index building and persistence
    - PostgreSQL metadata storage
    """

    def __init__(self):
        """Initialize indexer with Qdrant client (one client per Indexer singleton, not per call)."""
        self.qdrant_client = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = settings.qdrant_vector_size

    async def ensure_collection(self):
        """Create Qdrant collection if it doesn't exist (idempotent — safe to call on every upload)."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )

    async def index_document(
        self, chunks: List[Chunk], filename: str, content_hash: str
    ) -> tuple[str, int]:
        """
        Index a document: upsert vectors, build BM25, store metadata.

        Args:
            chunks: List of Chunk objects
            filename: Original filename
            content_hash: Hash of document content for deduplication

        Returns:
            Tuple of (doc_id, chunk_count)
        """
        if not chunks:
            raise ValueError("Cannot index a document without text chunks")

        # Chunks are created before indexing, so their document ID is the
        # canonical ID used by both Qdrant payloads and the database FK.
        doc_id = chunks[0].doc_id

        # Check for duplicates by content hash — re-uploading the same file is a no-op,
        # not a duplicate row in Postgres or duplicate vectors in Qdrant.
        existing = await db.fetchrow(
            "SELECT doc_id FROM documents WHERE content_hash = $1", content_hash
        )
        if existing:
            return existing["doc_id"], len(chunks)

        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = await embed_texts(texts)

        # Prepare Qdrant points
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Upsert to Qdrant
        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

        # Insert document metadata
        await db.execute(
            """
            INSERT INTO documents (doc_id, filename, page_count, chunk_count, content_hash)
            VALUES ($1, $2, $3, $4, $5)
            """,
            doc_id,
            filename,
            max(chunk.page_number for chunk in chunks),
            len(chunks),
            content_hash,
        )

        # Insert chunk metadata
        for chunk in chunks:
            await db.execute(
                """
                INSERT INTO chunks (chunk_id, doc_id, chunk_index, page_number, text, token_count)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                chunk.chunk_id,
                doc_id,
                chunk.chunk_index,
                chunk.page_number,
                chunk.text,
                chunk.token_count,
            )

        # Update BM25 index
        await self.update_bm25_index(chunks)

        return doc_id, len(chunks)

    async def update_bm25_index(self, chunks: List[Chunk]):
        """
        Update the in-memory BM25 index with new chunks.

        For production, this should be replaced with a persistent BM25 service.
        For now, we persist to disk using pickle.
        """
        # Path is a setting (not a hardcoded relative filename) so the index location
        # doesn't silently depend on the process's current working directory.
        index_path = settings.bm25_index_path

        # Load existing index if exists
        try:
            with open(index_path, "rb") as f:
                bm25_index = pickle.load(f)
                corpus = bm25_index["corpus"]
                chunk_ids = bm25_index["chunk_ids"]
        except (FileNotFoundError, EOFError):
            corpus = []
            chunk_ids = []

        # Add new chunks
        for chunk in chunks:
            corpus.append(chunk.text.lower().split())
            chunk_ids.append(chunk.chunk_id)

        # Rebuild BM25 index from the full corpus — BM25Okapi has no incremental update API,
        # so every ingestion re-scores IDF over all chunks seen so far.
        bm25 = BM25Okapi(corpus)

        # Persist to disk
        with open(index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "corpus": corpus, "chunk_ids": chunk_ids}, f)

    async def get_bm25_index(self) -> dict:
        """
        Load the BM25 index from disk.

        Returns:
            Dictionary with bm25 instance, corpus, and chunk_ids
        """
        try:
            with open(settings.bm25_index_path, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {"bm25": None, "corpus": [], "chunk_ids": []}


# Global indexer instance — same singleton caveat as _embedder above
_indexer: Indexer = None


def get_indexer() -> Indexer:
    """Get or create the global indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = Indexer()
    return _indexer
