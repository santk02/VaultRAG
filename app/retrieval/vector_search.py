from typing import List, Tuple
from qdrant_client import QdrantClient
from app.ingestion.embedder import embed_texts
from app.config import settings

# Module-level singleton client — mirrors get_indexer()/get_embedder(); creating a new
# QdrantClient (and its TCP connection) per search call was the same "load per request"
# anti-pattern the blueprint calls out for the embedder.
_client: QdrantClient = None


def get_qdrant_client() -> QdrantClient:
    """Get or create the shared Qdrant client for search calls."""
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


async def vector_search(query: str, k: int = None) -> List[Tuple[str, int]]:
    """
    Perform vector similarity search using Qdrant.

    Args:
        query: Search query string
        k: Number of results to return (default from settings)

    Returns:
        List of (chunk_id, rank) tuples sorted by similarity
    """
    if k is None:
        k = settings.vector_top_k

    client = get_qdrant_client()

    # Generate query embedding
    query_embedding = await embed_texts([query])

    if len(query_embedding) == 0:
        return []

    # Search Qdrant
    search_results = client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_embedding[0].tolist(),
        limit=k,
        with_payload=True,
    )

    # Return (chunk_id, rank) tuples
    results = [
        (result.payload["chunk_id"], rank + 1)  # rank is 1-indexed
        for rank, result in enumerate(search_results)
    ]

    return results
