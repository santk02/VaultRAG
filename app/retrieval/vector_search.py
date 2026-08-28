from typing import List, Tuple
from qdrant_client import QdrantClient
from app.ingestion.embedder import embed_texts
from app.config import settings


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
    
    # Initialize Qdrant client
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )
    
    # Generate query embedding
    query_embedding = await embed_texts([query])
    
    if len(query_embedding) == 0:
        return []
    
    # Search Qdrant
    search_results = client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_embedding[0].tolist(),
        limit=k,
        with_payload=True
    )
    
    # Return (chunk_id, rank) tuples
    results = [
        (result.payload["chunk_id"], rank + 1)  # rank is 1-indexed
        for rank, result in enumerate(search_results)
    ]
    
    return results
