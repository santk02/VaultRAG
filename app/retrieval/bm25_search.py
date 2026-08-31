from typing import List, Tuple

from app.config import settings
from app.ingestion.indexer import get_indexer


async def bm25_search(query: str, k: int = None) -> List[Tuple[str, int]]:
    """
    Perform BM25 keyword search on the document corpus.

    Args:
        query: Search query string
        k: Number of results to return (default from settings)

    Returns:
        List of (chunk_id, rank) tuples sorted by relevance
    """
    if k is None:
        k = settings.bm25_top_k

    # Load BM25 index (pickle read from disk each call — see indexer.py for the persistence format)
    indexer = get_indexer()
    bm25_data = await indexer.get_bm25_index()

    bm25 = bm25_data.get("bm25")
    corpus = bm25_data.get("corpus", [])
    chunk_ids = bm25_data.get("chunk_ids", [])

    if bm25 is None or not corpus:
        return []  # nothing indexed yet — caller treats this as "no BM25 signal"

    # Tokenize query — must match the lowercase/whitespace tokenization used when the index was built
    query_tokens = query.lower().split()

    # Get BM25 scores
    scores = bm25.get_scores(query_tokens)

    # Sort by score (descending) and get top-k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # Return (chunk_id, rank) tuples
    results = [
        (chunk_ids[i], rank + 1)  # rank is 1-indexed
        for rank, i in enumerate(top_indices)
        if scores[i]
        > 0  # a zero score means no query term overlap at all — exclude it, not just rank it low
    ]

    return results
