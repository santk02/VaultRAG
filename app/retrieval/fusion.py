from typing import List
from app.config import settings


def rrf(rank_lists: List[List[str]], k: int = None, top_n: int = None) -> List[str]:
    """
    Merge ranked ID lists using Reciprocal Rank Fusion (RRF).
    
    RRF ignores scores and uses only ranks, avoiding scale mismatch between
    different retrieval methods (e.g., BM25 scores vs cosine similarity).
    
    Formula: score(doc) = Σ 1/(k + rank) across all rank lists
    
    Args:
        rank_lists: List of ranked chunk_id lists (each from a different retrieval method)
        k: RRF constant (default from settings, typically 60)
        top_n: Number of top results to return (default from settings)
        
    Returns:
        List of chunk_ids sorted by RRF score (descending)
    """
    if k is None:
        k = settings.rrf_k
    if top_n is None:
        top_n = settings.rrf_top_n
    
    scores: dict[str, float] = {}
    
    for ranked in rank_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    
    # Sort by score (descending) and return top_n
    sorted_results = sorted(scores, key=scores.get, reverse=True)[:top_n]
    
    return sorted_results


def fuse_results(
    bm25_results: List[tuple],
    vector_results: List[tuple]
) -> List[str]:
    """
    Convenience function to fuse BM25 and vector search results.
    
    Args:
        bm25_results: List of (chunk_id, rank) tuples from BM25
        vector_results: List of (chunk_id, rank) tuples from vector search
        
    Returns:
        List of chunk_ids sorted by RRF score
    """
    # Extract chunk_ids from tuples
    bm25_ids = [chunk_id for chunk_id, _ in bm25_results]
    vector_ids = [chunk_id for chunk_id, _ in vector_results]
    
    return rrf([bm25_ids, vector_ids])
