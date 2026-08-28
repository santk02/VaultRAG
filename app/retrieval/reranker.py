from typing import List, Tuple
from sentence_transformers import CrossEncoder
from app.ingestion.indexer import get_indexer
from app.config import settings
import numpy as np


class Reranker:
    """
    Cross-encoder reranker for scoring (query, chunk) pairs directly.
    
    Much more accurate than bi-encoders but too slow to run on the whole corpus,
    so we run it only on the top-N candidates from hybrid retrieval.
    """
    
    def __init__(self):
        """Initialize cross-encoder model."""
        self.model_name = settings.reranker_model
        self.device = settings.reranker_device
        self.model = None
    
    def load_model(self):
        """Load the cross-encoder model (lazy loading)."""
        if self.model is None:
            self.model = CrossEncoder(self.model_name, device=self.device)
    
    async def rerank(
        self,
        query: str,
        chunk_ids: List[str],
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Rerank chunks by scoring (query, chunk) pairs.
        
        Args:
            query: Search query
            chunk_ids: List of chunk_ids to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, score) tuples sorted by score (descending)
        """
        if top_k is None:
            top_k = settings.rerank_top_k
        
        if not chunk_ids:
            return []
        
        self.load_model()
        
        # Get chunk texts from database
        chunk_texts = []
        for chunk_id in chunk_ids:
            chunk_data = await self._get_chunk_text(chunk_id)
            if chunk_data:
                chunk_texts.append(chunk_data)
            else:
                chunk_texts.append("")  # Placeholder for missing chunks
        
        # Create (query, chunk) pairs
        pairs = [[query, text] for text in chunk_texts]
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        indexed_scores = list(zip(chunk_ids, scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with scores
        return indexed_scores[:top_k]
    
    async def _get_chunk_text(self, chunk_id: str) -> str:
        """
        Retrieve chunk text from database.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk text or empty string if not found
        """
        from app.db import db
        
        result = await db.fetchrow(
            "SELECT text FROM chunks WHERE chunk_id = $1",
            chunk_id
        )
        
        return result["text"] if result else ""


# Global reranker instance
_reranker: Reranker = None


def get_reranker() -> Reranker:
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


async def rerank_chunks(
    query: str,
    chunk_ids: List[str],
    top_k: int = None
) -> List[Tuple[str, float]]:
    """
    Convenience function to rerank chunks.
    
    Args:
        query: Search query
        chunk_ids: List of chunk_ids to rerank
        top_k: Number of top results to return
        
    Returns:
        List of (chunk_id, score) tuples sorted by score
    """
    reranker = get_reranker()
    return await reranker.rerank(query, chunk_ids, top_k)
