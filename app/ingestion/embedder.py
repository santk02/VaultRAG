from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class Embedder:
    """
    Embedding generator using a local MiniLM model.

    Model is loaded once at module level to avoid the classic performance bug
    of loading per request.
    """

    def __init__(self):
        """Initialize the local encoder once per process."""
        self.model = settings.embedding_model
        self.encoder = SentenceTransformer(self.model, device=settings.embedding_device)

    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.encoder.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(embeddings, dtype=np.float32)

    async def close(self):
        """Release the encoder reference during application shutdown."""
        self.encoder = None


# Global embedder instance (loaded once at module level). Simple None-check singleton —
# fine for FastAPI's single-process asyncio event loop; a double-init race is possible
# under multiple worker processes/threads calling get_embedder() concurrently for the
# first time, but each just loads its own model copy rather than corrupting state.
_embedder: Embedder = None


def get_embedder() -> Embedder:
    """
    Get or create the global embedder instance.

    Returns:
        Embedder instance
    """
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


async def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convenience function to embed texts using the global embedder.

    Args:
        texts: List of text strings to embed

    Returns:
        numpy array of embeddings
    """
    embedder = get_embedder()
    return await embedder.embed(texts)
