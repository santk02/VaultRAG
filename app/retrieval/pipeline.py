import asyncio
import time
from typing import Dict, List

from app.db import db
from app.models import Chunk
from app.retrieval.bm25_search import bm25_search
from app.retrieval.fusion import fuse_results
from app.retrieval.reranker import rerank_chunks
from app.retrieval.vector_search import vector_search


async def _timed(coro, latencies: Dict[str, float], key: str):
    """Run a coroutine and record its own wall-clock duration under `key`.

    Needed because asyncio.gather runs both searches concurrently — wrapping each one
    individually (instead of timing the gather() call once) gives a true per-stage
    latency instead of reporting the same shared wall-clock duration for both.
    """
    start = time.time()
    result = await coro
    latencies[key] = (time.time() - start) * 1000
    return result


async def retrieval_pipeline(
    query: str, top_k: int = 5
) -> tuple[List[Chunk], Dict[str, float]]:
    """
    Run the complete retrieval pipeline with all 4 stages.

    Stages:
    1. BM25 search (keyword)
    2. Vector search (semantic)
    3. RRF fusion (merge)
    4. Cross-encoder reranking

    Args:
        query: Search query
        top_k: Number of final results to return

    Returns:
        Tuple of (list of Chunk objects, latency breakdown dict)
    """
    latencies = {}
    total_start = time.time()

    # The independent searches share the same query and run concurrently; each is timed
    # individually via _timed so bm25_ms/vector_ms reflect each stage's own duration
    # rather than the shared wall-clock time of the parallel block.
    bm25_results, vector_results = await asyncio.gather(
        _timed(bm25_search(query), latencies, "bm25_ms"),
        _timed(vector_search(query), latencies, "vector_ms"),
    )
    # Wall-clock time actually spent waiting on the parallel block (<= sum of the two stages
    # above since they overlap) — useful for accounting for the "total" latency budget.
    latencies["parallel_search_ms"] = max(latencies["bm25_ms"], latencies["vector_ms"])

    # Stage 3: RRF fusion
    stage_start = time.time()
    fused_ids = fuse_results(bm25_results, vector_results)
    latencies["fusion_ms"] = (time.time() - stage_start) * 1000

    # Stage 4: Reranking
    stage_start = time.time()
    reranked = await rerank_chunks(query, fused_ids, top_k)
    latencies["rerank_ms"] = (time.time() - stage_start) * 1000

    # Extract final chunk_ids
    final_chunk_ids = [chunk_id for chunk_id, _ in reranked]

    # Retrieve full chunk data from database
    stage_start = time.time()
    chunks = await _get_chunks(final_chunk_ids)
    latencies["fetch_ms"] = (time.time() - stage_start) * 1000

    # Total latency
    latencies["total_ms"] = (time.time() - total_start) * 1000

    return chunks, latencies


async def _get_chunks(chunk_ids: List[str]) -> List[Chunk]:
    """
    Retrieve full chunk data from database.

    Args:
        chunk_ids: List of chunk identifiers

    Returns:
        List of Chunk objects with metadata
    """
    if not chunk_ids:
        return []

    # Build query with ANY clause for array parameter
    query = """
        SELECT c.chunk_id, c.doc_id, c.chunk_index, c.page_number, c.text, c.token_count, d.filename
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        WHERE c.chunk_id = ANY($1)
    """

    results = await db.fetch(query, chunk_ids)
    rows_by_id = {row["chunk_id"]: row for row in results}

    # Rebuild in reranked order (dict lookup, not the DB's arbitrary row order) and
    # silently drop any chunk_id the DB no longer has (e.g. deleted between rerank and fetch)
    chunks = [
        Chunk(
            chunk_id=chunk_id,
            doc_id=rows_by_id[chunk_id]["doc_id"],
            chunk_index=rows_by_id[chunk_id]["chunk_index"],
            page_number=rows_by_id[chunk_id]["page_number"],
            text=rows_by_id[chunk_id]["text"],
            token_count=rows_by_id[chunk_id]["token_count"],
            filename=rows_by_id[chunk_id]["filename"],
        )
        for chunk_id in chunk_ids
        if chunk_id in rows_by_id
    ]

    return chunks
