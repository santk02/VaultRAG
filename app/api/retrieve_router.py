from fastapi import APIRouter
from app.retrieval.pipeline import retrieval_pipeline
from app.models import RetrievalRequest, RetrievalResult

router = APIRouter(prefix="/v1/retrieve", tags=["retrieval"])


@router.post("", response_model=RetrievalResult)
async def retrieve(request: RetrievalRequest):
    """
    Retrieve relevant document chunks using hybrid search.

    Runs the full retrieval pipeline:
    1. BM25 keyword search
    2. Vector similarity search
    3. RRF fusion
    4. Cross-encoder reranking

    Returns top-k chunks with latency breakdown. Exposed separately from /v1/ask so
    retrieval quality can be inspected/tested without paying for an LLM call.
    """
    chunks, latencies = await retrieval_pipeline(
        query=request.query, top_k=request.top_k
    )

    return RetrievalResult(
        chunks=chunks, latency_ms=latencies["total_ms"], stage_latencies=latencies
    )
