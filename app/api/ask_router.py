import time
import uuid

from fastapi import APIRouter, HTTPException

from app.db import db
from app.generation.citation import get_citation_checker
from app.generation.guards import get_output_guard
from app.generation.llm import generate_answer
from app.models import AskRequest, AskResponse, Source
from app.observability.tracing import span, start_trace
from app.retrieval.pipeline import retrieval_pipeline

router = APIRouter(prefix="/v1/ask", tags=["generation"])


@router.post("", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Ask a question over the document corpus with cited answers.

    Pipeline:
    1. Retrieve relevant chunks using hybrid search
    2. Generate answer using LLM with citation enforcement
    3. Validate citations against retrieved chunks
    4. Run output guards (toxicity, PII, format)
    5. Log query to audit trail
    """
    query_id = str(uuid.uuid4())
    total_start = time.time()

    # Top-level Langfuse trace for the whole request (no-op if Langfuse is unconfigured);
    # the OTel span below is the vendor-neutral counterpart that's always active.
    lf_trace = start_trace(
        "ask", input_data={"question": request.question, "mode": request.mode}
    )

    with span("ask.request"):
        try:
            # Stage 1: Retrieval — child span covers BM25 + vector + RRF + rerank as one unit
            with span("ask.retrieval"):
                retrieval_start = time.time()
                chunks, retrieval_latencies = await retrieval_pipeline(
                    query=request.question, top_k=request.top_k
                )
                retrieval_latency_ms = (time.time() - retrieval_start) * 1000

            # Check if we retrieved anything meaningful
            if not chunks:
                return AskResponse(
                    answer="I don't have enough information in the provided documents to answer that.",
                    sources=[],
                    mode=request.mode,
                    model_used="none",
                    latency_ms=retrieval_latency_ms,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=0,
                )

            # Stage 2: Generation
            with span("ask.generation"):
                answer, model_used, generation_latency_ms = await generate_answer(
                    question=request.question,
                    chunks=chunks,
                    model_override=None,
                    mode=request.mode,
                )

            # Stage 3: Citation validation
            with span("ask.citation_check"):
                citation_checker = get_citation_checker()
                is_valid_citations, invalid_sentences, valid_citations = (
                    citation_checker.validate_citations(answer, chunks)
                )

                if not is_valid_citations:
                    # Strip invalid citations
                    answer = citation_checker.strip_invalid_citations(answer, chunks)

            # Stage 4: Output guards
            guard = get_output_guard()
            is_valid_guards, guard_error = guard.validate_all(answer)

            if not is_valid_guards:
                raise HTTPException(
                    status_code=400, detail=f"Output validation failed: {guard_error}"
                )

            # Extract sources from answer and map to actual chunks
            source_map = {
                (chunk.filename, chunk.page_number): chunk.chunk_id for chunk in chunks
            }
            sources = [
                Source(
                    filename=filename,
                    page_number=page,
                    chunk_id=source_map.get((filename, page), ""),
                )
                for filename, page in valid_citations
            ]

            # Stage 5: Log to audit trail
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            await db.execute(
                """
                INSERT INTO queries (query_id, user_id, question, mode, model_used, chunk_ids, answer, latency_ms, token_cost)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                query_id,
                "system",  # user_id - would come from auth in production
                request.question,
                request.mode,
                model_used,
                ",".join(chunk_ids),
                answer,
                (time.time() - total_start) * 1000,
                0,  # token_cost - would be calculated from LiteLLM metadata
            )

            if lf_trace is not None:
                # Record final output + per-stage latencies on the Langfuse trace for the dashboard
                lf_trace.update(
                    output=answer,
                    metadata={
                        "retrieval_latency_ms": retrieval_latency_ms,
                        "generation_latency_ms": generation_latency_ms,
                        "model_used": model_used,
                    },
                )

            return AskResponse(
                answer=answer,
                sources=sources,
                mode=request.mode,
                model_used=model_used,
                latency_ms=(time.time() - total_start) * 1000,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
            )

        except HTTPException:
            # Re-raise as-is (e.g. the 400 from guard validation above) — don't let the
            # generic handler below wrap it into a misleading 500.
            raise

        except Exception as e:
            # Log failed query attempt
            try:
                await db.execute(
                    """
                    INSERT INTO queries (query_id, user_id, question, mode, model_used, chunk_ids, answer, latency_ms, token_cost)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    query_id,
                    "system",
                    request.question,
                    request.mode,
                    "error",
                    "",
                    f"Error: {str(e)}",
                    (time.time() - total_start) * 1000,
                    0,
                )
            except Exception:
                pass  # Don't fail if logging fails — the original error is what the caller needs to see

            raise HTTPException(
                status_code=500, detail=f"Query processing failed: {str(e)}"
            )
