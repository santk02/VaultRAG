#!/usr/bin/env python3
"""
Retrieval ablation study for VaultRAG (blueprint Phase 2).

Runs the evaluation question set through four retrieval configurations —
BM25-only, vector-only, hybrid+RRF, and hybrid+RRF+rerank — and reports
top-5 accuracy per stage, plus average latency. This is the "68% -> 91%"
table referenced throughout ARCHITECTURE.md and README.md.

Requires live Qdrant + Postgres with documents already ingested (the same
corpus the test set's `relevant_chunks`/expected answers were written against)
and a populated BM25 pickle (see app/ingestion/indexer.py). It is NOT run in
CI because it needs real indexed documents, not just services running —
document that limitation to whoever wires it into CI later.

Usage:
    python benchmarks/retrieval_ablation.py
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

# Reuse the actual retrieval building blocks — this measures the real pipeline,
# not a simulation of it.
from app.retrieval.bm25_search import bm25_search
from app.retrieval.fusion import fuse_results
from app.retrieval.reranker import rerank_chunks
from app.retrieval.vector_search import vector_search

TOP_K = 5
TEST_SET_PATH = Path("evaluation/test_set.json")
OUTPUT_PATH = Path("benchmarks/ablation_results.md")


def load_questions() -> List[dict]:
    """Load the shared eval question set so this ablation and RAGAS score the same questions."""
    with open(TEST_SET_PATH) as f:
        data = json.load(f)
    return data.get("questions", [])


def _is_hit(question: dict, chunk_ids: List[str]) -> bool:
    """
    A "hit" means a chunk expected to be relevant for this question is present in
    the top-K result. Falls back gracefully if the question has no `relevant_chunks`
    field annotated (current evaluation/test_set.json does not — see README caveat).
    """
    expected = set(question.get("relevant_chunks", []))
    if not expected:
        # No ground-truth chunk IDs annotated for this question — can't score a hit,
        # so it's excluded from the accuracy denominator rather than counted as a miss.
        return None
    return bool(expected & set(chunk_ids))


async def run_bm25_only(question: str) -> List[str]:
    """Stage 1: keyword search alone, no fusion or reranking."""
    results = await bm25_search(question)
    return [chunk_id for chunk_id, _ in results[:TOP_K]]


async def run_vector_only(question: str) -> List[str]:
    """Stage 2: semantic search alone."""
    results = await vector_search(question)
    return [chunk_id for chunk_id, _ in results[:TOP_K]]


async def run_hybrid_rrf(question: str) -> List[str]:
    """Stage 3: BM25 + vector merged by RRF, no reranking — top-K of the fused list."""
    bm25_results, vector_results = await asyncio.gather(
        bm25_search(question), vector_search(question)
    )
    fused = fuse_results(bm25_results, vector_results)
    return fused[:TOP_K]


async def run_hybrid_rrf_rerank(question: str) -> List[str]:
    """Stage 4: the full production pipeline — hybrid + RRF + cross-encoder rerank."""
    bm25_results, vector_results = await asyncio.gather(
        bm25_search(question), vector_search(question)
    )
    fused = fuse_results(bm25_results, vector_results)
    reranked = await rerank_chunks(question, fused, TOP_K)
    return [chunk_id for chunk_id, _ in reranked]


STAGES = {
    "BM25 only": run_bm25_only,
    "Vector only": run_vector_only,
    "Hybrid + RRF": run_hybrid_rrf,
    "Hybrid + RRF + Rerank": run_hybrid_rrf_rerank,
}


async def run_stage(name: str, fn, questions: List[dict]) -> Dict:
    """Run one retrieval configuration over every question and aggregate accuracy/latency."""
    hits = 0
    scored = 0  # questions with ground-truth chunk annotations we could actually score
    total_latency_ms = 0.0

    for q in questions:
        start = time.time()
        try:
            chunk_ids = await fn(q["question"])
        except Exception as e:
            print(f"  [{name}] error on '{q['question'][:40]}...': {e}")
            continue
        total_latency_ms += (time.time() - start) * 1000

        hit = _is_hit(q, chunk_ids)
        if hit is not None:
            scored += 1
            hits += int(hit)

    accuracy = hits / scored if scored else float("nan")
    avg_latency_ms = total_latency_ms / len(questions) if questions else 0.0
    return {
        "stage": name,
        "top5_accuracy": accuracy,
        "scored_questions": scored,
        "avg_latency_ms": avg_latency_ms,
    }


def render_markdown(results: List[Dict]) -> str:
    """Render the ablation table exactly as it's meant to be pasted into README.md."""
    lines = [
        "# Retrieval Ablation Results",
        "",
        "| Stage | Top-5 Accuracy | Avg Latency | Scored Questions |",
        "|---|---|---|---|",
    ]
    for r in results:
        acc = (
            "n/a"
            if r["top5_accuracy"] != r["top5_accuracy"]
            else f"{r['top5_accuracy']*100:.0f}%"
        )
        lines.append(
            f"| {r['stage']} | {acc} | {r['avg_latency_ms']:.0f}ms | {r['scored_questions']} |"
        )
    return "\n".join(lines) + "\n"


async def main():
    print("VaultRAG Retrieval Ablation Study")
    print("=" * 60)

    questions = load_questions()
    print(f"Loaded {len(questions)} questions from {TEST_SET_PATH}")
    if not questions:
        print("No questions found — nothing to run.")
        return

    results = []
    for name, fn in STAGES.items():
        print(f"\nRunning: {name}")
        result = await run_stage(name, fn, questions)
        results.append(result)
        acc = result["top5_accuracy"]
        acc_str = (
            "n/a (no relevant_chunks annotated)" if acc != acc else f"{acc*100:.0f}%"
        )
        print(
            f"  Top-5 accuracy: {acc_str}  |  Avg latency: {result['avg_latency_ms']:.0f}ms"
        )

    table = render_markdown(results)
    print("\n" + table)

    OUTPUT_PATH.write_text(table)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
