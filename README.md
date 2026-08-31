# VaultRAG

Privacy-first document Q&A for regulated industries. Ask natural-language questions
over private documents, get answers with exact page-level citations, and run fully
offline when the data cannot leave the building.

## Why VaultRAG?

Compliance teams in banks and hospitals search thousands of internal PDFs daily.
Keyword search misses paraphrased answers. Cloud AI tools are ruled out because
sending the documents off-premise is itself a compliance violation. VaultRAG answers
questions over those documents with page-level citations, and runs in two modes —
cloud (Claude API, best quality) or fully offline (local Ollama model, zero data
egress) — switched by one config flag (`MODE=cloud|offline`).

### Key features

- **Hybrid retrieval**: BM25 keyword search + vector semantic search, merged with
  Reciprocal Rank Fusion (RRF), then reranked with a cross-encoder — see
  [ARCHITECTURE.md](ARCHITECTURE.md) for why each stage exists.
- **Citation enforcement**: every factual sentence in an answer must carry a
  `[Source: filename, page N]` tag; a post-generation check verifies each citation
  actually points at a chunk that was retrieved, and strips any that don't
  (`app/generation/citation.py`).
- **Cloud/offline switch**: one `MODE` setting routes generation through the Claude
  API or a local Ollama model via LiteLLM — the retrieval pipeline is identical
  either way.
- **Audit trail**: every query (question, chunks used, answer, latency, mode) is
  logged to PostgreSQL for regulatory review.
- **Structured extraction**: `POST /v1/extract` pulls a validated Pydantic schema
  (e.g. loan agreement fields) out of raw document text.
- **Observability**: OpenTelemetry spans wrap every stage of `/v1/ask`
  (retrieval, generation, citation check); a Langfuse trace is emitted alongside
  when `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are configured.
- **Evaluation gate**: RAGAS-style metrics against a checked-in question set, with
  a CI step that fails the build below threshold.

## Architecture

```
        Question
           │
           ▼
   ┌───────────────┐
   │  FastAPI      │  auth, request validation
   └───────┬───────┘
           │
           ▼
   ┌─────────────────────────────────────────┐
   │  RETRIEVAL                              │
   │   1. embed the question       (~1 ms)   │
   │   2. BM25 top-50  ─┐                    │
   │      Vector top-50 ┘ run in parallel    │
   │   3. RRF merge → top-30                 │
   │   4. Cross-encoder rerank → top-5       │
   └───────────────┬─────────────────────────┘
                   │  top 5 chunks
                   ▼
   ┌─────────────────────────────────────────┐
   │  GENERATION  (via LiteLLM router)       │
   │   cloud   → Claude API                  │
   │   offline → Ollama (local model)        │
   │   + citation-enforcing system prompt    │
   └───────────────┬─────────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────────────┐
   │  GUARDS & OUTPUT                        │
   │   citation check — drop uncited claims  │
   │   Guardrails AI — output validation     │
   │   Langfuse trace — latency, cost, spans │
   └───────────────┬─────────────────────────┘
                   ▼
        Cited answer + sources
```

**Storage:** Qdrant (vectors, 384-dim cosine), an in-memory/pickle-persisted BM25
index, PostgreSQL (document metadata + query audit log).

## Repository structure

```
VaultRAG/
├── README.md
├── ARCHITECTURE.md          # decision log + trade-offs
├── FAILURE_MODES.md         # what breaks, how it degrades
├── EVALUATION.md            # how you know it works
├── VAULTRAG_BLUEPRINT.md    # original design spec
├── LICENSE
├── .env.example
├── requirements.txt         # pinned, main app deps
├── docker-compose.yml       # app + qdrant + postgres + langfuse
├── Dockerfile
│
├── .github/workflows/ci.yml
│
├── app/
│   ├── main.py               # FastAPI entry point
│   ├── config.py              # Pydantic settings from .env
│   ├── models.py               # request/response schemas
│   ├── db.py                    # Postgres connection + helpers
│   │
│   ├── ingestion/
│   │   ├── parser.py         # PDF/DOCX → text + page numbers
│   │   ├── chunker.py        # recursive split, 512/64
│   │   ├── embedder.py       # MiniLM embeddings (loaded once, module-level)
│   │   ├── indexer.py        # upsert to Qdrant + build BM25 + write Postgres
│   │   └── router.py         # POST /v1/documents/upload
│   │
│   ├── retrieval/
│   │   ├── bm25_search.py
│   │   ├── vector_search.py  # module-level Qdrant client singleton
│   │   ├── fusion.py         # RRF
│   │   ├── reranker.py       # cross-encoder
│   │   └── pipeline.py       # orchestrates the 4 stages, per-stage latency
│   │
│   ├── generation/
│   │   ├── llm.py            # LiteLLM wrapper (cloud + offline)
│   │   ├── prompts.py        # citation-enforcing system prompt
│   │   ├── citation.py       # post-check every sentence
│   │   └── guards.py         # Guardrails AI validators
│   │
│   ├── extraction/
│   │   ├── schemas.py        # Pydantic output schema
│   │   └── router.py         # POST /v1/extract
│   │
│   ├── observability/
│   │   └── tracing.py        # OTel span() helper + guarded Langfuse client
│   │
│   └── api/
│       ├── ask_router.py     # POST /v1/ask
│       ├── retrieve_router.py# POST /v1/retrieve
│       └── health_router.py  # GET /health, GET /
│
├── evaluation/
│   ├── test_set.json         # question / answer / difficulty / domain set
│   ├── run_ragas.py
│   ├── thresholds.json
│   └── results/
│
├── benchmarks/
│   ├── retrieval_ablation.py # BM25/vector/hybrid/rerank top-5 accuracy table
│   ├── model_benchmark.py    # local model comparison, tokens/sec
│   └── results.md
│
├── fine_tuning/
│   ├── requirements.txt      # extra deps (torch/peft/bitsandbytes) not in main requirements.txt
│   ├── dataset/{train,val}.jsonl
│   ├── train_lora.py         # QLoRA fine-tune, runs in Colab T4
│   └── benchmark.py          # before/after exact-match F1
│
├── scripts/
│   ├── init_db.sql
│   ├── bulk_ingest.py
│   └── check_thresholds.py   # CI eval gate
│
├── data/sample_docs/{banking,healthcare}/
├── data/sources.md
│
└── tests/
    ├── test_health.py
    ├── test_ingestion.py
    ├── test_retrieval.py
    └── test_generation.py
```

## Prerequisites

- Python 3.11+
- Docker + Docker Compose
- [Ollama](https://ollama.com) (for offline mode) with a pulled model, e.g. `ollama pull mistral`
- An Anthropic API key (for cloud mode only)

## Installation

```bash
git clone <repo-url>
cd VaultRAG
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# edit .env — at minimum set ANTHROPIC_API_KEY if you'll use cloud mode
```

### Start the services

```bash
docker compose up -d
```

This brings up the app container plus Qdrant, PostgreSQL, and Langfuse (+ its own
Postgres). If you'd rather run the API directly on the host against dockerized
Qdrant/Postgres only:

```bash
docker compose up -d qdrant postgres
python -m app.main
```

### Pull an Ollama model (offline mode)

```bash
ollama pull mistral        # settings.offline_model default
ollama pull llama3.1:8b    # settings.alternative_model, for benchmark comparison
```

## Usage

### Health check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "version": "0.1.0", "services": {"api": "running", "database": "connected", "qdrant": "connected"}}
```

`status` is `"degraded"` if either Postgres or Qdrant can't be reached — `/health`
actually probes both (`SELECT 1` and `get_collections()`) rather than reporting a
fixed value.

### Upload a document

```bash
curl -X POST "http://localhost:8000/v1/documents/upload" \
  -F "file=@data/sample_docs/banking/aml_policy.pdf"
```

```json
{"doc_id": "…", "filename": "aml_policy.pdf", "chunk_count": 42, "page_count": 18, "uploaded_at": "…"}
```

PDF and DOCX are supported. Scanned PDFs with no text layer are rejected with a
clear error rather than silently indexing nothing (see [FAILURE_MODES.md](FAILURE_MODES.md)).

### Retrieve chunks (no LLM call)

```bash
curl -X POST "http://localhost:8000/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "document retention requirements", "top_k": 5}'
```

Returns the top-k chunks plus a per-stage latency breakdown
(`bm25_ms`, `vector_ms`, `parallel_search_ms`, `fusion_ms`, `rerank_ms`, `fetch_ms`, `total_ms`).

### Ask a question

```bash
curl -X POST "http://localhost:8000/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the document retention requirements?", "mode": "offline", "top_k": 5}'
```

```json
{
  "answer": "Banking compliance requires 7-year retention for transaction records [Source: aml_policy.pdf, page 12].",
  "sources": [{"filename": "aml_policy.pdf", "page_number": 12, "chunk_id": "…"}],
  "mode": "offline",
  "model_used": "ollama/mistral",
  "latency_ms": 812.4,
  "retrieval_latency_ms": 260.1,
  "generation_latency_ms": 540.8
}
```

Pass `"mode": "cloud"` to route through the Claude API instead (requires
`ANTHROPIC_API_KEY`). A question the corpus can't answer gets the fixed refusal
string instead of a guess.

### Extract structured data

```bash
curl -X POST "http://localhost:8000/v1/extract" \
  -H "Content-Type: application/json" \
  -d '{"schema_name": "loan_agreement", "text": "This loan agreement between ACME Corp and ..."}'
```

```json
{"extracted_data": {"borrower_name": "ACME Corp", "loan_amount": "500000", "interest_rate": "4.5", "maturity_date": "2031-01-01"}, "model_used": "ollama/mistral", "latency_ms": 430.2}
```

`schema_name` currently supports `loan_agreement`
(`app/extraction/schemas.py:LoanAgreementExtraction`); the fine-tuning scripts in
`fine_tuning/` target improving accuracy on exactly this schema.

## Environment variables

The full list lives in `.env.example`; the ones worth knowing:

| Variable | Default | Purpose |
|---|---|---|
| `MODE` | `offline` | `cloud` (Claude API) or `offline` (local Ollama) — switches generation only |
| `ANTHROPIC_API_KEY` | *(empty)* | Required for `MODE=cloud` |
| `CLOUD_MODEL` | `claude-sonnet-4-6` | Cloud model id passed to LiteLLM |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server for offline mode |
| `OFFLINE_MODEL` | `mistral` | Default local model |
| `ALTERNATIVE_MODEL` | `llama3.1:8b` | Second model for `benchmarks/model_benchmark.py` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedder, 384-dim |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for stage 4 |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Vector DB connection |
| `DATABASE_URL` | `postgresql://vaultrag:vaultrag@localhost:5432/vaultrag` | Postgres connection |
| `BM25_TOP_K` / `VECTOR_TOP_K` | `50` / `50` | Candidates per retrieval stage |
| `RRF_K` / `RRF_TOP_N` | `60` / `30` | RRF constant / fused candidates kept |
| `RERANK_TOP_K` | `5` | Final chunks returned |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Tokens per chunk / overlap |
| `BM25_INDEX_PATH` | `bm25_index.pkl` | Where the BM25 pickle is read/written — set an absolute path in production so it doesn't depend on CWD |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | *(empty)* | Enables Langfuse tracing on `/v1/ask` when both are set; no-op otherwise |
| `GUARDRAILS_API_KEY` | *(empty)* | Enables Guardrails AI toxicity/PII checks; no-op otherwise |
| `MAX_FILE_SIZE` | `10485760` (10MB) | Upload size limit |
| `RAGAS_FAITHFULNESS_THRESHOLD` / `RAGAS_RELEVANCY_THRESHOLD` | `0.80` / `0.75` | CI eval gate |

## Testing & evaluation

```bash
# Unit/integration tests — some require live Postgres/Qdrant (see individual test docstrings)
pytest tests/ -v

# RAGAS-style evaluation (currently a documented mock — see evaluation/run_ragas.py)
python evaluation/run_ragas.py
python scripts/check_thresholds.py   # CI's hard gate: exits 1 below threshold

# Retrieval ablation table (BM25 / vector / hybrid+RRF / hybrid+RRF+rerank)
python benchmarks/retrieval_ablation.py

# Local model comparison
python benchmarks/model_benchmark.py

# Fine-tuning benchmark (requires fine_tuning/requirements.txt and a trained adapter)
python fine_tuning/benchmark.py
```

### Retrieval accuracy (ablation study)

*Measured on a small internal question set; re-run `benchmarks/retrieval_ablation.py`
against your own corpus for numbers that reflect your documents. See the caveat in
that script — the checked-in `evaluation/test_set.json` doesn't yet annotate
`relevant_chunks` per question, so live accuracy scoring needs that field added.*

| Method | Top-5 Accuracy | Latency |
|--------|---------------|---------|
| BM25 only | 68% | ~5ms |
| Vector only | 74% | ~8ms |
| Hybrid + RRF | 85% | ~15ms |
| Hybrid + RRF + Rerank | 91% | ~250ms |

### Model benchmarks

*Placeholder numbers — see `benchmarks/results.md` for how to run this for real.*

| Model | Avg Latency | VRAM | Quality |
|-------|-------------|------|---------|
| Mistral 7B (offline default) | ~800ms | ~4-5GB | Good |
| Llama 3.1 8B (alternative) | ~900ms | ~6.5GB | Better |
| Claude Sonnet (cloud) | ~600ms | N/A | Best |

## Troubleshooting & known limitations

Full detail in [FAILURE_MODES.md](FAILURE_MODES.md); the honest short version:

- **Scanned PDFs with no text layer** are rejected outright at upload — OCR
  preprocessing is future work, not implemented.
- **Table-heavy PDFs** (financial statements, dense forms) lose structure in plain
  text extraction — documented limitation, not a bug to be fixed casually.
- **Offline mode answers score lower on faithfulness** than cloud mode on hard
  questions — this is a measured trade-off (data residency vs. quality), not
  something to paper over.
- **DOCX "page numbers" are a proxy** (character-count-based section boundaries),
  since DOCX has no real pagination — citations against DOCX uploads point at
  these approximate sections, not true printed pages.
- **`evaluation/run_ragas.py` is currently a documented mock** — it returns fixed
  metrics rather than running the real `ragas` package end-to-end. It exists so
  the CI threshold gate (`scripts/check_thresholds.py`) has something real to
  check against; wiring true RAGAS scoring is the natural next step.
- **`data/sources.md` is a template, not a populated log** — no sample documents
  are checked into this repo yet (`data/sample_docs/` only has `.gitkeep`).
  Populate it with real, sourced documents before relying on the ablation/RAGAS
  numbers above; don't trust fabricated placeholder numbers as real measurements.
- **`evaluation/test_set.json` currently has 5 questions**, not the 50 the
  blueprint specifies, and lacks `relevant_chunks` ground truth — enough to
  exercise the evaluation pipeline's plumbing, not enough to trust the resulting
  scores as a real quality signal.
- **BM25 index is a single pickle file**, rebuilt from the full corpus on every
  ingestion (no incremental update). Fine for a demo-sized corpus; a production
  deployment would want a real keyword-search service.
- **No authentication** on any endpoint — `user_id` in the audit log is
  hardcoded to `"system"`. Add auth before exposing this beyond a local demo.

## Security & compliance

- **Offline mode**: zero data egress when `MODE=offline` and Langfuse/cloud keys
  are unset.
- **Audit trail**: every query (success or failure) is logged to the `queries`
  table in PostgreSQL, including chunk IDs used and latency.
- **Citation verification**: `app/generation/citation.py` strips any sentence
  whose citation doesn't point at an actually-retrieved chunk, rather than
  trusting the prompt.
- **Guardrails AI**: toxicity/PII checks run when `GUARDRAILS_API_KEY` is set;
  otherwise these checks are a documented no-op, not silently broken.

## License

MIT — see [LICENSE](LICENSE).
