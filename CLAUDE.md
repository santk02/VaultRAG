# VaultRAG

Privacy-first document Q&A for regulated industries: hybrid retrieval (BM25 + vectors → RRF → cross-encoder rerank) over private PDFs/DOCX, answers with page-level citations, switchable between cloud (Claude via LiteLLM) and fully offline (Ollama) generation with `MODE`. `VAULTRAG_BLUEPRINT.md` is the original spec; `ARCHITECTURE.md`, `FAILURE_MODES.md`, `EVALUATION.md` describe decisions, degradation and eval.

## Stack

Python 3.11 · FastAPI + Pydantic v2 / pydantic-settings · asyncpg (PostgreSQL 16) · Qdrant · `rank_bm25` (pickled index) · sentence-transformers (MiniLM embeddings, cross-encoder reranker) · LiteLLM (Claude / Ollama) · Guardrails AI · Langfuse + OpenTelemetry · RAGAS · pytest · ruff / black / mypy · Docker Compose.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # Windows: setup_env.bat

# Services (Qdrant :6333, Postgres :5432, Langfuse :3000)
docker compose up -d qdrant postgres      # `docker compose up -d` also starts the app container

# Run the API (reload follows DEBUG)
python -m app.main                        # or: uvicorn app.main:app --reload
curl http://localhost:8000/health         # docs at /docs

# Test
pytest tests/ -v
pytest tests/test_retrieval.py::test_rrf_basic -v     # single test

# Lint / format / types (exactly what CI runs — scope is app/ only)
ruff check app/
black --check app/                        # `black app/` to fix
mypy app/ --ignore-missing-imports

# Evaluation and benchmarks (need live services + indexed docs)
python evaluation/run_ragas.py
python scripts/check_thresholds.py        # CI gate: exits 1 below evaluation/thresholds.json
python benchmarks/retrieval_ablation.py
python benchmarks/model_benchmark.py
```

Offline mode needs Ollama running with the models pulled: `ollama pull mistral` (default `OFFLINE_MODEL`).

## Architecture map

- `app/main.py` wires routers; `app/config.py` holds the single `settings` object; `app/db.py` is the asyncpg pool (`db`); `app/models.py` holds all request/response schemas.
- `app/ingestion/` parse → chunk → embed → index (Qdrant + Postgres + BM25 pickle). `router.py` = `POST /v1/documents/upload`.
- `app/retrieval/` `bm25_search`, `vector_search`, `fusion` (RRF), `reranker`, `pipeline` (orchestrates the 4 stages and per-stage latencies). `api/retrieve_router.py` = `POST /v1/retrieve`.
- `app/generation/` `prompts` (citation-enforcing system prompt), `llm` (LiteLLM), `citation` (verifies/strips citations), `guards` (Guardrails). `api/ask_router.py` = `POST /v1/ask` and writes the audit row.
- `app/extraction/` structured extraction, `POST /v1/extract`. `app/observability/tracing.py` OTel spans + guarded Langfuse.
- Outside `app/`: `evaluation/` (RAGAS, thresholds), `benchmarks/`, `fine_tuning/` (QLoRA scaffold, separate `requirements.txt`), `scripts/` (`init_db.sql`, bulk ingest, CI threshold gate), `tests/`.

## Non-negotiable invariants

- **`page_number` must survive parse → chunk → index → retrieve → generate.** Citations are only real if it does.
- Citation format is exactly `[Source: filename, page N]`; the refusal string is exactly `I don't have enough information in the provided documents to answer that.` — `citation.py`, `guards.py` and tests match on these.
- Retrieval is identical in both modes; only the generation backend changes. Never add a code path that sends document text off-box in `offline` mode.
- Heavy models (embedder, reranker, Qdrant client) load once per process via module-level `get_*()` accessors — never per request.

## Code style

- Formatting is `black` defaults (88 cols); `ruff check` must be clean. Keep imports grouped stdlib / third-party / `app.*`.
- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on every public function and class; short inline `#` comments that explain *why*, not what.
- `snake_case` functions/modules, `PascalCase` classes, `UPPER_SNAKE` constants. Test files `test_<module>.py`, test functions `test_<behavior>` with a one-line docstring.
- I/O-bound code is `async`; routers use `APIRouter(prefix="/v1/...", tags=[...])` and Pydantic models from `app/models.py`.
- Config only through `app.config.settings` (env names are the upper-cased field names); add new knobs to both `config.py` and `.env.example`.
- Process-wide objects use the singleton pattern already in the repo: `_thing: Thing = None` plus `get_thing()`.
- Type hints: match the surrounding file (`typing.List/Optional` or builtin `tuple[...]`); don't mix styles within a file.
- IDs: `doc_id` is a UUID4 string; `chunk_id` is `f"{doc_id}_chunk_{index}"`.

## Known gotchas

- No `pyproject.toml`/`pytest.ini`/Makefile — tool config lives in CI and `requirements.txt`.
- CI runs `pytest --cov=app` but `pytest-cov` is **not** in `requirements.txt`; the CI test job will fail until it's added.
- `mypy app/` reports ~29 pre-existing "implicit Optional" errors (`x: int = None` signatures); the CI lint job is red until those are fixed. Don't add new ones — write `Optional[int] = None`.
- `tests/test_health.py` expects `status == "ok"`, which requires live Postgres + Qdrant (CI provides both as service containers).
- `evaluation/run_ragas.py` is a mock (fixed scores), `evaluation/test_set.json` has 5 questions (blueprint says 50), `scripts/bulk_ingest.py` doesn't call the API yet, and `data/sources.md` is a stub. Don't present their output as real measurements.
- Scanned PDFs (no text layer) are rejected at parse time by design; table extraction and OCR are documented non-goals.
- The BM25 index is a pickle at `BM25_INDEX_PATH`, rebuilt in full on every upload — fine for a demo corpus, not for scale.

## Working agreements

- Keep `README.md`, `ARCHITECTURE.md` and `.env.example` in sync with `config.py` when defaults change (they drifted once already).
- Don't commit documents, `.env`, pickles or `temp/` (all gitignored).
- Prefer the simplest thing that demonstrates the concept (blueprint design rule); no abstraction you can't defend.
