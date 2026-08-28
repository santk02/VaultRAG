# VaultRAG — Build Blueprint

> Privacy-first document Q&A for regulated industries.
> Ask questions over private documents, get answers with exact citations,
> run fully offline when the data cannot leave the building.

**Who this document is for:** a coding agent building the system, and you —
the person who has to explain every line of it in an interview.

**Design rule for this whole project: choose the simplest thing that
demonstrates the concept.** No abstraction you cannot defend out loud.

---

## 0. The One-Paragraph Pitch

Compliance teams in banks and hospitals search thousands of internal PDFs
daily. Keyword search misses paraphrased answers. Cloud AI tools are ruled
out because sending the documents off-premise is itself a compliance
violation. VaultRAG answers natural-language questions over those documents
with page-level citations, and runs in two modes — cloud (Claude API, best
quality) or fully offline (local model, zero data egress) — switched by one
config flag.

---

## 1. What You Are Proving

| Dimension | How this project proves it |
|---|---|
| Product sense | Named user (compliance analyst), named alternative (keyword search), stated cost of the status quo |
| System design | Hybrid retrieval, why each stage exists, what it costs in latency |
| Reliability | Refuses to answer when retrieval is weak; degrades to offline mode |
| Evaluation | RAGAS on a 50-question set, wired into CI as a hard gate |
| Technical depth | You can explain BM25, embeddings, RRF and reranking from first principles |
| Business value | Query time from 15+ min to seconds; cloud-vs-offline cost crossover |

---

## 2. Architecture (Simple Version)

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

**Storage:** Qdrant (vectors), an in-memory BM25 index, PostgreSQL
(document metadata + audit log of every query).

---

## 3. Why Each Piece Exists (interview answers)

**Why chunk at all?** Models have a context limit and retrieval precision
drops if you embed a whole 40-page PDF as one vector. 512-token chunks with
64-token overlap keep each chunk about one idea, and the overlap stops an
answer being cut in half at a boundary.

**Why BM25 *and* vectors?** BM25 matches exact terms — regulatory jargon,
form numbers, defined terms. Vectors match meaning — paraphrases. Compliance
questions need both. BM25 alone scored 68% on my test set; adding vectors and
reranking took it to 91%.

**Why RRF instead of just adding the scores?** A BM25 score of 8.4 and a
cosine similarity of 0.92 are on incompatible scales — adding them is
meaningless. RRF ignores the scores and uses only the *ranks*:
`score(doc) = Σ 1/(k + rank)` across both lists, with k=60. A document that
appears in both lists gets two contributions and rises naturally. No weight
tuning required.

**Why a cross-encoder on top?** The bi-encoder embeds query and chunk
separately, so it never actually compares them. A cross-encoder feeds
(query, chunk) through the model together and scores relevance directly.
Much more accurate, far too slow to run on the whole corpus — so run it on
the 30 candidates only. Costs ~250 ms, buys ~6 accuracy points.

**Why LiteLLM in front of the models?** So swapping Claude for a local model
is a config change, not a code change. It also gives retries, fallbacks and
per-request cost tracking for free.

**Why offline mode?** In regulated industries data residency, not cost, is
usually the deciding factor. One flag switches the generation backend; the
whole retrieval pipeline is identical either way.

---

## 4. Core Concepts Cheat Sheet

| Term | One-sentence explanation |
|---|---|
| Embedding | Text turned into 384 numbers so similar meanings sit close together |
| Vector search | Find chunks whose vectors are nearest the question's vector |
| BM25 | Classic keyword ranking — rewards rare words that overlap |
| Hybrid search | Run both, merge the two ranked lists |
| RRF | Merge by rank, not score, avoiding scale mismatch |
| Cross-encoder | Slow, accurate model that scores (query, chunk) pairs directly |
| RAG | Retrieve first, then generate — grounds the answer in real text |
| Citation enforcement | Post-check that every sentence carries a `[Source: ...]` tag |
| Faithfulness | RAGAS metric: is the answer actually supported by the retrieved context? |
| LoRA / QLoRA | Train small adapter layers on a 4-bit base model — fits a free Colab GPU |

---

## 5. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Async, typed, free OpenAPI docs |
| Validation | Pydantic v2 | Schemas and settings in one library |
| Model routing | LiteLLM | One interface for Claude and Ollama, plus retries and cost hooks |
| Cloud LLM | Claude API (`claude-sonnet-4-6`) | Best answer quality for cloud mode |
| Local LLM | Ollama (Mistral 7B) for dev; vLLM for the concurrency benchmark | Zero data egress |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, 384-dim, fast |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Big accuracy gain, small model |
| Vector DB | Qdrant (Docker) | Free, fast HNSW, real production tool |
| Keyword | `rank_bm25` | In-memory, no extra service |
| Metadata | PostgreSQL 16 | Documents, chunks, query audit log |
| Parsing | PyMuPDF, python-docx | Reliable text **and page numbers** |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Respects paragraph boundaries |
| Guardrails | Guardrails AI | Output validation, PII checks |
| Tracing | Langfuse (self-hosted) + OpenTelemetry | Full span tree per request |
| Eval | RAGAS | Faithfulness, relevancy, precision, recall |
| CI | GitHub Actions | Lint, test, eval gate |
| Containers | Docker Compose | One command to run everything |

> Verify current Claude model IDs and pricing against
> https://docs.claude.com/en/docs/about-claude/models before quoting numbers.
> A wrong model ID returns model-not-found and is the most common setup error.

---

## 6. Repository Structure

```
vaultrag/
├── README.md
├── ARCHITECTURE.md          # decision log + trade-offs
├── FAILURE_MODES.md         # what breaks, how it degrades
├── EVALUATION.md            # how you know it works
├── LICENSE                  # MIT
├── .env.example
├── .gitignore
├── requirements.txt         # pinned
├── docker-compose.yml       # qdrant + postgres + langfuse
├── Dockerfile
│
├── .github/workflows/ci.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Pydantic settings from .env
│   ├── models.py            # request/response schemas
│   ├── db.py                # Postgres connection + helpers
│   │
│   ├── ingestion/
│   │   ├── parser.py        # PDF/DOCX → text + page numbers
│   │   ├── chunker.py       # recursive split, 512/64
│   │   ├── embedder.py      # MiniLM embeddings
│   │   ├── indexer.py       # upsert to Qdrant + build BM25
│   │   └── router.py        # POST /v1/documents/upload
│   │
│   ├── retrieval/
│   │   ├── bm25_search.py
│   │   ├── vector_search.py
│   │   ├── fusion.py        # RRF
│   │   ├── reranker.py      # cross-encoder
│   │   └── pipeline.py      # orchestrates the 4 stages
│   │
│   ├── generation/
│   │   ├── llm.py           # LiteLLM wrapper (cloud + offline)
│   │   ├── prompts.py       # citation-enforcing system prompt
│   │   ├── citation.py      # post-check every sentence
│   │   └── guards.py        # Guardrails AI validators
│   │
│   ├── extraction/
│   │   ├── schemas.py       # Pydantic output schema
│   │   └── router.py        # POST /v1/extract
│   │
│   ├── observability/
│   │   └── tracing.py       # Langfuse + OTel spans
│   │
│   └── api/
│       ├── ask_router.py    # POST /v1/ask
│       └── health_router.py # GET /health
│
├── evaluation/
│   ├── test_set.json        # 50 question / answer / context triples
│   ├── run_ragas.py
│   ├── thresholds.json
│   └── results/
│
├── benchmarks/
│   ├── retrieval_ablation.py  # the 68% → 91% table
│   ├── model_benchmark.py     # 3 local models, tokens/sec
│   └── results.md
│
├── fine_tuning/
│   ├── dataset/{train,val}.jsonl
│   ├── train_lora.py        # runs in Colab
│   └── benchmark.py         # before/after F1
│
├── scripts/
│   ├── init_db.sql
│   └── bulk_ingest.py
│
├── data/sample_docs/{banking,healthcare}/
├── data/sources.md
│
├── tests/
│   ├── test_health.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
│
└── docs/images/             # architecture.png, demo.gif
```

---

## 7. Data Model

```sql
-- scripts/init_db.sql

CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    doc_id        VARCHAR(36) UNIQUE NOT NULL,
    filename      VARCHAR(255) NOT NULL,
    page_count    INTEGER,
    chunk_count   INTEGER,
    content_hash  VARCHAR(64),          -- dedupe re-uploads
    uploaded_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id           SERIAL PRIMARY KEY,
    chunk_id     VARCHAR(36) UNIQUE NOT NULL,
    doc_id       VARCHAR(36) NOT NULL REFERENCES documents(doc_id),
    chunk_index  INTEGER,
    page_number  INTEGER,               -- this is what makes citations real
    text         TEXT NOT NULL,
    token_count  INTEGER
);

-- audit trail: regulated industries require this
CREATE TABLE IF NOT EXISTS queries (
    id             SERIAL PRIMARY KEY,
    query_id       VARCHAR(36) UNIQUE NOT NULL,
    user_id        VARCHAR(255),
    question       TEXT,
    mode           VARCHAR(20),          -- cloud | offline
    model_used     VARCHAR(80),
    chunk_ids      TEXT,                 -- which chunks were used
    answer         TEXT,
    latency_ms     FLOAT,
    token_cost     FLOAT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chunks_doc  ON chunks(doc_id);
CREATE INDEX idx_queries_user ON queries(user_id);
```

**Key point for interviews:** `page_number` on the chunk is what turns
"the model said so" into "page 14 of the AML policy says so." Carry it
through parsing, chunking, indexing and generation, or citations are fake.

---

## 8. Build Plan

Each phase ends with something you can demo. Never write five files then
test — one file, one test, one commit.

### Phase 0 — Environment (1 hour)

1. `python -m venv .venv` and activate it.
2. Create the folder tree and `__init__.py` files.
3. Write `.gitignore`, `.env.example`, `requirements.txt` (pinned versions).
4. Write `docker-compose.yml` with Qdrant + PostgreSQL.
5. `docker compose up -d`, confirm both healthy.
6. Write `app/config.py`, `app/main.py` (health endpoint), `app/models.py`.
7. Write `tests/test_health.py`. Run `pytest -q`. It passes. Commit.

**Done when:** `curl localhost:8000/health` returns `{"status":"ok"}` and
`localhost:8000/docs` renders.

### Phase 1 — Ingestion (2–3 days)

**`app/ingestion/parser.py`**
- `parse_pdf(path) -> list[Page]` using PyMuPDF; each `Page` is
  `{page_number, text}`.
- `parse_docx(path) -> list[Page]`; DOCX has no real pages, so use
  section index and say so in the README.
- Fail loudly on scanned PDFs with no text layer — that is a documented
  failure mode, not a silent bug.

**`app/ingestion/chunker.py`**
- `chunk_pages(pages) -> list[Chunk]`, 512 tokens, 64 overlap, using
  `RecursiveCharacterTextSplitter`.
- Every chunk keeps `page_number` and `chunk_index`.

**`app/ingestion/embedder.py`**
- Load `all-MiniLM-L6-v2` once at module level (loading per request is a
  classic performance bug).
- `embed(texts: list[str]) -> np.ndarray` — batch, don't loop.

**`app/ingestion/indexer.py`**
- Create the Qdrant collection: 384 dims, cosine distance.
- Upsert vectors with payload `{chunk_id, doc_id, filename, page_number, text}`.
- Build the BM25 index over all chunk texts, persist it to disk with pickle.
- Insert rows into Postgres `documents` and `chunks`.

**`app/ingestion/router.py`** — `POST /v1/documents/upload`, multipart file,
returns `{doc_id, filename, chunk_count}`.

**Test:** upload 3 PDFs, check chunk count in Postgres, view the vectors in
the Qdrant dashboard at `localhost:6333/dashboard`.

### Phase 2 — Retrieval (3–4 days)

**`bm25_search.py`** → `search(query, k=50) -> list[(chunk_id, rank)]`
**`vector_search.py`** → embed the query, Qdrant search, `k=50`, same return shape
**`fusion.py`** → RRF:

```python
def rrf(rank_lists: list[list[str]], k: int = 60, top_n: int = 30) -> list[str]:
    """Merge ranked ID lists by rank, not by score."""
    scores: dict[str, float] = {}
    for ranked in rank_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top_n]
```

That is the whole algorithm. Being able to write it on a whiteboard in
under a minute is worth more than any framework you can name.

**`reranker.py`** → cross-encoder scores the 30 `(query, chunk)` pairs,
returns top 5 with scores.
**`pipeline.py`** → runs stages 1–4, returns top-5 chunks with scores and
a per-stage latency dict.

**Test:** `POST /v1/retrieve?q=...` on a question you know the answer to.
The right chunk should be in the top 5.

**Then build the ablation table** (`benchmarks/retrieval_ablation.py`):
run 50 questions through BM25-only, vector-only, hybrid, hybrid+RRF, and
hybrid+RRF+rerank. Record top-5 accuracy for each. This table is the single
most persuasive thing in your README — it shows you measured instead of
assumed.

### Phase 3 — Generation (3–4 days)

**`prompts.py`** — the system prompt is the product here:

```
You answer questions using ONLY the numbered context passages provided.

Rules:
1. Every factual sentence must end with a citation: [Source: filename, page N]
2. Use only the filenames and page numbers given in the context.
3. If the context does not contain the answer, reply exactly:
   "I don't have enough information in the provided documents to answer that."
4. Never use outside knowledge. Never guess a page number.
```

**`llm.py`** — one function, both modes, via LiteLLM:

```python
def generate(question: str, chunks: list[Chunk], mode: str) -> str:
    model = "anthropic/claude-sonnet-4-6" if mode == "cloud" else "ollama/mistral"
    context = "\n\n".join(
        f"[{i+1}] (Source: {c.filename}, page {c.page_number})\n{c.text}"
        for i, c in enumerate(chunks)
    )
    return litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        num_retries=2,
    ).choices[0].message.content
```

**`citation.py`** — split the answer into sentences; any sentence without a
`[Source: ...]` tag gets flagged. Any citation naming a file or page not in
the retrieved chunks is a **hallucinated citation** — remove it and log it.
This check is a real differentiator; most projects trust the prompt.

**`guards.py`** — Guardrails AI validators on the output: no PII leakage,
answer is non-empty, response matches the expected shape.

**`api/ask_router.py`** — `POST /v1/ask`: retrieve → generate → citation check
→ guards → write the audit row → return `{answer, sources, latency_ms, mode}`.

**Test:** 5 questions in each mode. Every citation must point at a chunk that
was actually retrieved. Ask one question the documents cannot answer — it must
refuse.

### Phase 4 — Offline mode + benchmark (2 days)

- Install Ollama, pull Mistral 7B, Llama 3.1 8B, Phi-3 Mini.
- Flip `MODE=offline`, re-run the same 5 questions.
- `benchmarks/model_benchmark.py`: same 20 questions across all three models,
  record tokens/sec, latency, and quality (RAGAS faithfulness or a manual 1–5).
- Optional but strong: serve one model with vLLM instead of Ollama and measure
  throughput at 10 concurrent requests. Ollama is for a laptop; vLLM is for
  production. Knowing the difference is the point.
- Write `benchmarks/results.md`, pick a default, **document why**.

### Phase 5 — Fine-tuned extraction (3–5 days)

Pick one narrow task: e.g. extract `{borrower_name, loan_amount,
interest_rate, maturity_date}` from loan agreements.

1. Hand-label 100 examples as JSONL (`{"input": raw_text, "output": json}`).
   Split 80/20.
2. `train_lora.py` in a free Colab T4: load the base model in 4-bit (QLoRA),
   LoRA rank 16 / alpha 32, 3 epochs. Save adapters (~50–100 MB).
3. `benchmark.py`: exact-match F1 per field, base vs tuned, on the 20 held-out
   examples. **Report your real numbers, whatever they are.**
4. `POST /v1/extract`: document in, validated Pydantic JSON out.

**Interview line:** "QLoRA loads a 7B model in 4-bit — about 4 GB instead of
42 — and trains only adapter layers, under 1% of parameters. That is why a
free T4 is enough."

### Phase 6 — Observability + eval gate (3 days)

- Add Langfuse to `docker-compose.yml`. Wrap `/v1/ask` in a trace with child
  spans for retrieval, rerank, LLM and citation check. Emit OpenTelemetry
  spans alongside so the tracing is vendor-neutral.
- Track per-request: latency per stage, token counts, estimated cost.
- Compute p50/p95 from the traces.
- `evaluation/test_set.json`: 50 question/ground-truth/context triples.
- `run_ragas.py`: score faithfulness, answer relevancy, context precision,
  context recall.
- `thresholds.json`: e.g. faithfulness ≥ 0.80, relevancy ≥ 0.75.
- `ci.yml`: ruff → mypy → pytest → docker compose up → RAGAS → fail if below
  threshold.

**Test it honestly:** deliberately break retrieval so it returns random
chunks. Push. CI must fail. Screenshot that failure for `EVALUATION.md`.
A blocked build is the proof; a passing build is just a claim.

### Phase 7 — Ship (2–3 days)

- README from the template, with the ablation table, RAGAS scores, benchmark
  table and fine-tuning numbers.
- Architecture diagram as an image; demo GIF at the top.
- `ARCHITECTURE.md`, `FAILURE_MODES.md`, `EVALUATION.md`.
- 2–3 minute demo video: upload → ask → cited answer → switch to offline →
  same question → Langfuse trace → RAGAS scores → CI gate.

---

## 9. Sample Documents (free and legal)

**Banking:** RBI notifications (rbi.org.in), SEBI circulars (sebi.gov.in),
SEC EDGAR filings (sec.gov/edgar), BIS Basel documents (bis.org/bcbs).
**Healthcare:** FDA drug labels (accessdata.fda.gov), WHO guidelines
(who.int/publications), CMS manuals (cms.gov), PubMed Central open access.

Take 15–20 documents, 5–50 pages each, mixed across both domains. Name the
files descriptively — **the filename appears in every citation**. Log the
source of each in `data/sources.md`.

---

## 10. Known Failure Modes (write these up honestly)

| Failure | Trigger | Detection | Degradation |
|---|---|---|---|
| No text layer | Scanned PDF | Parser returns empty text | Reject upload with a clear message; note OCR as future work |
| Nothing relevant retrieved | Question outside the corpus | Top rerank score below threshold | Return the refusal string, never guess |
| Hallucinated citation | Model invents a page number | Citation not in retrieved chunk set | Strip the sentence, log the incident |
| Table-heavy PDFs | Financial statements | Manual spot check | Documented limitation — table parsing is future work |
| Offline quality drop | `MODE=offline` on hard questions | RAGAS run in offline mode | Documented trade-off, measured not guessed |
| Prompt injection in a document | Malicious uploaded PDF | Promptfoo-style test cases | Guardrails validator + system prompt hardening |

---

## 11. Success Criteria

- [ ] `docker compose up -d` brings up Qdrant, Postgres and Langfuse healthy
- [ ] PDF and DOCX upload → chunks visible in Qdrant with page numbers
- [ ] `POST /v1/retrieve` returns sensible top-5 chunks
- [ ] `POST /v1/ask` returns cited answers in **both** modes
- [ ] An unanswerable question triggers the refusal, not a hallucination
- [ ] Ablation table measured and committed
- [ ] Local model benchmark measured and committed
- [ ] QLoRA before/after F1 measured and committed
- [ ] Langfuse traces visible with per-stage latency
- [ ] CI fails on a deliberately broken retrieval, passes when fixed
- [ ] README, ARCHITECTURE, FAILURE_MODES, EVALUATION all written
- [ ] Repo public, pinned, demo GIF at the top

---

## 12. Interview Prep

**Why did you build this?**
"Compliance teams search thousands of internal documents daily. Keyword search
misses paraphrases; cloud tools are a compliance violation for that data. I
built a system that does accurate cited Q&A and can run entirely offline."

**What makes it production-grade, not a demo?**
"Three things most portfolio RAG projects skip: hybrid retrieval with
reranking instead of naive vector search; citation verification that catches
the model inventing page numbers; and a CI gate that blocks any deploy where
RAGAS faithfulness drops below threshold."

**Walk me through retrieval.**
Four stages, name the cost of each, and explain RRF from first principles.

**How do you know it works?**
"RAGAS on 50 questions, plus an ablation table showing what each retrieval
component contributes. Every request is traced in Langfuse with a latency
breakdown. The LLM call is ~60% of end-to-end latency, so caching — not
retrieval optimisation — is where the wins are."

**What would you do differently at 10x scale?**
"Semantic caching for repeated queries, a clustered Qdrant deployment,
prompt caching for the system prompt, and swapping Ollama for vLLM to handle
concurrency. I benchmarked the last one."

**What is the weakest part?**
Have a real answer ready. Table-heavy PDFs and scanned documents are honest
choices — knowing your limits reads as senior, not weak.
