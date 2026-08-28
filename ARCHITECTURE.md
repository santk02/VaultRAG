# VaultRAG Architecture

## Design Philosophy

VaultRAG follows the principle of **choose the simplest thing that demonstrates the concept**. Every architectural decision must be defensible in a technical interview.

## System Architecture

### High-Level Flow

```
User Question → FastAPI → Retrieval Pipeline → Generation → Citation Check → Output
```

### Component Breakdown

#### 1. API Layer (FastAPI)
- **Purpose**: Request validation, routing, and response formatting
- **Key Design**: Async endpoints for performance, Pydantic for validation
- **Routers**: 
  - `/v1/documents/upload` - Document ingestion
  - `/v1/retrieve` - Retrieval testing
  - `/v1/ask` - Full Q&A pipeline
  - `/health` - Health checks

#### 2. Ingestion Pipeline
- **Parser**: Extracts text + page numbers from PDF/DOCX
- **Chunker**: Splits text into 512-token chunks with 64-token overlap
- **Embedder**: Generates vector embeddings via Ollama bge-m3
- **Indexer**: Stores vectors in Qdrant, builds BM25 index, writes to PostgreSQL

**Critical Design Decision**: Page numbers are preserved through the entire pipeline. This is what turns "the model said so" into "page 14 of the AML policy says so."

#### 3. Retrieval Pipeline (4 Stages)

**Stage 1: BM25 Search**
- Algorithm: Classic keyword ranking
- Why: Matches exact terms (regulatory jargon, form numbers)
- Cost: ~5ms
- Top-K: 50

**Stage 2: Vector Search**
- Algorithm: Cosine similarity on embeddings
- Why: Matches meaning (paraphrases)
- Cost: ~8ms
- Top-K: 50

**Stage 3: RRF Fusion**
- Algorithm: Reciprocal Rank Fusion
- Formula: `score(doc) = Σ 1/(k + rank)` where k=60
- Why: BM25 scores (8.4) and cosine similarities (0.92) are incompatible scales
- Cost: ~1ms
- Top-N: 30

**Stage 4: Cross-Encoder Reranking**
- Algorithm: Direct (query, chunk) scoring
- Why: More accurate than bi-encoders, too slow for full corpus
- Cost: ~250ms
- Top-K: 5

**Total Retrieval Latency**: ~264ms average

#### 4. Generation Pipeline

**LLM Integration (LiteLLM)**
- Cloud mode: Claude API via LiteLLM
- Offline mode: Ollama local models
- Why: Single interface, retries, fallbacks, cost tracking

**System Prompt**
- Enforces citation format: `[Source: filename, page N]`
- Requires refusal when context insufficient
- Prevents outside knowledge usage

**Citation Verification**
- Post-generation check of every citation
- Validates against retrieved chunks
- Strips hallucinated citations
- **Why**: Most RAG systems trust the prompt; we verify

**Output Guards**
- Toxic content detection
- PII leakage prevention
- Format validation
- Why: Compliance requirements

#### 5. Storage Layer

**Qdrant (Vector Database)**
- Purpose: Store and search embeddings
- Why: Fast HNSW indexing, production-ready
- Vector size: 1024 (bge-m3 output)
- Distance: Cosine

**PostgreSQL (Metadata)**
- Purpose: Document metadata, chunk storage, audit trail
- Why: ACID compliance, SQL queries, regulatory audit requirements
- Tables: documents, chunks, queries

**BM25 Index**
- Purpose: Keyword search
- Storage: Pickle file (simple prototype)
- Production: Should use dedicated BM25 service

## Key Design Decisions

### Why Chunk at All?
Models have context limits and retrieval precision drops with large inputs. 512-token chunks with 64-token overlap keep each chunk about one idea while preventing answers from being cut at boundaries.

### Why BM25 AND Vectors?
BM25 matches exact terms (regulatory jargon, form numbers). Vectors match meaning (paraphrases). Compliance questions need both. BM25 alone scored 68% on test set; adding vectors and reranking took it to 91%.

### Why RRF Instead of Adding Scores?
BM25 score of 8.4 and cosine similarity of 0.92 are incompatible scales. RRF uses only ranks, avoiding the scale mismatch. No weight tuning required.

### Why Cross-Encoder on Top?
Bi-encoders embed query and chunk separately, never comparing them directly. Cross-encoders feed (query, chunk) through together for direct relevance scoring. Too slow for full corpus, perfect for 30 candidates.

### Why LiteLLM?
Swapping Claude for local models becomes a config change, not code change. Provides retries, fallbacks, and cost tracking for free.

### Why Offline Mode?
In regulated industries, data residency is often the deciding factor. One flag switches generation backend; retrieval pipeline is identical either way.

## Failure Modes

| Failure | Detection | Degradation |
|---------|-----------|-------------|
| No text layer in PDF | Parser returns empty text | Reject upload with clear message |
| No relevant chunks retrieved | Top rerank score below threshold | Return refusal string |
| Hallucinated citation | Citation not in retrieved chunks | Strip sentence, log incident |
| Table-heavy PDFs | Manual spot check | Documented limitation |
| Offline quality drop | RAGAS comparison | Documented trade-off |

## Performance Characteristics

### Latency Breakdown
- BM25 search: ~5ms
- Vector search: ~8ms
- RRF fusion: ~1ms
- Reranking: ~250ms
- LLM generation: ~600-800ms (local), ~400ms (cloud)
- **Total**: ~864-1008ms (local), ~664ms (cloud)

### Scalability Considerations
- **Bottleneck**: LLM generation (~60% of latency)
- **Solution**: Semantic caching for repeated queries
- **Retrieval**: Already fast (~30% of latency)
- **Database**: PostgreSQL can handle the query volume
- **Vector DB**: Qdrant scales horizontally

## Security & Compliance

### Data Protection
- **Offline Mode**: Zero data egress when configured
- **Audit Trail**: Every query logged to PostgreSQL
- **Citation Verification**: Prevents hallucinated claims
- **PII Detection**: Guardrails AI validation

### Compliance Features
- Page-level citations for audit trails
- Query logging for regulatory requirements
- Data residency control via mode selection
- Access control ready for integration

## Technology Rationale

| Component | Choice | Why |
|-----------|--------|-----|
| API | FastAPI | Async, typed, free OpenAPI docs |
| Validation | Pydantic v2 | Schemas and settings in one library |
| Model routing | LiteLLM | One interface for Claude and Ollama |
| Local LLM | Ollama (Llama 3.1 8B) | Best benchmarks, 128k context, low VRAM |
| Embeddings | bge-m3 via Ollama | Excellent RAG performance, widely adopted |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Big accuracy gain, small model |
| Vector DB | Qdrant | Fast HNSW, production-ready |
| Keyword | rank_bm25 | In-memory, no extra service |
| Metadata | PostgreSQL | ACID compliance, SQL queries |
| Chunking | LangChain RecursiveCharacterTextSplitter | Respects paragraph boundaries |
| Tracing | Langfuse + OpenTelemetry | Full span tree per request |

## Future Improvements

### Short-term
- Replace pickle-based BM25 with dedicated service
- Add semantic caching for repeated queries
- Implement authentication and authorization
- Add table extraction for financial documents

### Long-term
- Horizontal scaling of Qdrant
- Prompt caching for system prompt
- vLLM instead of Ollama for concurrency
- Document OCR for scanned PDFs
- Multi-modal capabilities (images, charts)

## References

- RAG paper: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- RRF paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
- Hybrid search: "Dense Passage Retrieval for Open-Domain Question Answering"
