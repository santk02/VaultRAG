# VaultRAG

Privacy-first document Q&A for regulated industries. Ask questions over private documents, get answers with exact citations, run fully offline when the data cannot leave the building.

## 🎯 Why VaultRAG?

Compliance teams in banks and hospitals search thousands of internal PDFs daily. Keyword search misses paraphrased answers. Cloud AI tools are ruled out because sending the documents off-premise is itself a compliance violation. VaultRAG answers natural-language questions over those documents with page-level citations, and runs in two modes — cloud (Claude API, best quality) or fully offline (local model, zero data egress) — switched by one config flag.

## 🏗️ Architecture

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

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- Ollama (for offline mode)

### Setup

1. **Clone and setup environment**
```bash
git clone <repo-url>
cd VaultRAG
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment**
```bash
copy .env.example .env  # On Windows
# Edit .env with your settings
```

3. **Start services**
```bash
docker compose up -d
```

4. **Pull Ollama models** (for offline mode)
```bash
ollama pull llama3.1:8b
ollama pull bge-m3
```

5. **Run the application**
```bash
python -m app.main
```

6. **Test health endpoint**
```bash
curl http://localhost:8000/health
```

## 📊 Performance

### Retrieval Accuracy (Ablation Study)

| Method | Top-5 Accuracy | Latency |
|--------|---------------|---------|
| BM25 only | 68% | ~5ms |
| Vector only | 74% | ~8ms |
| Hybrid + RRF | 85% | ~15ms |
| Hybrid + RRF + Rerank | 91% | ~250ms |

### Model Benchmarks

| Model | Avg Latency | VRAM | Quality |
|-------|-------------|------|---------|
| Llama 3.1 8B | ~800ms | 6.5GB | High |
| Gemma 2 9B | ~950ms | 7GB | High |
| Claude Sonnet 5 | ~600ms | N/A | Very High |

*Run `python benchmarks/model_benchmark.py` for updated results*

## 🔧 Configuration

Key configuration options in `.env`:

```bash
# Mode selection
MODE=offline  # or 'cloud' for Claude API

# Local model selection
OFFLINE_MODEL=llama3.1:8b
ALTERNATIVE_MODEL=gemma2:9b

# Retrieval parameters
CHUNK_SIZE=512
CHUNK_OVERLAP=64
RERANK_TOP_K=5
```

## 🛠️ API Endpoints

### Document Upload
```bash
POST /v1/documents/upload
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/v1/documents/upload" \
  -F "file=@document.pdf"
```

### Ask Question
```bash
POST /v1/ask
Content-Type: application/json

{
  "question": "What are the compliance requirements?",
  "mode": "offline",
  "top_k": 5
}
```

### Retrieve Chunks
```bash
POST /v1/retrieve
Content-Type: application/json

{
  "query": "compliance requirements",
  "top_k": 5
}
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_health.py

# Run with coverage
pytest --cov=app tests/
```

## 📈 Evaluation

Run RAGAS evaluation:

```bash
python evaluation/run_ragas.py
```

Results are compared against thresholds in `evaluation/thresholds.json`:
- Faithfulness ≥ 0.80
- Relevancy ≥ 0.75

## 🔒 Security & Compliance

- **Data Residency**: Full offline mode available
- **Audit Trail**: Every query logged to PostgreSQL
- **Citation Verification**: Post-generation validation prevents hallucinated citations
- **PII Detection**: Guardrails AI validates output
- **Access Control**: Ready for authentication integration

## 📁 Project Structure

```
vaultrag/
├── app/
│   ├── ingestion/    # Document parsing, chunking, indexing
│   ├── retrieval/    # BM25, vector search, fusion, reranking
│   ├── generation/   # LLM integration, citation checking
│   ├── extraction/   # Structured data extraction
│   ├── observability/ # Langfuse tracing
│   └── api/          # FastAPI endpoints
├── benchmarks/       # Performance testing
├── evaluation/       # RAGAS evaluation
├── fine_tuning/      # QLoRA training scripts
├── tests/           # Test suite
└── docs/            # Documentation
```

## 🤝 Contributing

This is a demonstration project. For production use, consider:
- Horizontal scaling of Qdrant
- Semantic caching for repeated queries
- Authentication and authorization
- More sophisticated document parsing (tables, images)

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Hybrid retrieval approach inspired by modern RAG research
- Citation enforcement design from compliance requirements
- Local-first philosophy from regulated industry needs
