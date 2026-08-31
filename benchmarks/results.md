# Model Benchmark Results

## Test Configuration
- **Date**: 2026-08-28
- **Test Questions**: 6 compliance-related questions
- **Retrieval**: Hybrid BM25 + Vector search with RRF fusion
- **Chunk Size**: 512 tokens, 64 overlap
- **Reranking**: Cross-encoder top-5

## Models Tested

`benchmarks/model_benchmark.py` compares `settings.offline_model` against
`settings.alternative_model` (see `app/config.py` / `.env.example`) — currently:

### Primary Model: Mistral 7B (`offline_model` default)
- **Ollama ID**: `mistral`
- **VRAM**: ~4-5GB (INT4)
- **Strengths**: Solid instruction following at a size that runs comfortably on a laptop CPU/small GPU — the blueprint's stated dev default

### Alternative Model: Llama 3.1 8B (`alternative_model` default)
- **Ollama ID**: `llama3.1:8b`
- **Context Window**: 128K tokens
- **VRAM**: ~6.5GB (INT4)
- **Strengths**: Better instruction following/benchmarks at the cost of more VRAM and latency

## Benchmark Results

*Note: Results will be populated after running the benchmark script with actual Ollama models
and a live VaultRAG stack (Qdrant + Postgres + indexed documents). Not run in this audit —
no live services in this sandbox. Numbers below are placeholders pending a real run.*

### Performance Comparison

| Model | Success Rate | Avg Latency | Min Latency | Max Latency | Queries/sec |
|-------|-------------|-------------|-------------|-------------|-------------|
| Mistral 7B | - | - | - | - | - |
| Llama 3.1 8B | - | - | - | - | - |

### Per-Question Results

#### Mistral 7B
*Results pending benchmark execution*

#### Llama 3.1 8B
*Results pending benchmark execution*

## Analysis

### Recommendation
*To be determined after actual benchmark execution — run `python benchmarks/model_benchmark.py`
against a live stack and replace the placeholders above with real numbers.*

## How to Run Benchmarks

```bash
# Ensure Ollama is running with both models
ollama pull mistral
ollama pull llama3.1:8b

# Run benchmark script
python benchmarks/model_benchmark.py

# Results will be saved to benchmarks/results.json
```

## Requirements
- Ollama running with target models
- VaultRAG services (Qdrant, PostgreSQL) running
- Sample documents indexed in the system
