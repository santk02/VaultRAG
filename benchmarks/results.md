# Model Benchmark Results

## Test Configuration
- **Date**: 2026-08-28
- **Test Questions**: 6 compliance-related questions
- **Retrieval**: Hybrid BM25 + Vector search with RRF fusion
- **Chunk Size**: 512 tokens, 64 overlap
- **Reranking**: Cross-encoder top-5

## Models Tested

### Primary Model: Llama 3.1 8B Instruct
- **Ollama ID**: `llama3.1:8b`
- **Context Window**: 128K tokens
- **VRAM**: ~6.5GB (INT4)
- **Strengths**: Best instruction following, coding capability, lower VRAM

### Alternative Model: Gemma 2 9B
- **Ollama ID**: `gemma2:9b`
- **Context Window**: 8K tokens
- **VRAM**: ~7GB (INT4)
- **Strengths**: Better context utilization for RAG workloads

## Benchmark Results

*Note: Results will be populated after running the benchmark script with actual Ollama models.*

### Performance Comparison

| Model | Success Rate | Avg Latency | Min Latency | Max Latency | Queries/sec |
|-------|-------------|-------------|-------------|-------------|-------------|
| Llama 3.1 8B | - | - | - | - | - |
| Gemma 2 9B | - | - | - | - | - |

### Per-Question Results

#### Llama 3.1 8B
*Results pending benchmark execution*

#### Gemma 2 9B
*Results pending benchmark execution*

## Analysis

### Expected Findings (Based on Research)
- **Llama 3.1 8B**: Expected to have lower latency due to smaller size and better optimization
- **Gemma 2 9B**: Expected to have better context utilization (88.3% vs 86.2%) which may improve answer quality
- **Quality Trade-off**: Llama 3.1 8B shows better benchmarks on HumanEval (72.6 vs 40.2)

### Recommendation
*To be determined after actual benchmark execution*

## How to Run Benchmarks

```bash
# Ensure Ollama is running with both models
ollama pull llama3.1:8b
ollama pull gemma2:9b

# Run benchmark script
python benchmarks/model_benchmark.py

# Results will be saved to benchmarks/results.json
```

## Requirements
- Ollama running with target models
- VaultRAG services (Qdrant, PostgreSQL) running
- Sample documents indexed in the system
