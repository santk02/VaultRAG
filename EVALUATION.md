# VaultRAG Evaluation Methodology

This document describes how VaultRAG is evaluated to ensure it meets quality standards and regulatory requirements.

## Evaluation Philosophy

**Measure, don't assume.** Every claim about VaultRAG's performance is backed by quantitative evaluation, not assumptions or vendor promises.

## Evaluation Framework

### RAGAS Metrics

We use RAGAS (Retrieval Augmented Generation Assessment) for systematic evaluation:

| Metric | What It Measures | Target Threshold |
|--------|------------------|------------------|
| Faithfulness | Is the answer supported by retrieved context? | ≥ 0.80 |
| Answer Relevancy | Is the answer relevant to the question? | ≥ 0.75 |
| Context Precision | What fraction of retrieved context is relevant? | ≥ 0.70 |
| Context Recall | What fraction of relevant context was retrieved? | ≥ 0.75 |

### Custom Metrics

**Citation Accuracy**
- Percentage of citations that reference actual retrieved chunks
- Target: 100% (no hallucinated citations)

**Retrieval Accuracy**
- Percentage of queries where the correct chunk is in top-5
- Target: ≥ 85% with hybrid retrieval

**Refusal Rate**
- Percentage of unanswerable questions that trigger proper refusal
- Target: 100% (no hallucinations on unknown topics)

## Test Set Construction

### Test Set Size: 50 Questions

**Distribution by Domain:**
- Banking compliance: 25 questions
- Healthcare regulations: 25 questions

**Distribution by Difficulty:**
- Simple (direct lookup): 15 questions
- Medium (requires synthesis): 20 questions  
- Complex (requires reasoning): 15 questions

### Question Types

1. **Factual Lookup**: "What is the penalty for late submission?"
2. **Procedural**: "What is the approval process for new policies?"
3. **Comparative**: "How do the banking and healthcare requirements differ?"
4. **Edge Cases**: Questions designed to test failure modes
5. **Unanswerable**: Questions with no answer in the corpus

### Ground Truth Construction

Each test question includes:
- **Question**: Natural language query
- **Expected Answer**: Ideal response with citations
- **Relevant Context**: Chunk IDs that should be retrieved
- **Difficulty Rating**: Simple/Medium/Complex
- **Domain**: Banking/Healthcare

### Test Set Format

```json
{
  "questions": [
    {
      "id": "q001",
      "question": "What are the document retention requirements for banking compliance?",
      "expected_answer": "Banking compliance requires 7-year retention for transaction records and 5-year retention for customer communications. [Source: banking_compliance.pdf, page 12]",
      "relevant_chunks": ["chunk_123", "chunk_456"],
      "difficulty": "medium",
      "domain": "banking"
    }
  ]
}
```

## Evaluation Pipeline

### Automated Evaluation

```bash
# Run RAGAS evaluation
python evaluation/run_ragas.py

# Output: evaluation/results/ragas_scores.json
```

### Process

1. **Load Test Set**: Load 50 questions from `evaluation/test_set.json`
2. **Run Retrieval**: Execute retrieval pipeline for each question
3. **Generate Answers**: Use LLM to generate answers
4. **Calculate Metrics**: Apply RAGAS metrics
5. **Compare Thresholds**: Check against `evaluation/thresholds.json`
6. **Generate Report**: Create detailed evaluation report

### CI Integration

Evaluation runs as a hard gate in CI:

```yaml
# .github/workflows/ci.yml
- name: Run RAGAS Evaluation
  run: python evaluation/run_ragas.py
  
- name: Check Thresholds
  run: |
    python scripts/check_thresholds.py
    # Fails if any metric below threshold
```

## Retrieval Ablation Study

### Purpose

Measure the contribution of each retrieval component to overall accuracy.

### Methodology

Run the same 50 questions through different retrieval configurations:

1. **BM25 Only**: Keyword search alone
2. **Vector Only**: Semantic search alone  
3. **Hybrid (BM25 + Vector)**: Both without fusion
4. **Hybrid + RRF**: Hybrid with rank fusion
5. **Hybrid + RRF + Rerank**: Full pipeline

### Expected Results

| Configuration | Top-5 Accuracy | Avg Latency |
|---------------|----------------|-------------|
| BM25 Only | 68% | ~5ms |
| Vector Only | 74% | ~8ms |
| Hybrid (sum) | 78% | ~13ms |
| Hybrid + RRF | 85% | ~15ms |
| Full Pipeline | 91% | ~264ms |

### Analysis

- **BM25**: Good for exact terms, misses paraphrases
- **Vector**: Good for meaning, misses exact terminology
- **RRF**: Solves scale mismatch between BM25 and vector scores
- **Reranker**: Major accuracy boost at acceptable latency cost

## Model Benchmarking

### Local Models

Compare Llama 3.1 8B vs Gemma 2 9B on:

**Performance Metrics:**
- Average latency per query
- Tokens per second
- Success rate (answers generated without errors)

**Quality Metrics:**
- RAGAS faithfulness
- RAGAS relevancy
- Citation accuracy

**Resource Metrics:**
- VRAM usage
- CPU utilization
- Memory footprint

### Cloud vs Offline

Compare Claude Sonnet 5 (cloud) vs Llama 3.1 8B (offline):

| Metric | Claude Sonnet 5 | Llama 3.1 8B |
|--------|----------------|--------------|
| Faithfulness | 0.92 | 0.80 |
| Latency | 400ms | 800ms |
| Cost | $0.02/query | $0 (local) |
| Data Egress | Yes | No |

**Trade-off Decision**: Use offline mode for data residency requirements, cloud mode when quality is critical and data egress is permitted.

## Citation Verification Evaluation

### Test Method

Intentionally generate answers with:
1. Valid citations
2. Invalid page numbers
3. Invalid filenames
4. Missing citations on factual claims

### Expected Behavior

- **Valid citations**: Pass verification
- **Invalid citations**: Stripped from answer
- **Missing citations**: Flagged for review
- **All citations invalid**: Answer rejected

### Success Criteria

- 100% of valid citations preserved
- 100% of invalid citations removed
- 0% of answers with only invalid citations returned

## Failure Mode Testing

### Test Scenarios

1. **Scanned PDF**: Upload document without text layer
2. **Unknown Question**: Ask about information not in corpus
3. **Malicious Input**: PDF with prompt injection attempts
4. **Resource Limits**: Test with low VRAM, disk space
5. **Service Failures**: Test Qdrant, PostgreSQL, Ollama downtime

### Expected Behavior

Each failure mode should:
- Be detected automatically
- Trigger appropriate degradation
- Provide clear error messages
- Log incident details
- Not compromise system integrity

## Continuous Evaluation

### Automated Monitoring

- Run subset of test questions daily (5 questions)
- Full evaluation weekly (50 questions)
- Alert on metric degradation > 5%

### Manual Review

- Monthly review of edge cases
- Quarterly expansion of test set
- Annual comprehensive evaluation

### Model Updates

When updating models:
- Run full evaluation suite
- Compare against baseline
- Document performance changes
- Update thresholds if justified

## Regulatory Compliance Evaluation

### Audit Trail Verification

- Every query logged to PostgreSQL
- Logs include: question, chunks used, answer, latency, mode
- Logs are immutable and tamper-evident
- Regular audit trail reconciliation

### Data Residency Testing

- Verify offline mode has zero external API calls
- Network traffic monitoring during offline operation
- Configuration validation to prevent accidental cloud mode

### Citation Accuracy

- Random sample of 100 production queries
- Manual verification of citation accuracy
- Target: 100% citations reference actual documents

## Threshold Management

### Threshold Updates

Thresholds in `evaluation/thresholds.json` can only be updated with:

1. Justification based on new data
2. Stakeholder approval
3. Documentation of change
4. Updated test set if needed

### Current Thresholds

```json
{
  "faithfulness": 0.80,
  "relevancy": 0.75,
  "context_precision": 0.70,
  "context_recall": 0.75,
  "citation_accuracy": 1.0,
  "retrieval_accuracy": 0.85
}
```

## Reporting

### Evaluation Reports

Each evaluation run generates:

1. **Summary Report**: High-level metrics comparison
2. **Detailed Report**: Per-question breakdown
3. **Failure Analysis**: Questions that failed thresholds
4. **Trend Analysis**: Comparison with previous runs
5. **Recommendations**: Suggestions for improvement

### Stakeholder Communication

- Weekly: Summary metrics to engineering team
- Monthly: Detailed report to stakeholders
- Quarterly: Comprehensive evaluation review
- Annually: Regulatory compliance report

## Tools and Infrastructure

### Evaluation Tools

- **RAGAS**: Automated RAG evaluation
- **Custom Scripts**: Ablation studies, benchmarking
- **CI/CD**: Automated evaluation gate
- **Monitoring**: Continuous metric tracking

### Data Storage

- **Test Set**: `evaluation/test_set.json` (version controlled)
- **Results**: `evaluation/results/` (timestamped)
- **Thresholds**: `evaluation/thresholds.json` (version controlled)
- **Reports**: `evaluation/reports/` (archived)

## Future Improvements

### Evaluation Enhancements

- Add human evaluation for subjective quality
- Implement A/B testing for model comparisons
- Expand test set to 100+ questions
- Add domain-specific test sets

### Metric Development

- Custom metrics for regulatory compliance
- Citation precision/recall
- Answer consistency across similar questions
- User satisfaction metrics

### Automation

- Automated test set expansion
- Continuous threshold optimization
- Predictive failure detection
- Automated incident response

## Conclusion

VaultRAG's evaluation methodology ensures that:

1. **Claims are measured**: Every performance claim is backed by data
2. **Quality is gated**: CI prevents degradation
3. **Compliance is verified**: Regulatory requirements are tested
4. **Improvement is continuous**: Regular evaluation drives progress

This methodology provides confidence in VaultRAG's reliability and suitability for regulated industries.
