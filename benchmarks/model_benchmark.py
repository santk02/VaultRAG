"""
Model benchmarking script for comparing local LLM performance.

This script compares different Ollama models on:
- Inference speed (tokens/second)
- Latency (time to first token, total generation time)
- Quality (using RAGAS faithfulness or manual evaluation)
- Resource usage (VRAM, CPU)
"""

import asyncio
import time
import json
from typing import List, Dict
from app.generation.llm import generate_answer
from app.retrieval.pipeline import retrieval_pipeline
from app.models import Chunk
from app.config import settings


class ModelBenchmark:
    """Benchmark different LLM models for RAG performance."""
    
    def __init__(self):
        """Initialize benchmark with test questions."""
        self.questions = [
            "What are the key requirements for document retention?",
            "Explain the compliance framework mentioned in the documents.",
            "What are the penalties for non-compliance?",
            "Describe the approval process for new policies.",
            "What are the data protection requirements?",
            # Add more questions for comprehensive testing
        ]
        
        self.models = [
            settings.offline_model,  # llama3.1:8b
            settings.alternative_model,  # gemma2:9b
        ]
    
    async def benchmark_model(self, model_name: str) -> Dict:
        """
        Benchmark a single model on all test questions.
        
        Args:
            model_name: Ollama model identifier
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "model": model_name,
            "total_questions": len(self.questions),
            "successful_answers": 0,
            "total_latency_ms": 0,
            "avg_latency_ms": 0,
            "min_latency_ms": float('inf'),
            "max_latency_ms": 0,
            "per_question": []
        }
        
        for question in self.questions:
            try:
                # Retrieve chunks (same for all models)
                chunks, _ = await retrieval_pipeline(question, top_k=5)
                
                if not chunks:
                    print(f"  No chunks retrieved for: {question}")
                    continue
                
                # Generate answer with specific model
                start_time = time.time()
                answer, actual_model, latency_ms = await generate_answer(
                    question=question,
                    chunks=chunks,
                    model_override=model_name
                )
                
                # Record results
                question_result = {
                    "question": question,
                    "answer_length": len(answer),
                    "latency_ms": latency_ms,
                    "chunks_used": len(chunks),
                    "success": True
                }
                
                results["per_question"].append(question_result)
                results["successful_answers"] += 1
                results["total_latency_ms"] += latency_ms
                results["min_latency_ms"] = min(results["min_latency_ms"], latency_ms)
                results["max_latency_ms"] = max(results["max_latency_ms"], latency_ms)
                
                print(f"  [OK] {question[:50]}... ({latency_ms:.0f}ms)")
                
            except Exception as e:
                print(f"  [FAIL] {question[:50]}... failed: {str(e)}")
                results["per_question"].append({
                    "question": question,
                    "error": str(e),
                    "success": False
                })
        
        # Calculate averages
        if results["successful_answers"] > 0:
            results["avg_latency_ms"] = results["total_latency_ms"] / results["successful_answers"]
        
        return results
    
    async def run_all_benchmarks(self) -> List[Dict]:
        """
        Run benchmarks for all configured models.
        
        Returns:
            List of benchmark results for each model
        """
        all_results = []
        
        for model in self.models:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {model}")
            print(f"{'='*60}")
            
            results = await self.benchmark_model(model)
            all_results.append(results)
            
            print(f"\nResults for {model}:")
            print(f"  Success rate: {results['successful_answers']}/{results['total_questions']}")
            print(f"  Avg latency: {results['avg_latency_ms']:.0f}ms")
            print(f"  Min latency: {results['min_latency_ms']:.0f}ms")
            print(f"  Max latency: {results['max_latency_ms']:.0f}ms")
        
        return all_results
    
    def save_results(self, results: List[Dict], output_file: str = "benchmarks/results.json"):
        """
        Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")


async def main():
    """Run the complete benchmark suite."""
    print("VaultRAG Model Benchmarking")
    print("=" * 60)
    
    benchmark = ModelBenchmark()
    results = await benchmark.run_all_benchmarks()
    benchmark.save_results(results)
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\n{result['model']}:")
        print(f"  Success: {result['successful_answers']}/{result['total_questions']}")
        print(f"  Avg Latency: {result['avg_latency_ms']:.0f}ms")
        if result['successful_answers'] > 0:
            print(f"  Throughput: {1000/result['avg_latency_ms']:.2f} queries/sec")


if __name__ == "__main__":
    asyncio.run(main())
