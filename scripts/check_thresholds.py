#!/usr/bin/env python3
"""
Check evaluation results against quality thresholds.
This script is used in CI to ensure quality standards are met.
"""

import json
import sys
from pathlib import Path


def load_thresholds():
    """Load quality thresholds from configuration."""
    thresholds_file = Path("evaluation/thresholds.json")
    if not thresholds_file.exists():
        print("Warning: thresholds.json not found, using defaults")
        return {
            "faithfulness": 0.80,
            "relevancy": 0.75,
            "context_precision": 0.70,
            "context_recall": 0.75,
            "citation_accuracy": 1.0,
            "retrieval_accuracy": 0.85
        }
    
    with open(thresholds_file) as f:
        return json.load(f)


def load_latest_results():
    """Load the latest evaluation results."""
    results_dir = Path("evaluation/results")
    if not results_dir.exists():
        print("Error: No evaluation results found")
        sys.exit(1)
    
    # Find the most recent results file
    results_files = sorted(results_dir.glob("ragas_*.json"), reverse=True)
    if not results_files:
        print("Error: No RAGAS results files found")
        sys.exit(1)
    
    latest_file = results_files[0]
    print(f"Using results from: {latest_file}")
    
    with open(latest_file) as f:
        return json.load(f)


def check_thresholds(results, thresholds):
    """Check if results meet all thresholds."""
    failures = []
    
    # Check each metric
    for metric, threshold in thresholds.items():
        if metric in results:
            actual = results[metric]
            if actual < threshold:
                failures.append({
                    "metric": metric,
                    "threshold": threshold,
                    "actual": actual,
                    "diff": threshold - actual
                })
            else:
                print(f"✓ {metric}: {actual:.3f} ≥ {threshold:.3f}")
        else:
            print(f"⚠ {metric}: Not found in results")
    
    return failures


def main():
    """Main function to run threshold checks."""
    print("Checking evaluation results against quality thresholds...")
    print("=" * 60)
    
    # Load thresholds and results
    thresholds = load_thresholds()
    results = load_latest_results()
    
    # Check thresholds
    failures = check_thresholds(results, thresholds)
    
    # Report results
    print("=" * 60)
    
    if failures:
        print("❌ Threshold check FAILED")
        print("\nFailed metrics:")
        for failure in failures:
            print(f"  - {failure['metric']}: {failure['actual']:.3f} < {failure['threshold']:.3f} (diff: {failure['diff']:.3f})")
        sys.exit(1)
    else:
        print("✅ All thresholds passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
