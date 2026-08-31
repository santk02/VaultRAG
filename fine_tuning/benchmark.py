#!/usr/bin/env python3
"""
Before/after benchmark for the QLoRA-tuned loan-extraction adapter (blueprint Phase 5).

Runs the 20 held-out examples (fine_tuning/dataset/val.jsonl) through the base model
and the LoRA-adapted model, computes exact-match F1 per field, and reports both so
the fine-tuning's actual contribution is measured, not assumed.

"Exact-match F1 per field" here means: for each of the 4 schema fields, treat the
task as binary correct/incorrect per example (predicted value string-equals
ground-truth value after normalization), and compute precision/recall/F1 as if
"predicted this field non-null and correct" were a positive prediction.

Requires: fine_tuning/requirements.txt, plus an adapter directory produced by
train_lora.py.

Usage:
    python fine_tuning/benchmark.py \
        --base_model mistralai/Mistral-7B-Instruct-v0.2 \
        --adapter_dir fine_tuning/adapters/loan_extraction \
        --val_file fine_tuning/dataset/val.jsonl
"""

import argparse
import json
from typing import Dict, Optional

# Reuse the real Pydantic schema so "correct" is judged against the same validation
# the production /v1/extract endpoint applies.
from app.extraction.schemas import LoanAgreementExtraction

EXTRACTION_INSTRUCTION = (
    "Extract the following fields as JSON: borrower_name, loan_amount, "
    "interest_rate, maturity_date. Return JSON only, no explanation.\n\n"
    "Document:\n{document}\n\nJSON:"
)

FIELDS = ["borrower_name", "loan_amount", "interest_rate", "maturity_date"]


def normalize(value) -> Optional[str]:
    """Loose string normalization so '1,000,000' vs '1000000' etc. don't count as mismatches."""
    if value is None:
        return None
    return str(value).strip().lower().replace(",", "").replace("$", "").replace("%", "")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Before/after F1 benchmark for the extraction adapter"
    )
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--adapter_dir", default="fine_tuning/adapters/loan_extraction")
    parser.add_argument("--val_file", default="fine_tuning/dataset/val.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def load_jsonl(path: str) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run_model(model, tokenizer, prompt: str, max_new_tokens: int) -> Optional[dict]:
    """Generate one completion and parse it against the extraction schema; returns None on any failure
    (bad JSON, schema validation failure) — that counts as "field not extracted", not a crash.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    completion = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    try:
        payload = json.loads(completion.strip())
        validated = LoanAgreementExtraction.model_validate(payload)
        return validated.model_dump(mode="json")
    except Exception:
        return None


def score_predictions(
    predictions: list, ground_truths: list
) -> Dict[str, Dict[str, float]]:
    """Per-field precision/recall/F1 across all examples."""
    scores = {}
    for field in FIELDS:
        tp = fp = fn = 0
        for pred, truth in zip(predictions, ground_truths):
            pred_val = normalize((pred or {}).get(field))
            truth_val = normalize(truth.get(field))

            if truth_val is None:
                continue  # field not present in ground truth for this example — excluded from scoring

            if pred_val == truth_val:
                tp += 1
            elif pred_val is None:
                fn += 1  # model failed to extract a field that was present
            else:
                fp += 1  # model extracted a field but got the value wrong

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        scores[field] = {"precision": precision, "recall": recall, "f1": f1}
    return scores


def print_report(title: str, scores: Dict[str, Dict[str, float]]):
    print(f"\n{title}")
    print("-" * 60)
    for field, m in scores.items():
        print(
            f"  {field:18s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}"
        )
    avg_f1 = sum(m["f1"] for m in scores.values()) / len(scores)
    print(f"  {'AVERAGE F1':18s} {avg_f1:.2f}")


def main():
    args = parse_args()

    # Deferred imports — same rationale as train_lora.py (inspectable without the GPU deps installed)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    val_examples = load_jsonl(args.val_file)
    ground_truths = [json.loads(ex["output"]) for ex in val_examples]
    prompts = [
        EXTRACTION_INSTRUCTION.format(document=ex["input"]) for ex in val_examples
    ]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading base model {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto"
    )

    # --- Base model (before fine-tuning) ---
    print("Running base model on held-out set...")
    base_predictions = [
        run_model(base_model, tokenizer, p, args.max_new_tokens) for p in prompts
    ]
    base_scores = score_predictions(base_predictions, ground_truths)

    # --- Adapted model (after fine-tuning) ---
    print(f"Loading LoRA adapter from {args.adapter_dir}...")
    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    print("Running tuned model on held-out set...")
    tuned_predictions = [
        run_model(tuned_model, tokenizer, p, args.max_new_tokens) for p in prompts
    ]
    tuned_scores = score_predictions(tuned_predictions, ground_truths)

    print_report("BASE MODEL (before fine-tuning)", base_scores)
    print_report("TUNED MODEL (after fine-tuning)", tuned_scores)

    # Report real numbers, whatever they are — per blueprint Phase 5, no cherry-picking
    print("\nDelta (tuned - base) per field F1:")
    for field in FIELDS:
        delta = tuned_scores[field]["f1"] - base_scores[field]["f1"]
        print(f"  {field:18s} {delta:+.2f}")


if __name__ == "__main__":
    main()
