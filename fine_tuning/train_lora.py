#!/usr/bin/env python3
"""
QLoRA fine-tuning for the loan-agreement extraction task (blueprint Phase 5).

Trains a LoRA adapter on top of a 4-bit-quantized base model to extract
{borrower_name, loan_amount, interest_rate, maturity_date} from raw loan
agreement text — the schema in app/extraction/schemas.py:LoanAgreementExtraction.

Designed to run on a free Colab T4 (16GB VRAM): 4-bit quantization (QLoRA)
keeps a 7B base model to ~4GB instead of ~28GB in fp16, and only the LoRA
adapter layers (well under 1% of total parameters) are trained.

Requires: fine_tuning/requirements.txt (torch, transformers, peft, bitsandbytes,
accelerate, datasets) — NOT installed by the main requirements.txt.

Dataset format (JSONL, one example per line):
    {"input": "<raw loan agreement text>", "output": "{\"borrower_name\": ..., ...}"}

Usage (in Colab or a local CUDA machine):
    pip install -r fine_tuning/requirements.txt
    python fine_tuning/train_lora.py \
        --base_model mistralai/Mistral-7B-Instruct-v0.2 \
        --train_file fine_tuning/dataset/train.jsonl \
        --val_file fine_tuning/dataset/val.jsonl \
        --output_dir fine_tuning/adapters/loan_extraction
"""

import argparse
import json
import os

EXTRACTION_INSTRUCTION = (
    "Extract the following fields as JSON: borrower_name, loan_amount, "
    "interest_rate, maturity_date. Return JSON only, no explanation.\n\n"
    "Document:\n{document}\n\nJSON:"
)


def build_prompt(example: dict) -> str:
    """Format one training example as an instruction-following prompt+completion pair."""
    return (
        EXTRACTION_INSTRUCTION.format(document=example["input"])
        + " "
        + example["output"]
    )


def load_jsonl(path: str) -> list:
    """Read a JSONL dataset file into a list of dicts."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def parse_args():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for loan agreement extraction"
    )
    parser.add_argument(
        "--base_model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HF model id of the base model to quantize and fine-tune",
    )
    parser.add_argument("--train_file", default="fine_tuning/dataset/train.jsonl")
    parser.add_argument("--val_file", default="fine_tuning/dataset/val.jsonl")
    parser.add_argument("--output_dir", default="fine_tuning/adapters/loan_extraction")
    # Values below match the blueprint's Phase 5 spec exactly (rank 16 / alpha 32 / 3 epochs) —
    # chosen as a reasonable default for ~100 labeled examples on a T4, not independently tuned.
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha (scaling factor)"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Effective batch size = batch_size * grad_accum_steps",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()

    # Imports deferred to inside main() so this file can be inspected/imported (e.g. for
    # its constants and argparse config) without torch/transformers/peft installed.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(
            f"{args.train_file} not found — hand-label ~100 examples as JSONL "
            "(80/20 train/val split) per blueprint Phase 5 before running this script."
        )

    # 4-bit quantization config (QLoRA) — this is what shrinks a 7B model to ~4GB,
    # making a free T4 (16GB VRAM) sufficient.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # Freezes base weights and preps norm layers/gradient checkpointing for 4-bit training
    model = prepare_model_for_kbit_training(model)

    # Only these adapter layers get trained — under 1% of total parameters, which is
    # why this fits on a T4 and trains in a few minutes per epoch on ~100 examples.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],  # attention projections — standard LoRA target for Mistral/Llama-family models
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(example):
        prompt = build_prompt(example)
        tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )
        tokens["labels"] = tokens[
            "input_ids"
        ].copy()  # causal LM: predict the same sequence shifted by one
        return tokens

    train_examples = load_jsonl(args.train_file)
    train_dataset = Dataset.from_list(train_examples).map(
        tokenize, remove_columns=["input", "output"]
    )

    eval_dataset = None
    if os.path.exists(args.val_file):
        val_examples = load_jsonl(args.val_file)
        eval_dataset = Dataset.from_list(val_examples).map(
            tokenize, remove_columns=["input", "output"]
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,  # T4 supports bf16; more numerically stable than fp16 for QLoRA
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        report_to="none",  # no wandb/hub dependency for a Colab run
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # Save only the adapter (~50-100MB), not the full base model — the base model is
    # re-downloaded/re-quantized at inference time and the adapter is applied on top.
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
