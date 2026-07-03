#!/usr/bin/env python3
"""
Classify thoracic CT reports using a fine-tuned ModernBERT model.

By default, uses the pre-trained model:
https://huggingface.co/anonymous-usr/thomas-sounack_BioClinical-ModernBERT-base_report_classifier
"""

import argparse
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Default model (fine-tuned on German reports translated to English)
DEFAULT_MODEL = "anonymous-usr/thomas-sounack_BioClinical-ModernBERT-base_report_classifier"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify CT reports using a fine-tuned ModernBERT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input CSV file with reports",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model name or path to local checkpoint",
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output CSV file. If not specified, saves to ../results/",
    )
    parser.add_argument(
        "--findings-col",
        default="Findings_EN",
        help="Column name for findings",
    )
    parser.add_argument(
        "--impressions-col",
        default="Impressions_EN",
        help="Column name for impressions",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=3000,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )

    return parser.parse_args()


def get_output_path(input_path: str, model_name: str) -> str:
    """Generate output path based on naming convention."""
    input_stem = Path(input_path).stem
    model_clean = model_name.split("/")[-1].replace(":", "-")

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)

    return str(results_dir / f"{input_stem}_{model_clean}.csv")


def tokenize_pair(df: pd.DataFrame, tokenizer, max_length: int):
    """Tokenize findings + impressions as text pair."""
    def tokenize_batch(batch):
        return tokenizer(
            batch["findings"],
            text_pair=batch["impressions"],
            truncation=True,
            max_length=max_length,
        )

    ds = Dataset.from_pandas(df[["findings", "impressions"]], preserve_index=False)
    return ds.map(tokenize_batch, batched=True, remove_columns=["findings", "impressions"])


def main():
    args = parse_args()

    output_path = args.output or get_output_path(args.input, args.model)

    print(f"Loading model from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    model.eval()

    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input, dtype=str).fillna("")

    if args.findings_col not in df.columns:
        raise ValueError(f"Findings column '{args.findings_col}' not found")
    if args.impressions_col not in df.columns:
        raise ValueError(f"Impressions column '{args.impressions_col}' not found")

    df_features = df.copy()
    df_features["findings"] = df_features[args.findings_col].fillna("").astype(str).str.strip()
    df_features["impressions"] = df_features[args.impressions_col].fillna("").astype(str).str.strip()

    print(f"Total samples: {len(df_features)}")

    ds = tokenize_pair(df_features, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=os.path.dirname(output_path) or ".",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print("Running inference...")
    pred_out = trainer.predict(ds)
    preds = pred_out.predictions.argmax(axis=-1)

    # Add prediction column
    out_df = df.copy()
    out_df.insert(0, "Predicted_label", preds.astype(int))

    out_df.to_csv(output_path, sep=";", index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
