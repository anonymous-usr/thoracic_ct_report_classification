#!/usr/bin/env python3
"""
Fine-tune ModernBERT models for thoracic CT report classification.

Trains a 3-class classifier using findings + impressions as input pair.

NOTE: The pre-trained model was trained on German thoracic CT reports
(translated to English). Training on the external CT-RATE-based test set
will produce different results. For inference with the pre-trained model, use
classify_reports.py which downloads the checkpoint from HuggingFace.
"""

import argparse
import os
import random
import shutil

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
import torch
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ModernBERT for CT report classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to training CSV file",
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Directory to save model checkpoint",
    )
    parser.add_argument(
        "-m", "--model",
        default="answerdotai/ModernBERT-base",
        help="HuggingFace model name or path",
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
        "--label-col",
        default="Label",
        help="Column name for labels",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.9,
        help="Fraction of data for training (rest for validation)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=3000,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def prepare_dataframe(
    df: pd.DataFrame,
    findings_col: str,
    impressions_col: str,
    label_col: str,
) -> pd.DataFrame:
    """Prepare dataframe with standardized columns."""
    df = df.copy()

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    labels = pd.to_numeric(df[label_col].astype(str).str.strip(), errors="coerce")
    invalid = labels.isna() | ~labels.isin([0, 1, 2])
    if invalid.any():
        raise ValueError(f"Invalid labels found at rows: {df.index[invalid].tolist()[:10]}")

    df["label"] = labels.astype(int)
    df["findings"] = df[findings_col].fillna("").astype(str).str.strip()
    df["impressions"] = df[impressions_col].fillna("").astype(str).str.strip()

    df = df[(df["findings"] != "") | (df["impressions"] != "")].reset_index(drop=True)
    return df


def stratified_split(df: pd.DataFrame, train_frac: float, seed: int):
    """Stratified train/validation split."""
    rng = random.Random(seed)
    train_idx, val_idx = [], []

    for cls in [0, 1, 2]:
        idxs = df.index[df["label"] == cls].tolist()
        rng.shuffle(idxs)
        n_train = int(len(idxs) * train_frac)
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
    )


def tokenize_pair(df: pd.DataFrame, tokenizer, max_length: int):
    """Tokenize findings + impressions as text pair."""
    def tokenize_batch(batch):
        return tokenizer(
            batch["findings"],
            text_pair=batch["impressions"],
            truncation=True,
            max_length=max_length,
        )

    ds = Dataset.from_pandas(df[["findings", "impressions", "label"]], preserve_index=False)
    return ds.map(tokenize_batch, batched=True, remove_columns=["findings", "impressions"])


class F1PlotCallback(TrainerCallback):
    """Save F1 plot after each epoch."""

    def __init__(self, output_dir: str, trainer_ref: list):
        self.output_dir = output_dir
        self.trainer_ref = trainer_ref
        self.epochs = []
        self.train_f1 = []
        self.val_f1 = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_f1_macro" not in metrics:
            return

        trainer = self.trainer_ref[0]
        if trainer.train_dataset is None:
            return

        pred_out = trainer.predict(trainer.train_dataset)
        train_preds = np.argmax(pred_out.predictions, axis=-1)
        train_f1 = f1_score(pred_out.label_ids, train_preds, average="macro")

        self.epochs.append(state.epoch or len(self.epochs) + 1)
        self.train_f1.append(train_f1)
        self.val_f1.append(metrics["eval_f1_macro"])

        self._save_plot()

    def _save_plot(self):
        plt.figure(figsize=(8, 4.5))
        plt.plot(self.epochs, self.train_f1, marker="o", label="Train F1")
        plt.plot(self.epochs, self.val_f1, marker="o", label="Val F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "f1_curve.png"), dpi=150)
        plt.close()

        pd.DataFrame({
            "epoch": self.epochs,
            "train_f1": self.train_f1,
            "val_f1": self.val_f1,
        }).to_csv(os.path.join(self.output_dir, "f1_curve.csv"), index=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input, dtype=str).fillna("")

    df = prepare_dataframe(df, args.findings_col, args.impressions_col, args.label_col)
    print(f"Total samples: {len(df)}")

    train_df, val_df = stratified_split(df, args.train_frac, args.seed)
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)

    train_ds = tokenize_pair(train_df, tokenizer, args.max_length)
    val_ds = tokenize_pair(val_df, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )

    trainer_ref = [None]
    f1_callback = F1PlotCallback(args.output_dir, trainer_ref)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            f1_callback,
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.0005,
            ),
        ],
    )
    trainer_ref[0] = trainer

    print(f"\nModel: {args.model}")
    print(f"Device: {training_args.device}")
    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)}")

    trainer.train()

    # Save best checkpoint
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Clean up intermediate checkpoints
    for name in os.listdir(args.output_dir):
        path = os.path.join(args.output_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            shutil.rmtree(path, ignore_errors=True)

    # Final validation results
    pred_out = trainer.predict(val_ds)
    preds = np.argmax(pred_out.predictions, axis=-1)
    print("\nValidation Results:")
    print(classification_report(
        pred_out.label_ids,
        preds,
        target_names=["0 (normal)", "1 (findings-level)", "2 (impression-level)"],
        digits=4,
    ))

    print(f"\nModel saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
