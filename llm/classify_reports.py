#!/usr/bin/env python3
"""
LLM-based classification of thoracic CT reports.

Classifies radiology reports into three categories based on lung findings:
- 0: Normal (no current lung-related abnormality)
- 1: Findings-level abnormal (abnormality in findings only)
- 2: Impression-level abnormal (abnormality in impressions)

Requires Ollama to be installed and running locally.
"""

import argparse
import os
import re
import random
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from ollama import chat

from prompts import get_prompt, get_system_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify thoracic CT reports using LLMs via Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input CSV file with reports",
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output CSV file. If not specified, saves to ../results/ with naming convention: {input_name}_{model}_{strategy}.csv",
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Ollama model name (e.g., llama3.1:8b, mistral:7b)",
    )
    parser.add_argument(
        "-s", "--strategy",
        choices=["0-shot", "system", "rule-guided", "3-shot", "2-step"],
        required=True,
        help="Prompting strategy",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (0 = deterministic)",
    )
    parser.add_argument(
        "--findings-col",
        default="Findings_EN",
        help="Column name for findings/results section",
    )
    parser.add_argument(
        "--impressions-col",
        default="Impressions_EN",
        help="Column name for impressions/conclusion section",
    )
    parser.add_argument(
        "--label-col",
        default="Label",
        help="Column name for labels (required for 3-shot strategy)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Save checkpoint every N samples",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per sample on LLM failure",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for few-shot example selection",
    )

    return parser.parse_args()


def clean_label(response: str) -> int:
    """
    Extract a single label (0, 1, or 2) from the LLM response.
    Returns -1 if extraction fails.
    """
    matches = re.findall(r'(?<!\d)([012])(?!\d)', response)
    return int(matches[0]) if len(matches) == 1 else -1


def run_llm(messages: list[dict], model: str, temperature: float) -> str:
    """Send messages to Ollama and return the response."""
    response = chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
        stream=False,
    )
    return response["message"]["content"].strip()


def build_few_shot_examples(
    df: pd.DataFrame,
    current_idx: int,
    findings_col: str,
    impressions_col: str,
    label_col: str,
    seed: int,
) -> str:
    """Build few-shot examples string (one example per class)."""
    labels_num = pd.to_numeric(df[label_col], errors="coerce")
    rng = random.Random(seed + current_idx)

    examples = []
    for cls in [0, 1, 2]:
        candidates = df.index[(labels_num == cls) & (df.index != current_idx)].tolist()
        if not candidates:
            continue
        idx = rng.choice(candidates)
        row = df.loc[idx]
        label = int(str(row[label_col]).strip())
        findings = str(row.get(findings_col, "") or "").strip()
        impressions = str(row.get(impressions_col, "") or "").strip()

        examples.append(
            f"[EXAMPLE {len(examples) + 1}]\n"
            f"[Result]\n{findings}\n\n"
            f"[Conclusion]\n{impressions}\n\n"
            f"[Class] {label}\n"
        )

    rng.shuffle(examples)
    return "\n\n".join(examples)


def classify_single(
    findings: str,
    impressions: str,
    model: str,
    strategy: str,
    temperature: float,
    few_shot_examples: str = "",
) -> str:
    """Classify a single report and return the predicted label as string."""
    prompt_template = get_prompt(strategy)

    if strategy == "2-step":
        conclusion_prompt, findings_prompt = prompt_template

        # Step 1: Check conclusion
        msg1 = [{"role": "user", "content": conclusion_prompt.format(conclusion=impressions)}]
        answer1 = clean_label(run_llm(msg1, model, temperature))

        if answer1 == 1:
            return "2"
        elif answer1 == 0:
            # Step 2: Check findings
            msg2 = [{"role": "user", "content": findings_prompt.format(result=findings)}]
            answer2 = clean_label(run_llm(msg2, model, temperature))

            if answer2 == 1:
                return "1"
            elif answer2 == 0:
                return "0"
            else:
                return "-1"
        else:
            return "-1"

    elif strategy == "system":
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt_template.format(result=findings, conclusion=impressions)},
        ]
        return run_llm(messages, model, temperature)

    elif strategy == "3-shot":
        filled_prompt = prompt_template.format(
            examples=few_shot_examples,
            result=findings,
            conclusion=impressions,
        )
        messages = [{"role": "user", "content": filled_prompt}]
        return run_llm(messages, model, temperature)

    else:  # 0-shot, rule-guided
        filled_prompt = prompt_template.format(result=findings, conclusion=impressions)
        messages = [{"role": "user", "content": filled_prompt}]
        return run_llm(messages, model, temperature)


def safe_checkpoint_save(df: pd.DataFrame, path: str):
    """Atomically save checkpoint to avoid corruption."""
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, sep=";", index=False)
    os.replace(tmp_path, path)


def get_output_path(input_path: str, model: str, strategy: str) -> str:
    """Generate output path based on naming convention."""
    input_stem = Path(input_path).stem
    model_clean = model.replace(":", "-").replace("/", "-")
    strategy_clean = strategy.replace("-", "")

    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)

    return str(results_dir / f"{input_stem}_{model_clean}_{strategy_clean}.csv")


def main():
    args = parse_args()

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        out_path = get_output_path(args.input, args.model, args.strategy)

    checkpoint_path = out_path.replace(".csv", "_checkpoint.csv")

    # Load data (resume from checkpoint if available)
    if os.path.exists(out_path):
        print(f"Output file exists, loading: {out_path}")
        df = pd.read_csv(out_path, sep=";", dtype=str).fillna("")
    elif os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        df = pd.read_csv(checkpoint_path, sep=";", dtype=str).fillna("")
    else:
        print(f"Loading input: {args.input}")
        df = pd.read_csv(args.input, sep=",", dtype=str).fillna("")
        df.insert(0, "Predicted_label", "")
        df.insert(0, "Predicted_label_cleaned", "")

    # Validate columns
    if args.findings_col not in df.columns:
        sys.exit(f"Error: Findings column '{args.findings_col}' not found in CSV")
    if args.impressions_col not in df.columns:
        sys.exit(f"Error: Impressions column '{args.impressions_col}' not found in CSV")

    print(f"Model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print(f"Output: {out_path}")
    print(f"Total samples: {len(df)}")

    # Process each report
    for idx in tqdm(range(len(df)), desc="Classifying"):
        row = df.iloc[idx]

        # Skip if already processed
        if str(row.get("Predicted_label", "")).strip():
            continue

        findings = str(row.get(args.findings_col, "") or "").strip()
        impressions = str(row.get(args.impressions_col, "") or "").strip()

        # Build few-shot examples if needed
        few_shot_examples = ""
        if args.strategy == "3-shot" and args.label_col in df.columns:
            few_shot_examples = build_few_shot_examples(
                df, idx, args.findings_col, args.impressions_col, args.label_col, args.seed
            )

        # Classify with retries
        for attempt in range(args.max_retries):
            try:
                predicted = classify_single(
                    findings=findings,
                    impressions=impressions,
                    model=args.model,
                    strategy=args.strategy,
                    temperature=args.temperature,
                    few_shot_examples=few_shot_examples,
                )
                break
            except Exception as e:
                print(f"\nError at row {idx}, attempt {attempt + 1}/{args.max_retries}: {e}")
                if attempt == args.max_retries - 1:
                    print("Max retries reached. Stopping.")
                    sys.exit(1)
                time.sleep(1)

        df.at[idx, "Predicted_label"] = predicted

        # Checkpoint save
        if (idx + 1) % args.checkpoint_every == 0:
            safe_checkpoint_save(df, checkpoint_path)

    # Final processing and save
    df["Predicted_label_cleaned"] = df["Predicted_label"].apply(clean_label)
    df.to_csv(out_path, sep=";", index=False)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
