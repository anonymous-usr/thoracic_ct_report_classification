#!/usr/bin/env python3
"""
Compare two classification models using paired bootstrap testing.

Computes the difference in performance metrics between two models on the same
test set and provides confidence intervals and p-values.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two models using paired bootstrap test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_a",
        help="Path to first model's prediction CSV (Model A)",
    )
    parser.add_argument(
        "csv_b",
        help="Path to second model's prediction CSV (Model B)",
    )
    parser.add_argument(
        "--metric",
        choices=["macro_f1", "accuracy", "f1_class_0", "f1_class_1", "f1_class_2"],
        default="macro_f1",
        help="Metric to compare",
    )
    parser.add_argument(
        "--label-col",
        default="Label",
        help="Column name for ground truth labels",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=10000,
        help="Number of bootstrap iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def get_prediction_column(df):
    """
    Auto-detect prediction column.
    LLMs use Predicted_label_cleaned, ModernBERT uses Predicted_label.
    """
    if "Predicted_label_cleaned" in df.columns:
        return "Predicted_label_cleaned"
    if "Predicted_label" in df.columns:
        return "Predicted_label"
    raise ValueError(
        "Missing prediction column. Expected 'Predicted_label_cleaned' or 'Predicted_label'."
    )


def load_predictions(csv_path, label_col):
    """Load ground truth and predictions from CSV."""
    df = pd.read_csv(csv_path, sep=";")

    pred_col = get_prediction_column(df)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

    y_true = pd.to_numeric(df[label_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")

    valid = y_true.notna() & y_pred.notna() & y_pred.isin([0, 1, 2])

    return (
        y_true[valid].astype(int).to_numpy(),
        y_pred[valid].astype(int).to_numpy(),
        pred_col,
    )


def compute_metric(y_true, y_pred, metric):
    """Compute the specified metric."""
    if metric == "macro_f1":
        return f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    elif metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "f1_class_0":
        return f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)[0]
    elif metric == "f1_class_1":
        return f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)[1]
    elif metric == "f1_class_2":
        return f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)[2]
    else:
        raise ValueError(f"Unknown metric: {metric}")


def paired_bootstrap_test(y_true, y_pred_a, y_pred_b, metric, n_bootstrap, seed):
    """
    Paired bootstrap test comparing Model A vs Model B.

    Returns observed difference (A - B), 95% CI, and two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    observed_a = compute_metric(y_true, y_pred_a, metric)
    observed_b = compute_metric(y_true, y_pred_b, metric)
    observed_diff = observed_a - observed_b

    boot_diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_a = compute_metric(y_true[idx], y_pred_a[idx], metric)
        boot_b = compute_metric(y_true[idx], y_pred_b[idx], metric)
        boot_diffs[i] = boot_a - boot_b

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    # Two-sided p-value
    if observed_diff >= 0:
        p_value = 2 * np.mean(boot_diffs <= 0)
    else:
        p_value = 2 * np.mean(boot_diffs >= 0)

    p_value = min(float(p_value), 1.0)

    return {
        "metric": metric,
        "score_a": observed_a,
        "score_b": observed_b,
        "diff": observed_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def main():
    args = parse_args()

    try:
        y_true_a, y_pred_a, col_a = load_predictions(args.csv_a, args.label_col)
        y_true_b, y_pred_b, col_b = load_predictions(args.csv_b, args.label_col)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    if not np.array_equal(y_true_a, y_true_b):
        print("Warning: Ground-truth labels differ between files.", file=sys.stderr)
        print("Make sure both files contain the same test samples in the same order.", file=sys.stderr)
        sys.exit(1)

    if len(y_true_a) == 0:
        print("Error: No valid samples found.", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing models on {len(y_true_a)} samples")
    print(f"Model A: {args.csv_a}")
    print(f"  Prediction column: {col_a}")
    print(f"Model B: {args.csv_b}")
    print(f"  Prediction column: {col_b}")
    print()

    result = paired_bootstrap_test(
        y_true_a,
        y_pred_a,
        y_pred_b,
        metric=args.metric,
        n_bootstrap=args.bootstrap_iters,
        seed=args.seed,
    )

    print("=" * 60)
    print("PAIRED BOOTSTRAP TEST RESULTS")
    print("=" * 60)
    print(f"Metric              : {result['metric']}")
    print(f"Model A score       : {result['score_a']:.4f}")
    print(f"Model B score       : {result['score_b']:.4f}")
    print(f"Difference (A - B)  : {result['diff']:+.4f}")
    print(f"95% CI              : [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    print(f"p-value             : {result['p_value']:.4f}")
    print("=" * 60)

    if result['p_value'] < 0.05:
        if result['diff'] > 0:
            print("Model A is significantly better than Model B (p < 0.05)")
        else:
            print("Model B is significantly better than Model A (p < 0.05)")
    else:
        print("No significant difference between models (p >= 0.05)")


if __name__ == "__main__":
    main()
