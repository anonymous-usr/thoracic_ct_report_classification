#!/usr/bin/env python3
"""
Compute evaluation metrics for CT report classification results.

Computes class-wise F1, macro F1, accuracy, fail rate, and 95% bootstrap CIs.
Automatically detects prediction column (Predicted_label_cleaned for LLMs,
Predicted_label for ModernBERT).
"""

import argparse
import sys
import numpy as np
import pandas as pd


VALID_CLASSES = [0, 1, 2]
FAILED_LABEL = -1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute classification metrics with bootstrap confidence intervals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        help="Path to the prediction CSV file",
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
        help="Random seed for bootstrap",
    )
    return parser.parse_args()


def safe_div(num, den):
    return num / den if den != 0 else 0.0


def f1_from_counts(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


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


def coerce_label_series(series, name, allowed_values):
    """Convert series to integer labels, validating values."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad_values = series[numeric.isna()].dropna().unique().tolist()
        raise ValueError(f"Column '{name}' contains non-numeric values: {bad_values[:10]}")

    numeric = numeric.astype(int)
    invalid = sorted(set(numeric.unique()) - set(allowed_values))
    if invalid:
        raise ValueError(
            f"Column '{name}' contains invalid values {invalid}. Allowed: {allowed_values}"
        )
    return numeric


def compute_metrics(y_true, y_pred):
    """Compute all classification metrics."""
    n_total = len(y_true)
    n_failed = int(np.sum(y_pred == FAILED_LABEL))
    fail_rate = safe_div(n_failed, n_total)

    n_correct = int(np.sum(y_true == y_pred))
    accuracy = safe_div(n_correct, n_total)

    class_f1 = {}
    class_counts = {}

    for cls in VALID_CLASSES:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        support = int(np.sum(y_true == cls))

        class_f1[cls] = f1_from_counts(tp, fp, fn)
        class_counts[cls] = {"tp": tp, "fp": fp, "fn": fn, "support": support}

    macro_f1 = float(np.mean([class_f1[c] for c in VALID_CLASSES]))

    return {
        "n_total": n_total,
        "n_failed": n_failed,
        "fail_rate": fail_rate,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "class_f1": class_f1,
        "class_counts": class_counts,
        "macro_f1": macro_f1,
    }


def percentile_ci(values, alpha=0.05):
    """Compute percentile confidence interval."""
    low = float(np.percentile(values, 100 * (alpha / 2)))
    high = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return low, high


def stratified_bootstrap(y_true, y_pred, n_iters=10000, seed=42):
    """Compute bootstrap confidence intervals with stratified sampling."""
    rng = np.random.default_rng(seed)

    strata_indices = {cls: np.where(y_true == cls)[0] for cls in VALID_CLASSES}
    for cls, idx in strata_indices.items():
        if len(idx) == 0:
            raise ValueError(f"Cannot bootstrap: class {cls} has 0 samples.")

    acc_values = np.empty(n_iters, dtype=float)
    fail_values = np.empty(n_iters, dtype=float)
    macro_values = np.empty(n_iters, dtype=float)
    class_f1_values = {cls: np.empty(n_iters, dtype=float) for cls in VALID_CLASSES}

    for i in range(n_iters):
        sampled_idx = np.concatenate([
            rng.choice(strata_indices[cls], size=len(strata_indices[cls]), replace=True)
            for cls in VALID_CLASSES
        ])

        metrics = compute_metrics(y_true[sampled_idx], y_pred[sampled_idx])

        acc_values[i] = metrics["accuracy"]
        fail_values[i] = metrics["fail_rate"]
        macro_values[i] = metrics["macro_f1"]
        for cls in VALID_CLASSES:
            class_f1_values[cls][i] = metrics["class_f1"][cls]

    ci = {
        "accuracy": percentile_ci(acc_values),
        "fail_rate": percentile_ci(fail_values),
        "macro_f1": percentile_ci(macro_values),
    }
    for cls in VALID_CLASSES:
        ci[f"class_{cls}_f1"] = percentile_ci(class_f1_values[cls])

    return ci


def fmt_pct(x):
    return f"{x:.4f}"


def fmt_ci(ci):
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def print_results(metrics, ci, pred_col, label_col):
    """Print formatted evaluation results."""
    print("=" * 88)
    print("EVALUATION RESULTS")
    print("=" * 88)
    print(f"Ground-truth column : {label_col}")
    print(f"Prediction column   : {pred_col}")
    print(f"Total samples       : {metrics['n_total']}")
    print()

    print("OVERALL METRICS")
    print("-" * 88)
    print(
        f"Accuracy            : {fmt_pct(metrics['accuracy']):>8}   "
        f"({metrics['n_correct']}/{metrics['n_total']})   "
        f"95% CI {fmt_ci(ci['accuracy'])}"
    )
    print(
        f"Fail rate           : {fmt_pct(metrics['fail_rate']):>8}   "
        f"({metrics['n_failed']}/{metrics['n_total']})   "
        f"95% CI {fmt_ci(ci['fail_rate'])}"
    )
    print(
        f"Macro F1            : {fmt_pct(metrics['macro_f1']):>8}   "
        f"95% CI {fmt_ci(ci['macro_f1'])}"
    )
    print()

    print("CLASS-WISE F1")
    print("-" * 88)
    print(f"{'Class':<8}{'F1':>10}   {'95% CI':<24}{'TP':>8}{'FP':>8}{'FN':>8}{'Support':>10}")
    print("-" * 88)

    class_names = {0: "0 (normal)", 1: "1 (non-act)", 2: "2 (action)"}
    for cls in VALID_CLASSES:
        counts = metrics["class_counts"][cls]
        print(
            f"{class_names[cls]:<12}"
            f"{fmt_pct(metrics['class_f1'][cls]):>8}   "
            f"{fmt_ci(ci[f'class_{cls}_f1']):<24}"
            f"{counts['tp']:>6}"
            f"{counts['fp']:>8}"
            f"{counts['fn']:>8}"
            f"{counts['support']:>10}"
        )
    print("=" * 88)


def main():
    args = parse_args()

    try:
        df = pd.read_csv(args.csv_path, sep=";")
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if args.label_col not in df.columns:
        print(f"Missing label column: '{args.label_col}'", file=sys.stderr)
        sys.exit(1)

    try:
        pred_col = get_prediction_column(df)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    try:
        y_true = coerce_label_series(
            df[args.label_col], args.label_col, VALID_CLASSES
        ).to_numpy()
        y_pred = coerce_label_series(
            df[pred_col], pred_col, VALID_CLASSES + [FAILED_LABEL]
        ).to_numpy()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if len(y_true) == 0:
        print("Error: CSV contains no rows.", file=sys.stderr)
        sys.exit(1)

    print(f"Computing metrics with {args.bootstrap_iters} bootstrap iterations...")
    metrics = compute_metrics(y_true, y_pred)
    ci = stratified_bootstrap(y_true, y_pred, n_iters=args.bootstrap_iters, seed=args.seed)

    print()
    print_results(metrics, ci, pred_col, args.label_col)


if __name__ == "__main__":
    main()
