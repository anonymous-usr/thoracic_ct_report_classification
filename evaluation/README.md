# Evaluation

This module provides scripts for evaluating and comparing classification results.

## Requirements

- Python dependencies (see root `requirements.txt`)

## Files

| File | Description |
|------|-------------|
| `metrics.py` | Compute metrics with bootstrap confidence intervals |
| `compare.py` | Paired bootstrap test to compare two models |

## Computing Metrics

Evaluate a single model's predictions:

```bash
python metrics.py <prediction_csv>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `csv_path` | (required) | Path to prediction CSV file |
| `--label-col` | `Label` | Ground truth column name |
| `--bootstrap-iters` | `10000` | Number of bootstrap iterations |
| `--seed` | `42` | Random seed |

### Example

```bash
python metrics.py ../results/<prediction_csv>
```

## Comparing Models

Compare two models using paired bootstrap test:

```bash
python compare.py <model_a_csv> <model_b_csv>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `csv_a` | (required) | First model's predictions (Model A) |
| `csv_b` | (required) | Second model's predictions (Model B) |
| `--metric` | `macro_f1` | Metric to compare |
| `--label-col` | `Label` | Ground truth column name |
| `--bootstrap-iters` | `10000` | Number of bootstrap iterations |
| `--seed` | `42` | Random seed |

### Available Metrics

- `macro_f1` - Macro-averaged F1 score
- `accuracy` - Overall accuracy
- `f1_class_0` - F1 for class 0 (normal)
- `f1_class_1` - F1 for class 1 (non-actionable)
- `f1_class_2` - F1 for class 2 (actionable)

### Example

```bash
python compare.py \
    ../results/external_test_set_llama3.1-8b_2step.csv \
    ../results/external_test_set_BioClinicalModernBERT.csv \
    --metric macro_f1
```

## Prediction Column Auto-Detection

Both scripts automatically detect the prediction column:
- **LLM results**: Uses `Predicted_label_cleaned` (handles parsing failures as -1)
- **ModernBERT results**: Uses `Predicted_label`

## Input Format

Expected CSV format:
- **Delimiter**: Semicolon (`;`)
- **Encoding**: UTF-8
- **Required columns**:
  - `Label` - Ground truth labels (0, 1, 2)
  - `Predicted_label` or `Predicted_label_cleaned` - Model predictions
