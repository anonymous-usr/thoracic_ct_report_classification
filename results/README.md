# Results

This directory contains predictions on the external CT-RATE test set from the model configurations that performed best on the development set.

## Files

| File | Model | Strategy |
|------|-------|----------|
| `external_test_set_BioClinical-ModernBERT-base.csv` | BioClinical-ModernBERT-base | Fine-tuned |
| `external_test_set_llama3.1-8b_2step.csv` | Llama 3.1 8B | 2-step prompting |
| `external_test_set_gpt-oss-120b_cot.csv` | GPT-OSS 120B | Chain-of-thought |

## File Format

All files are semicolon-separated CSVs containing only essential columns (predictions and labels). The full report text is omitted to reduce file size.

**ModernBERT results:**
| Column | Description |
|--------|-------------|
| `Predicted_label` | Model prediction (0, 1, 2) |
| `VolumeName` | CT-RATE volume identifier |
| `Label` | Ground truth label (0, 1, 2) |

**LLM results:**
| Column | Description |
|--------|-------------|
| `Predicted_label_cleaned` | Parsed prediction (0, 1, 2, or -1 for failures) |
| `Predicted_label` | Raw LLM output |
| `VolumeName` | CT-RATE volume identifier |
| `Label` | Ground truth label (0, 1, 2) |

## Running Evaluation

Compute metrics for a single model:

```bash
python ../evaluation/metrics.py <prediction_csv>
```

Compare two models:

```bash
python ../evaluation/compare.py <model_a_csv> <model_b_csv> --metric macro_f1
```
