# ModernBERT Classification

This module provides fine-tuning and inference scripts for ModernBERT-based classification of thoracic CT reports.

## Files

| File | Description |
|------|-------------|
| `train.py` | Fine-tune a ModernBERT model on labeled reports |
| `classify_reports.py` | Run inference with a trained model |

## Requirements

- Python dependencies (see root `requirements.txt`)

## Pre-trained Model

The fine-tuned model that achieved the best results on the development set is available on HuggingFace:

**[anonymous-usr/thomas-sounack_BioClinical-ModernBERT-base_report_classifier](https://huggingface.co/anonymous-usr/thomas-sounack_BioClinical-ModernBERT-base_report_classifier)**

The inference script uses this model by default. For faster downloads, authenticate with HuggingFace:

```bash
hf auth login
```

> **Note:** The pre-trained model was fine-tuned on the development set consisting
> of reports from a German institution, translated to English for fine-tuning.
> Fine-tuning on the external CT-RATE-based test set will produce different results.

## Inference

Classify reports using the pre-trained model:

```bash
python classify_reports.py -i <input_csv>
```

### Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input` | (required) | Path to input CSV |
| `-m, --model` | see above | Model name or local checkpoint path |
| `-o, --output` | auto-generated | Output CSV path |
| `--findings-col` | `Findings_EN` | Column name for findings |
| `--impressions-col` | `Impressions_EN` | Column name for impressions |
| `--max-length` | `3000` | Maximum sequence length |
| `--batch-size` | `8` | Batch size for inference |

### Example

```bash
# Using the default pre-trained model
python classify_reports.py -i ../data/external_test_set.csv

# Using a custom model
python classify_reports.py -i ../data/external_test_set.csv -m ./my_checkpoint
```

## Input Format

Expected CSV format:
- **Delimiter**: Comma (`,`)
- **Encoding**: UTF-8
- **Required columns**:
  - `Findings_EN` - Findings section of the radiology report
  - `Impressions_EN` - Impressions section of the radiology report
  - `Label` - Ground truth labels (only required for training)

## Output Format

Output CSV contains all original columns plus:
- `Predicted_label` - Predicted class (0, 1, or 2)

## Training

Fine-tune a model on your own labeled dataset:

```bash
python train.py -i <training_csv> -o <output_dir> -m <model_name>
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input` | (required) | Path to training CSV |
| `-o, --output-dir` | (required) | Directory for model checkpoint |
| `-m, --model` | `answerdotai/ModernBERT-base` | Base model from HuggingFace |
| `--findings-col` | `Findings_EN` | Column name for findings |
| `--impressions-col` | `Impressions_EN` | Column name for impressions |
| `--label-col` | `Label` | Column name for labels (0/1/2) |
| `--train-frac` | `0.9` | Train/validation split ratio |
| `--epochs` | `10` | Maximum training epochs |
| `--learning-rate` | `4e-5` | Learning rate |
| `--batch-size` | `4` | Batch size per device |
| `--max-length` | `3000` | Maximum sequence length |
| `--early-stopping-patience` | `2` | Early stopping patience |
| `--seed` | `42` | Random seed |

### Example

```bash
python train.py \
    -i ../data/external_test_set.csv \
    -o ./models/my_model \
    -m thomas-sounack/BioClinical-ModernBERT-base \
    --epochs 5
```

### Output

Training produces:
- `checkpoint/` - Best model weights and tokenizer
- `f1_curve.png` - Training/validation F1 over epochs
- `f1_curve.csv` - F1 values per epoch

## Supported Base Models

Models evaluated in the paper:
- [`LSX-UniWue/ModernGBERT_134M`](https://huggingface.co/LSX-UniWue/ModernGBERT_134M)
- [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
- [`thomas-sounack/BioClinical-ModernBERT-base`](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base)

Any ModernBERT-compatible model from HuggingFace can be used.
