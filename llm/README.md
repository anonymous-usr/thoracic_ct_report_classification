# LLM-based Classification

This module provides LLM-based classification of thoracic CT reports using [Ollama](https://ollama.ai/).

## Files

| File | Description |
|------|-------------|
| `classify_reports.py` | Main classification script |
| `prompts.py` | Prompt templates for all strategies |

## Requirements

- [Ollama](https://ollama.ai/) installed and running locally
- Python dependencies (see root `requirements.txt`)

## Installation

1. Install Ollama following the instructions at https://ollama.ai/

2. Pull the desired model:
   ```bash
   ollama pull <model>
   ```

3. Verify Ollama is running:
   ```bash
   ollama list
   ```

## Usage

```bash
python classify_reports.py -i <input_csv> -m <model> -s <strategy>
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input` | (required) | Path to input CSV file |
| `-m, --model` | (required) | Ollama model name |
| `-s, --strategy` | (required) | Prompting strategy |
| `-o, --output` | auto-generated | Output CSV path |
| `--temperature` | `0.0` | LLM temperature |
| `--findings-col` | `Findings_EN` | Column name for findings |
| `--impressions-col` | `Impressions_EN` | Column name for impressions |
| `--checkpoint-every` | `20` | Checkpoint frequency |
| `--seed` | `42` | Random seed for few-shot |
| `--label-col` | `Label` | Label column (required for 3-shot) |

### Prompting Strategies

| Strategy | Description |
|----------|-------------|
| `0-shot` | Direct classification without examples |
| `system` | Uses a system prompt defining the classifier role |
| `rule-guided` | Rule-guided with silent reasoning steps |
| `3-shot` | Three examples (one per class) included in prompt |
| `2-step` | Two-stage: first check impressions, then findings |

### Example

```bash
python classify_reports.py -i ../data/external_test_set.csv -m llama3.1:8b -s rule-guided
```

## Input Format

Expected CSV format:
- **Delimiter**: Comma (`,`)
- **Encoding**: UTF-8
- **Required columns**:
  - `Findings_EN` - Findings section of the radiology report
  - `Impressions_EN` - Impressions section of the radiology report
  - `Label` - Ground truth labels (only required for 3-shot strategy)

## Output Format

The output CSV contains all original columns plus:
- `Predicted_label`: Raw LLM output
- `Predicted_label_cleaned`: Parsed label (0, 1, 2, or -1 for failed parsing)

Output is saved to `../results/` with naming convention:
```
{input_filename}_{model}_{strategy}.csv
```

## Supported Models

Models evaluated in the paper:

| Model | Size |
|-------|------|
| [`mistral:7b`](https://ollama.com/library/mistral:7b) | 7B |
| [`llama3.1:8b`](https://ollama.com/library/llama3.1:8b) | 8B |
| [`hf.co/bartowski/Llama-3-SauerkrautLM-8b-Instruct-GGUF:q4_K_M`](https://huggingface.co/bartowski/Llama-3-SauerkrautLM-8b-Instruct-GGUF) | 8B |
| [`medllama2:7b`](https://ollama.com/library/medllama2:7b) | 7B |
| [`hf.co/BioMistral/BioMistral-7B-GGUF:Q4_K_M`](https://huggingface.co/BioMistral/BioMistral-7B-GGUF) | 7B |
| [`llama3.3:70b`](https://ollama.com/library/llama3.3:70b) | 70B |
| [`gpt-oss:120b`](https://ollama.com/library/gpt-oss:120b) | 120B |
| [`mistral-large:123b`](https://ollama.com/library/mistral-large:123b) | 123B |

Any model available through Ollama can be used. See https://ollama.ai/library
