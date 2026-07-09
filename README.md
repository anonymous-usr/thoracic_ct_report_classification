# Thoracic CT Report Classification

This repository contains the implementation of the paper *Privacy-Preserving Classification of German Thoracic CT Reports Under Resource Constraints: Large Language Models vs Fine-Tuned Encoders for Report-Section-Based Lung Abnormality Assessment*.

## Repository Structure

```
thoracic_ct_report_classification/
├── data/           # External test set (1000 labeled reports from CT-RATE)
├── llm/            # LLM-based report classification via Ollama
├── modern_bert/    # ModernBERT training and inference
├── evaluation/     # Metrics and model comparison
└── results/        # Pre-computed predictions
```

## Installation

```bash
git clone https://github.com/anonymous-usr/thoracic_ct_report_classification.git
cd thoracic_ct_report_classification

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Modules

| Module | Description | Documentation |
|--------|-------------|---------------|
| `data/` | External test set from CT-RATE | [data/README.md](data/README.md) |
| `llm/` | LLM-based classification | [llm/README.md](llm/README.md) |
| `modern_bert/` | ModernBERT training and inference | [modern_bert/README.md](modern_bert/README.md) |
| `evaluation/` | Metrics, model comparison | [evaluation/README.md](evaluation/README.md) |
| `results/` | Pre-computed predictions | [results/README.md](results/README.md) |

## License

- **Code**: MIT License
- **Data**: CC-BY-NC-SA 4.0 (see `data/LICENSE`)

## Contact

For problems or questions, please open a [GitHub issue](https://github.com/anonymous-usr/thoracic_ct_report_classification/issues).
