# Verifoto Deep Learning

Training pipeline for fraud detection in food delivery images. Part of the Verifoto-AI SaaS platform that helps restaurants detect manipulated photos used to obtain undeserved refunds.

## About Verifoto

Verifoto-AI is a web app for restaurants working with food delivery to defend against photographic fraud. Some customers digitally manipulate food images (adding fake mold, artificial burns, or AI-generated defects) to get unjustified refunds. This project trains the deep learning model that powers the fraud detection.

**Key Principle**: Conservative approach - optimize for precision over recall. Better to miss some fraud than falsely accuse legitimate complaints.

## This Repository

Pipeline for training and evaluating fraud detection models on Google Colab with GPU, while keeping code versioned on GitHub and checkpoints on Google Drive.

## Project Structure

```
verifoto-dl/
├── src/                  # Core training/evaluation code
├── configs/              # YAML configurations
├── scripts/              # Helper scripts + Colab notebook
├── docs/                 # User documentation
├── .kiro/                # AI assistant context (steering + agent state)
├── outputs/runs/         # Training results (versioned)
└── checkpoints/          # Model weights (NOT versioned, on Drive)
```

For AI assistant working context, see `.kiro/USAGE.md`.

## Quick Start

### 1. First Time Setup

```bash
# Clone and setup
git clone https://github.com/<YOUR_USERNAME>/verifoto-dl.git
cd verifoto-dl

# Test setup (optional)
pip install -r requirements.txt
python scripts/quick_test.py
```

**Note**: This project now supports the new `augmented_v6` dataset structure with hierarchical metadata. See `docs/AUGMENTED_V6_DATASET.md` for details.

### 2. Develop Locally (Kiro)

```bash
# Edit code in src/ or configs/
# Commit and push
git add .
git commit -m "Update model architecture"
git push
```

### 3. Train on Colab

See `scripts/colab_bootstrap.md` for complete Colab setup. Quick version:

```python
# In Colab
!git clone https://github.com/<USER>/verifoto-dl.git && cd verifoto-dl
!pip install -q -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

# Run training
!python -m src.train \
    --config configs/baseline.yaml \
    --run_name "2026-02-16_baseline" \
    --checkpoint_dir "/content/drive/MyDrive/verifoto_checkpoints"
```

### 4. Analyze Results

```bash
# Pull results
git pull

# Compare all runs
python scripts/compare_runs.py

# View specific run
cat outputs/runs/<run_name>/metrics.json
```

For detailed workflow, see `docs/WORKFLOW.md`.

## CLI Usage

### Training

```bash
python -m src.train \
    --config configs/baseline.yaml \
    --run_name "2026-02-16_baseline1" \
    --checkpoint_dir "/content/drive/MyDrive/verifoto_checkpoints"
```

### Evaluation

```bash
python -m src.eval \
    --config configs/baseline.yaml \
    --run_name "2026-02-16_eval" \
    --checkpoint_path "/path/to/checkpoint.pt" \
    --threshold 0.5
```

## Output Format

Each run creates:

```
outputs/runs/<run_name>/
├── metrics.json       # All metrics + config
├── notes.md           # Human-readable summary
├── predictions.csv    # All predictions with metadata
├── group_metrics_food.csv        # Metrics by food category
├── group_metrics_defect.csv      # Metrics by defect type
├── group_metrics_generator.csv   # Metrics by generator
├── group_metrics_quality.csv     # Metrics by quality
├── top_false_positives.csv       # Top FP errors
├── top_false_negatives.csv       # Top FN errors
├── cm.png             # Confusion matrix
├── roc_curve.png      # ROC curve
├── pr_curve.png       # Precision-Recall curve
└── prob_dist.png      # Probability distribution
```

See `docs/AUGMENTED_V6_DATASET.md` for details on metadata and analysis.

### metrics.json Structure

```json
{
  "run_name": "2026-02-16_baseline1",
  "git_commit": "abc123...",
  "timestamp": "2026-02-16 14:30:00",
  "threshold": 0.5,
  "test_metrics": {
    "acc": 0.95,
    "prec": 0.93,
    "rec": 0.91,
    "f1": 0.92,
    "pr_auc": 0.96,
    "roc_auc": 0.97
  },
  "confusion_matrix": [[tn, fp], [fn, tp]],
  "config": {...}
}
```

## Configuration

Edit `configs/baseline.yaml` to change:

- Model architecture (`model_name`)
- Training hyperparameters
- Data augmentation (edit `src/utils/data.py`)
- Dataset path

## Key Features

- **Reproducible**: Fixed seeds, git commit tracking
- **Efficient**: Checkpoints on Drive, only lightweight results on GitHub
- **Fast iteration**: Pull code changes in Colab, no manual file copying
- **AI-friendly output**: Structured JSON for easy parsing
- **Group-aware splitting**: Prevents data leakage from near-duplicates
- **Metadata tracking**: Analyze errors by food category, defect type, generator
- **Conservative approach**: Optimized for high precision (low false positives)

## Production Context

This model will be deployed in the Verifoto-AI web app where:
- Restaurants upload suspicious images
- Model provides confidence score + technical indicators
- LLM generates human-readable explanation
- Report used for internal decision-making
- **False positives damage trust** → precision is critical

## Tips

1. **Multiple experiments**: Create `configs/experiment2.yaml` for different setups
2. **Threshold tuning**: Use `eval.py` with different `--threshold` values
3. **Quick tests**: Reduce `epochs_head` and `epochs_finetune` in config for fast debugging
4. **Drive organization**: Keep checkpoints organized by date/experiment name

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (for training)
- Google Drive (for checkpoint storage)
