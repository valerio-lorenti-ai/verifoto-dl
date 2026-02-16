# Quick Start Guide

## 🚀 30-Second Start

### On Colab (Training)

1. Upload `scripts/Verifoto_Training.ipynb` to Colab
2. Update `GITHUB_REPO` and `DATASET_ROOT` in first cell
3. Runtime → Change runtime type → GPU
4. Run all cells

### On Local (Development)

```bash
git clone https://github.com/<USER>/verifoto-dl.git
cd verifoto-dl

# Edit code
vim src/utils/model.py

# Commit
git add .
git commit -m "Update model"
git push

# Then run on Colab (it will pull your changes)
```

## 📋 Common Commands

### Training

```bash
# Basic training
python -m src.train \
    --config configs/baseline.yaml \
    --run_name "2026-02-16_exp1" \
    --checkpoint_dir "/content/drive/MyDrive/verifoto_checkpoints"

# Quick test (1 epoch)
python -m src.train \
    --config configs/quick_test.yaml \
    --run_name "debug" \
    --checkpoint_dir "checkpoints"
```

### Evaluation

```bash
# Evaluate checkpoint
python -m src.eval \
    --config configs/baseline.yaml \
    --run_name "eval_t0.5" \
    --checkpoint_path "/path/to/best.pt" \
    --threshold 0.5
```

### Analysis

```bash
# Compare all runs
python scripts/compare_runs.py

# Analyze specific run with metadata (NEW!)
python scripts/analyze_results.py <run_name>

# View specific run
cat outputs/runs/<run_name>/metrics.json

# View group metrics (NEW!)
cat outputs/runs/<run_name>/group_metrics_food.csv
```

## 📁 File Locations

| What | Where | Versioned? |
|------|-------|------------|
| Code | `src/` | ✅ Yes (GitHub) |
| Configs | `configs/` | ✅ Yes (GitHub) |
| Results | `outputs/runs/` | ✅ Yes (GitHub) |
| Predictions | `outputs/runs/<run>/predictions.csv` | ✅ Yes (GitHub) |
| Group Metrics | `outputs/runs/<run>/group_metrics_*.csv` | ✅ Yes (GitHub) |
| Checkpoints | Drive or `checkpoints/` | ❌ No (too large) |
| Dataset | Drive (`augmented_v6/`) | ❌ No (too large) |

**New in augmented_v6**: Detailed metadata tracking for error analysis. See `docs/AUGMENTED_V6_DATASET.md`.

## 🔄 Typical Workflow

```
1. Edit code locally (Kiro)
   ↓
2. git commit + push
   ↓
3. Open Colab → git pull
   ↓
4. Run training (saves to Drive)
   ↓
5. Commit results from Colab
   ↓
6. git pull locally
   ↓
7. Analyze results
   ↓
8. Repeat
```

## 🎯 Key Files

- `src/train.py` - Main training script
- `src/eval.py` - Evaluation script
- `configs/baseline.yaml` - Default config
- `scripts/Verifoto_Training.ipynb` - Colab notebook
- `scripts/compare_runs.py` - Compare experiments
- `docs/WORKFLOW.md` - Detailed guide

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Dataset not found" | Update `dataset_root` in config YAML |
| "Out of memory" | Reduce `batch_size` in config |
| "No GPU" | Runtime → Change runtime type → GPU |
| "Module not found" | `!pip install -r requirements.txt` |
| "Git push failed" | Setup git credentials in Colab |

## 💡 Pro Tips

1. **Name runs clearly**: Use `YYYY-MM-DD_description` format
2. **Create new configs**: Don't edit `baseline.yaml`, copy it
3. **Commit often**: Before each Colab run
4. **Use quick_test.yaml**: For debugging (1-2 epochs)
5. **Compare runs**: Use `scripts/compare_runs.py` regularly
6. **Clean Drive**: Archive old checkpoints periodically

## 📊 Output Structure

Every run creates:

```
outputs/runs/<run_name>/
├── metrics.json       # All metrics + config
├── notes.md           # Human-readable summary
├── cm.png             # Confusion matrix
├── roc_curve.png      # ROC curve
├── pr_curve.png       # Precision-Recall curve
└── prob_dist.png      # Probability distribution
```

## 🔧 Customization

### Change Model

Edit `configs/baseline.yaml`:
```yaml
model_name: "convnext_tiny"  # or "resnet50", "efficientnet_b3", etc.
```

### Change Augmentation

Edit `src/utils/data.py`:
```python
def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        # Add/remove augmentations here
    ])
```

### Change Training Strategy

Edit config:
```yaml
epochs_head: 10        # More head-only training
epochs_finetune: 50    # More fine-tuning
lr_finetune: 0.00005   # Lower learning rate
```

## 🤖 AI Assistant Context

This project uses Kiro steering for optimal AI collaboration:
- `.kiro/steering/` - Auto-included guidelines (project context, code standards)
- `.kiro/agent/` - Working context (status, decisions) - NOT versioned
- See `.kiro/USAGE.md` for details

Agent context is centralized and lightweight - no scattered files in project root.

## 📚 More Info

- Full workflow: `docs/WORKFLOW.md`
- Colab setup: `scripts/colab_bootstrap.md`
- Differences from ChatGPT: `docs/CRITICAL_DIFFERENCES.md`

## ✅ Checklist

Before first run:

- [ ] Update `GITHUB_REPO` in notebook
- [ ] Update `DATASET_ROOT` in config
- [ ] Verify GPU is enabled in Colab
- [ ] Mount Google Drive
- [ ] Create checkpoint directory on Drive

Before each run:

- [ ] Commit and push code changes
- [ ] Pull latest code in Colab
- [ ] Choose appropriate config
- [ ] Set meaningful run name

After each run:

- [ ] Review metrics.json
- [ ] Check plots
- [ ] Commit results to GitHub
- [ ] Compare with previous runs
