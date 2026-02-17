# Training Results Directory

This directory contains results from training runs.

## What Gets Committed to Git

### ✅ Tracked Files (Committed)
- `metrics.json` - Complete metrics and configuration
- `predictions.csv` - Model predictions with metadata
- `group_metrics_*.csv` - Performance by group (food, defect, generator, quality)
- `top_false_*.csv` - Top false positives and false negatives
- `notes.md` - Run summary and notes
- `CRITICAL_ANALYSIS.md` - Detailed analysis (if present)
- `*.png`, `*.jpg`, `*.jpeg` - Visualizations (confusion matrix, ROC, PR curves)

### ❌ Ignored Files (Not Committed)
- `*.pt`, `*.pth`, `*.ckpt` - Model checkpoints (stored on Google Drive)
- `*.h5`, `*.pkl` - Other model formats
- `*.npy`, `*.npz` - Large numpy arrays

## Directory Structure

```
outputs/runs/
├── 2026-02-16_noK/
│   ├── metrics.json              ✅ Committed
│   ├── predictions.csv           ✅ Committed
│   ├── group_metrics_food.csv    ✅ Committed
│   ├── group_metrics_defect.csv  ✅ Committed
│   ├── top_false_positives.csv   ✅ Committed
│   ├── cm.png                    ✅ Committed
│   ├── roc_curve.png             ✅ Committed
│   └── notes.md                  ✅ Committed
├── 2026-02-17_no_leakage_v1/
│   └── ...
└── README.md                     ✅ This file
```

## Pushing Results from Colab

After training on Colab, push results to GitHub:

```bash
# In Colab
%cd /content/verifoto-dl

# Configure git
!git config user.email "colab@verifoto.ai"
!git config user.name "Colab Training"

# Add results (force to override .gitignore for outputs/)
!git add -f outputs/runs/{EXPERIMENT_NAME}

# Commit
!git commit -m "Add training results: {EXPERIMENT_NAME}"

# Push (use your GitHub token)
!git push https://{GITHUB_TOKEN}@github.com/valerio-lorenti-ai/verifoto-dl.git main
```

## Storage Strategy

- **Git**: Metrics, CSVs, visualizations (small files)
- **Google Drive**: Model checkpoints (large files)
- **Local**: Full results for analysis

## File Sizes

Typical sizes per run:
- `metrics.json`: ~5 KB
- `predictions.csv`: ~50-200 KB
- `group_metrics_*.csv`: ~5-20 KB each
- `top_false_*.csv`: ~10-50 KB each
- `*.png`: ~50-500 KB each
- `notes.md`: ~2-5 KB

**Total per run**: ~500 KB - 2 MB (acceptable for git)

## Best Practices

1. **Always commit results** after training
2. **Use descriptive run names** (e.g., `2026-02-17_no_leakage_v1`)
3. **Add CRITICAL_ANALYSIS.md** for important findings
4. **Keep checkpoints on Drive** (too large for git)
5. **Pull before training** to get latest code
6. **Push after training** to share results

## Cleanup

To remove old runs locally (but keep on git):

```bash
# Remove specific run
rm -rf outputs/runs/2026-02-16_old_run

# Remove all runs except latest
ls -t outputs/runs/ | tail -n +2 | xargs -I {} rm -rf outputs/runs/{}
```

To remove from git history (if accidentally committed large files):

```bash
# Use git filter-branch or BFG Repo-Cleaner
# See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
```
