# Your First Run - Step by Step

Follow these exact steps for your first successful training run.

## Prerequisites

- [ ] Google account with Drive access
- [ ] Dataset uploaded to Google Drive
- [ ] GitHub account
- [ ] This repo pushed to your GitHub

## Step 1: Prepare GitHub Repo (5 minutes)

### On your local machine (Kiro):

```bash
# If you haven't already, initialize git
cd verifoto-dl
git init
git add .
git commit -m "Initial commit"

# Create GitHub repo (do this on github.com)
# Then push
git remote add origin https://github.com/<YOUR_USERNAME>/verifoto-dl.git
git branch -M main
git push -u origin main
```

## Step 2: Verify Dataset on Drive (2 minutes)

### In Google Drive:

1. Navigate to your dataset folder
2. Verify structure looks like:
   ```
   DatasetVerifoto/images/exp_3_augmented_v6.1/
   ├── images/
   │   ├── NON_FRODE/  (or real/)
   │   │   ├── img1.jpg
   │   │   ├── img2.jpg
   │   │   └── ...
   │   └── FRODE/  (or fake/)
   │       ├── img1.jpg
   │       ├── img2.jpg
   │       └── ...
   ```
3. Copy the full path (you'll need it)

## Step 3: Update Config (2 minutes)

### On your local machine:

Edit `configs/baseline.yaml`:

```yaml
dataset_root: "/content/drive/MyDrive/YOUR_ACTUAL_PATH_HERE"
```

Replace `YOUR_ACTUAL_PATH_HERE` with your dataset path from Step 2.

```bash
git add configs/baseline.yaml
git commit -m "Update dataset path"
git push
```

## Step 4: Open Colab (1 minute)

1. Go to https://colab.research.google.com/
2. File → Upload notebook
3. Upload `scripts/Verifoto_Training.ipynb` from your local repo
4. Runtime → Change runtime type → GPU (T4 or better) → Save

## Step 5: Configure Notebook (2 minutes)

In the first cell, update:

```python
GITHUB_REPO = "https://github.com/<YOUR_USERNAME>/verifoto-dl.git"
DATASET_ROOT = "/content/drive/MyDrive/YOUR_ACTUAL_PATH_HERE"
DRIVE_CKPT_DIR = "/content/drive/MyDrive/verifoto_checkpoints"
```

## Step 6: Run Setup Cells (3 minutes)

Run these cells in order:

### Cell 1: Clone and Install
```python
!git clone {GITHUB_REPO}
%cd verifoto-dl
!pip install -q -r requirements.txt
print("✓ Setup complete")
```

Wait for completion (~2 minutes).

### Cell 2: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p {DRIVE_CKPT_DIR}

import os
if os.path.exists(DATASET_ROOT):
    print(f"✓ Dataset found at {DATASET_ROOT}")
else:
    print(f"⚠️  Dataset NOT found at {DATASET_ROOT}")
```

If you see "⚠️ Dataset NOT found", go back to Step 3 and fix the path.

### Cell 3: Check GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

You should see "CUDA available: True" and a GPU name (e.g., "Tesla T4").

## Step 7: Quick Test Run (5 minutes)

Before running full training, do a quick test:

```python
import time
run_name = "quick_test_" + time.strftime("%Y%m%d_%H%M%S")

!python -m src.train \
    --config configs/quick_test.yaml \
    --run_name {run_name} \
    --checkpoint_dir {DRIVE_CKPT_DIR}
```

This runs only 1-2 epochs to verify everything works.

### Expected Output:
```
Device: cuda
GPU: Tesla T4
Dataset: /content/drive/MyDrive/...
NON_FRODE: XXXX | FRODE: YYYY
Groups: ZZZ
Split: train=AAA, val=BBB, test=CCC

=== Phase 1: head-only ===
[Head 1/1] loss=0.XXXX val_pr_auc=0.XXXX val_f1=0.XXXX

=== Phase 2: finetune-all ===
[FT 1/2] loss=0.XXXX val_pr_auc=0.XXXX val_f1=0.XXXX
[FT 2/2] loss=0.XXXX val_pr_auc=0.XXXX val_f1=0.XXXX

=== Test Evaluation ===
Test metrics: {...}

✓ Training complete. Results saved to outputs/runs/quick_test_...
```

If you see this, everything is working! 🎉

## Step 8: Full Training Run (30-60 minutes)

Now run the full training:

```python
run_name = time.strftime("%Y-%m-%d_%H%M%S") + "_baseline"
print(f"Run name: {run_name}")

!python -m src.train \
    --config configs/baseline.yaml \
    --run_name {run_name} \
    --checkpoint_dir {DRIVE_CKPT_DIR}
```

This will take 30-60 minutes depending on:
- Dataset size
- GPU type (T4 vs A100)
- Number of epochs

### What's Happening:
1. Loading and deduplicating images (~5 min)
2. Phase 1: Training head only (~5-10 min)
3. Phase 2: Fine-tuning full model (~20-40 min)
4. Test evaluation (~2 min)
5. Saving results (~1 min)

You can monitor progress in the output.

## Step 9: View Results (2 minutes)

After training completes:

```python
# Display metrics
import json
from pathlib import Path

metrics_file = Path(f"outputs/runs/{run_name}/metrics.json")
with open(metrics_file) as f:
    metrics = json.load(f)

print("Test Metrics:")
for k, v in metrics["test_metrics"].items():
    if v is not None:
        print(f"  {k}: {v:.4f}")
```

```python
# Display plots
from IPython.display import Image, display

run_dir = Path(f"outputs/runs/{run_name}")
for plot in ["cm.png", "roc_curve.png", "pr_curve.png"]:
    display(Image(filename=str(run_dir / plot)))
```

## Step 10: Save Results to GitHub (3 minutes)

```python
# Configure git (first time only)
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

# Commit results
!git add outputs/runs/{run_name}
!git commit -m "Add results for {run_name}"
!git push
```

If push fails with authentication error, you need to set up a GitHub personal access token. See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

## Step 11: Pull Results Locally (1 minute)

On your local machine:

```bash
cd verifoto-dl
git pull

# View results
cat outputs/runs/<run_name>/metrics.json

# Compare with other runs (if you have multiple)
python scripts/compare_runs.py
```

## Troubleshooting

### "Dataset not found"
- Check path in `configs/baseline.yaml`
- Verify Drive is mounted
- Make sure path starts with `/content/drive/MyDrive/`

### "CUDA available: False"
- Runtime → Change runtime type → GPU → Save
- Restart runtime

### "Out of memory"
- Edit `configs/baseline.yaml`:
  ```yaml
  batch_size: 8  # Reduce from 16
  ```
- Commit and push
- In Colab: `!git pull`
- Try again

### "Module not found"
```python
!pip install -r requirements.txt
```

### "Git push failed"
- Set up GitHub personal access token
- Or download results manually and commit locally

## Success Checklist

After your first run, you should have:

- [ ] Training completed without errors
- [ ] Results in `outputs/runs/<run_name>/`
- [ ] Checkpoint in Google Drive
- [ ] Metrics showing reasonable performance (F1 > 0.5)
- [ ] Plots generated (cm.png, roc_curve.png, etc.)
- [ ] Results committed to GitHub
- [ ] Results pulled to local machine

## What's Next?

Now that you have a working baseline:

1. **Compare with original code**: Run your old `codice_google_colab.py` and verify metrics are similar
2. **Try different configs**: Edit `configs/baseline.yaml` or create new configs
3. **Experiment with models**: Change `model_name` to "convnext_tiny" or "resnet50"
4. **Tune thresholds**: Use `src/eval.py` to test different classification thresholds
5. **Read full docs**: Check `docs/WORKFLOW.md` for advanced usage

## Time Breakdown

- Setup (Steps 1-6): ~15 minutes
- Quick test (Step 7): ~5 minutes
- Full training (Step 8): ~30-60 minutes
- Results (Steps 9-11): ~6 minutes

Total: ~1 hour for first complete run

## Tips for Second Run

Next time it's much faster:

```python
# In Colab (assuming you kept the notebook open)
%cd /content/verifoto-dl
!git pull  # Get latest code changes

run_name = time.strftime("%Y-%m-%d_%H%M%S") + "_experiment2"
!python -m src.train --config configs/baseline.yaml --run_name {run_name} --checkpoint_dir {DRIVE_CKPT_DIR}
```

That's it! No need to re-clone or reinstall.

## Getting Help

If you're stuck:
1. Check error message carefully
2. Review this guide again
3. Check `QUICKSTART.md` for common commands
4. Read `docs/WORKFLOW.md` for detailed explanations

Good luck! 🚀
