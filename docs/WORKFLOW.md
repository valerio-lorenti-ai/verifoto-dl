# Complete Workflow Guide

## Overview

This guide covers the complete development cycle from local changes to Colab training to result analysis.

## Phase 1: Local Development (Kiro/PC)

### Initial Setup

```bash
# Clone the repo
git clone https://github.com/<YOUR_USERNAME>/verifoto-dl.git
cd verifoto-dl

# Install dependencies locally (optional, for testing)
pip install -r requirements.txt
```

### Making Changes

1. **Edit code** in `src/` directory
2. **Modify configs** in `configs/` directory
3. **Test locally** (optional):
   ```bash
   python scripts/quick_test.py
   ```

### Commit and Push

```bash
git add .
git commit -m "Descriptive message about changes"
git push origin main
```

## Phase 2: Training on Google Colab

### First-Time Setup

1. Open new Colab notebook
2. Go to Runtime → Change runtime type → GPU (T4 or better)
3. Copy cells from `scripts/colab_bootstrap.md`

### Cell 1: Clone and Install

```python
!git clone https://github.com/<YOUR_USERNAME>/verifoto-dl.git
%cd verifoto-dl
!pip install -q -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

### Cell 2: Configure Paths

```python
DRIVE_CKPT_DIR = "/content/drive/MyDrive/verifoto_checkpoints"
DATASET_ROOT = "/content/drive/MyDrive/DatasetVerifoto/images/exp_3_augmented_v6.1"

!mkdir -p {DRIVE_CKPT_DIR}

import os
assert os.path.exists(DATASET_ROOT), f"Dataset not found at {DATASET_ROOT}"
print(f"✓ Dataset found")
```

### Cell 3: Run Training

```python
import time

run_name = time.strftime("%Y-%m-%d_%H%M%S") + "_baseline"
print(f"Run: {run_name}")

!python -m src.train \
    --config configs/baseline.yaml \
    --run_name {run_name} \
    --checkpoint_dir {DRIVE_CKPT_DIR}
```

### Subsequent Runs (After Code Changes)

Just pull latest changes and run:

```python
# Pull latest code
%cd /content/verifoto-dl
!git pull

# Run with new code
run_name = time.strftime("%Y-%m-%d_%H%M%S") + "_experiment2"
!python -m src.train \
    --config configs/convnext_experiment.yaml \
    --run_name {run_name} \
    --checkpoint_dir {DRIVE_CKPT_DIR}
```

## Phase 3: Retrieve and Analyze Results

### Option A: Commit from Colab (Recommended for final results)

```python
# In Colab
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

!git add outputs/runs/{run_name}
!git commit -m "Results for {run_name}"
!git push
```

### Option B: Download and Commit Locally

1. In Colab, download the results:
   ```python
   from google.colab import files
   !zip -r results.zip outputs/runs/{run_name}
   files.download('results.zip')
   ```

2. On local machine:
   ```bash
   unzip results.zip
   git add outputs/runs/
   git commit -m "Add training results"
   git push
   ```

### Option C: Use Colab Files Panel

1. Navigate to `outputs/runs/<run_name>/` in Colab file browser
2. Download individual files (metrics.json, plots)
3. Add to repo locally

## Phase 4: Analysis and Iteration

### Review Results

```bash
# Pull latest results
git pull

# Check metrics
cat outputs/runs/<run_name>/metrics.json

# View plots
open outputs/runs/<run_name>/cm.png
open outputs/runs/<run_name>/pr_curve.png
```

### Compare Runs

```python
# Create a comparison script
import json
from pathlib import Path

runs_dir = Path("outputs/runs")
results = []

for run_path in runs_dir.iterdir():
    if run_path.is_dir():
        metrics_file = run_path / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                results.append({
                    "run": data["run_name"],
                    "f1": data["test_metrics"]["f1"],
                    "pr_auc": data["test_metrics"]["pr_auc"],
                    "model": data["config"]["model_name"]
                })

# Sort by PR-AUC
results.sort(key=lambda x: x["pr_auc"], reverse=True)

for r in results:
    print(f"{r['run']:30s} | F1: {r['f1']:.4f} | PR-AUC: {r['pr_auc']:.4f} | {r['model']}")
```

### Iterate

1. Analyze what worked/didn't work
2. Modify code or config locally
3. Commit and push
4. Return to Phase 2

## Common Scenarios

### Scenario 1: Quick Experiment

```bash
# Local: Create new config
cp configs/baseline.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml
git add configs/my_experiment.yaml
git commit -m "Add experiment config"
git push

# Colab: Pull and run
!git pull
!python -m src.train --config configs/my_experiment.yaml --run_name "exp1" --checkpoint_dir {DRIVE_CKPT_DIR}
```

### Scenario 2: Evaluate Existing Checkpoint

```python
# In Colab
checkpoint_path = f"{DRIVE_CKPT_DIR}/2026-02-16_baseline1/best.pt"

# Try different thresholds
for t in [0.3, 0.5, 0.7, 0.9]:
    !python -m src.eval \
        --config configs/baseline.yaml \
        --run_name "baseline1_eval_t{t}" \
        --checkpoint_path {checkpoint_path} \
        --threshold {t}
```

### Scenario 3: Debug Training Issues

```bash
# Local: Create minimal test config
# Edit configs/quick_test.yaml (already exists)
git add configs/quick_test.yaml
git commit -m "Add debug config"
git push

# Colab: Run quick test
!git pull
!python -m src.train --config configs/quick_test.yaml --run_name "debug_test" --checkpoint_dir {DRIVE_CKPT_DIR}
```

### Scenario 4: Resume from Checkpoint

Currently not implemented, but you can add this feature by:

1. Modify `src/train.py` to accept `--resume_from` argument
2. Load checkpoint and continue training
3. This is useful for long training runs

## Best Practices

1. **Naming conventions**: Use `YYYY-MM-DD_description` for run names
2. **Config versioning**: Create new config files for experiments, don't overwrite
3. **Commit often**: Commit code changes before each Colab run
4. **Document experiments**: Add notes in commit messages
5. **Clean up**: Periodically archive old checkpoints from Drive
6. **Backup**: Keep important checkpoints in multiple locations

## Troubleshooting

### "Dataset not found"

- Check `DATASET_ROOT` path in Colab
- Verify Drive is mounted
- Update path in config YAML

### "Out of memory"

- Reduce `batch_size` in config
- Use smaller model (e.g., `efficientnet_b0` instead of `convnext_tiny`)
- Reduce `img_size`

### "Git push failed"

- Set up authentication in Colab:
  ```python
  # Use personal access token
  !git remote set-url origin https://<TOKEN>@github.com/<USER>/verifoto-dl.git
  ```

### "Module not found"

- Reinstall requirements:
  ```python
  !pip install -r requirements.txt
  ```

## File Size Management

### What Goes Where

- **GitHub**: Code, configs, lightweight results (<1MB per run)
- **Google Drive**: Checkpoints (.pt files), datasets
- **Local**: Development code, analysis scripts

### Keeping GitHub Clean

```bash
# Check file sizes before committing
find outputs/runs -type f -size +1M

# If you accidentally committed large files
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/large/file' \
  --prune-empty --tag-name-filter cat -- --all
```

## Advanced: Automated Workflow

You can create a script to automate the full cycle:

```python
# scripts/auto_experiment.py
import subprocess
import time

experiments = [
    ("configs/baseline.yaml", "baseline"),
    ("configs/convnext_experiment.yaml", "convnext"),
]

for config, name in experiments:
    run_name = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{name}"
    cmd = f"python -m src.train --config {config} --run_name {run_name} --checkpoint_dir /content/drive/MyDrive/verifoto_checkpoints"
    subprocess.run(cmd, shell=True)
    time.sleep(60)  # Cool down between runs
```

This allows you to queue multiple experiments and let them run overnight.
