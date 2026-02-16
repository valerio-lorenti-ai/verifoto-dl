# Migration from Original Colab Code

This guide helps you understand how your original `codice_google_colab.py` maps to the new structure.

## Code Mapping

### Original → New Structure

| Original Code Section | New Location | Notes |
|----------------------|--------------|-------|
| Imports | `src/utils/*.py` | Split by functionality |
| `set_seed()` | `src/train.py`, `src/eval.py` | Called at start of each script |
| Dataset finding | `src/utils/data.py` | `find_class_dirs()`, `list_images_in_dir()` |
| Hashing & deduplication | `src/utils/data.py` | `compute_hashes()`, `group_near_duplicates()` |
| Splitting | `src/utils/data.py` | `stratified_group_split()` |
| Augmentations | `src/utils/data.py` | `RandomJPEGCompression`, `RandomGaussianNoise`, `build_transforms()` |
| Dataset class | `src/utils/data.py` | `ImageBinaryDataset` |
| Model building | `src/utils/model.py` | `build_model()`, `set_backbone_trainable()` |
| Training loop | `src/train.py` | `train_one_epoch()`, `fit()` |
| Metrics | `src/utils/metrics.py` | `predict_proba()`, `compute_metrics_from_probs()` |
| Early stopping | `src/utils/metrics.py` | `EarlyStopping` class |
| Visualization | `src/utils/visualization.py` | `plot_*()` functions |
| Evaluation | `src/eval.py` | Separate script for checkpoint evaluation |
| Config | `configs/*.yaml` | Hyperparameters moved to YAML |

## Key Changes

### 1. Configuration

**Before (hardcoded):**
```python
IMG_SIZE = 224
BATCH_SIZE = 16
MODEL_NAME = "efficientnet_b0"
```

**After (YAML config):**
```yaml
# configs/baseline.yaml
img_size: 224
batch_size: 16
model_name: "efficientnet_b0"
```

### 2. Dataset Path

**Before:**
```python
DATASET_ROOT = "/content/drive/MyDrive/DatasetVerifoto/images/exp_3_augmented_v6.1"
```

**After:**
```yaml
# In config YAML
dataset_root: "/content/drive/MyDrive/DatasetVerifoto/images/exp_3_augmented_v6.1"
```

### 3. Running Training

**Before (Colab cells):**
```python
# Cell 1: imports and setup
# Cell 2: load data
# Cell 3: create model
# Cell 4: train
# Cell 5: evaluate
```

**After (single command):**
```bash
python -m src.train \
    --config configs/baseline.yaml \
    --run_name "2026-02-16_baseline" \
    --checkpoint_dir "/content/drive/MyDrive/verifoto_checkpoints"
```

### 4. Output Location

**Before:**
```python
OUT_DIR = Path("/content/verifoto_runs")
RUN_DIR = OUT_DIR / time.strftime("%Y%m%d-%H%M%S")
```

**After:**
```
outputs/runs/<run_name>/
```

### 5. Checkpoint Saving

**Before:**
```python
best_path = RUN_DIR / "best.pt"
torch.save(payload, str(best_path))
```

**After:**
```python
# Saved to Drive
checkpoint_path = Path(checkpoint_dir) / run_name / "best.pt"
```

## What Stayed the Same

✅ **Deduplication logic** - Identical implementation
✅ **Group-aware splitting** - Same algorithm
✅ **Custom augmentations** - JPEG compression and noise
✅ **Two-phase training** - Head-only then full finetune
✅ **Early stopping** - Same logic
✅ **Metrics** - Same calculations
✅ **Pos_weight** - Same imbalance handling

## What's New

### 1. Modular Code

Code is split into logical modules instead of one long script.

### 2. CLI Interface

```bash
# Training
python -m src.train --config configs/baseline.yaml --run_name "exp1"

# Evaluation
python -m src.eval --config configs/baseline.yaml --run_name "eval1" --checkpoint_path "path/to/best.pt"
```

### 3. Multiple Configs

Create different configs for different experiments:
- `configs/baseline.yaml`
- `configs/convnext_experiment.yaml`
- `configs/quick_test.yaml`

### 4. Structured Output

Every run creates:
- `metrics.json` - Machine-readable metrics
- `notes.md` - Human-readable summary
- Plots (PNG files)

### 5. Git Integration

- Tracks git commit in metrics
- Results can be versioned on GitHub
- Checkpoints stay on Drive

### 6. Comparison Tools

```bash
python scripts/compare_runs.py
```

Shows all runs side-by-side.

## Migration Steps

### Step 1: Verify Original Code Works

Keep your original `codice_google_colab.py` as backup. Make sure it runs successfully in Colab.

### Step 2: Test New Structure

```bash
# In Colab
!git clone https://github.com/<USER>/verifoto-dl.git
%cd verifoto-dl
!pip install -r requirements.txt

# Quick test
!python scripts/quick_test.py
```

### Step 3: Run First Training

```bash
!python -m src.train \
    --config configs/quick_test.yaml \
    --run_name "migration_test" \
    --checkpoint_dir "/content/drive/MyDrive/verifoto_checkpoints"
```

This uses minimal epochs to verify everything works.

### Step 4: Compare Results

Run both old and new code on same data. Verify metrics are similar (small differences due to randomness are OK).

### Step 5: Full Migration

Once verified, use new structure for all future experiments.

## Troubleshooting

### "Different results than original"

- Check seed is same (42)
- Verify dataset path is correct
- Ensure same augmentation parameters
- Small differences (<1%) are normal due to GPU randomness

### "Import errors"

```bash
!pip install -r requirements.txt
```

### "Can't find dataset"

Update `dataset_root` in config YAML to match your Drive structure.

### "Slower than original"

- Check `batch_size` in config
- Verify GPU is enabled
- Try `NUM_WORKERS=2` in DataLoader (edit `src/train.py`)

## Advantages of New Structure

1. **Reproducibility**: Git commit tracking, structured configs
2. **Flexibility**: Easy to try different models/hyperparameters
3. **Maintainability**: Modular code is easier to debug
4. **Collaboration**: Others can understand and modify code
5. **Iteration speed**: Just `git pull` to get latest changes
6. **Analysis**: Compare multiple runs easily

## When to Use Original vs New

### Use Original If:
- Quick one-off experiment
- Prototyping new ideas
- Don't need version control

### Use New Structure If:
- Production pipeline
- Multiple experiments
- Team collaboration
- Need reproducibility
- Want to track results over time

## Gradual Migration

You don't have to migrate everything at once:

1. **Week 1**: Use new structure for new experiments, keep old for existing work
2. **Week 2**: Migrate configs to YAML
3. **Week 3**: Start using git for version control
4. **Week 4**: Fully migrate to new structure

## Getting Help

If you encounter issues:

1. Check `docs/WORKFLOW.md` for detailed instructions
2. Review `QUICKSTART.md` for common commands
3. Look at `scripts/Verifoto_Training.ipynb` for working example
4. Compare with original `codice_google_colab.py` to see what changed

## Rollback Plan

If new structure doesn't work:

1. Your original code is still in `codice_google_colab.py`
2. Just use that in Colab as before
3. No data or checkpoints are affected
4. You can try migration again later

The new structure is designed to coexist with your original workflow, so there's no risk in trying it.
