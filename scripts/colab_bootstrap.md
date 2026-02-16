# Google Colab Bootstrap

Copy and paste these cells into your Colab notebook to run training.

## Cell 1: Setup

```python
# Clone repo
!git clone https://github.com/<YOUR_USERNAME>/verifoto-dl.git
%cd verifoto-dl

# Install dependencies
!pip install -q -r requirements.txt

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Check GPU
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## Cell 2: Configure Paths

```python
# Drive paths for persistent storage
DRIVE_CKPT_DIR = "/content/drive/MyDrive/verifoto_checkpoints"
DATASET_ROOT = "/content/drive/MyDrive/DatasetVerifoto/images/exp_3_augmented_v6.1"

# Create checkpoint directory
!mkdir -p {DRIVE_CKPT_DIR}

# Verify dataset exists
import os
if not os.path.exists(DATASET_ROOT):
    print(f"⚠️  Dataset not found at {DATASET_ROOT}")
    print("Please update DATASET_ROOT to match your Drive structure")
else:
    print(f"✓ Dataset found at {DATASET_ROOT}")
```

## Cell 3: Run Training

```python
import time

# Generate run name with timestamp
run_name = time.strftime("%Y-%m-%d_%H%M%S") + "_baseline"
print(f"Run name: {run_name}")

# Train
!python -m src.train \
    --config configs/baseline.yaml \
    --run_name {run_name} \
    --checkpoint_dir {DRIVE_CKPT_DIR}
```

## Cell 4: Push Results to GitHub (Optional)

```python
# Configure git (first time only)
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

# Add and commit results
!git add outputs/runs/{run_name}
!git commit -m "Add results for {run_name}"

# Push (you'll need to authenticate)
!git push
```

## Cell 5: Run Evaluation on Existing Checkpoint

```python
# Evaluate a specific checkpoint
checkpoint_path = f"{DRIVE_CKPT_DIR}/{run_name}/best.pt"
eval_run_name = f"{run_name}_eval"

!python -m src.eval \
    --config configs/baseline.yaml \
    --run_name {eval_run_name} \
    --checkpoint_path {checkpoint_path} \
    --threshold 0.5
```

## Cell 6: Threshold Sweep

```python
# Test multiple thresholds
for threshold in [0.3, 0.5, 0.7, 0.9]:
    eval_name = f"{run_name}_eval_t{threshold:.1f}"
    print(f"\n=== Evaluating at threshold {threshold} ===")
    !python -m src.eval \
        --config configs/baseline.yaml \
        --run_name {eval_name} \
        --checkpoint_path {checkpoint_path} \
        --threshold {threshold}
```

## Tips

1. **Checkpoints on Drive**: All model weights are saved to Drive and persist across sessions
2. **Results on GitHub**: Lightweight outputs (metrics, plots) can be committed to GitHub
3. **Quick iterations**: Just `git pull` in Colab to get latest code changes
4. **Multiple configs**: Create `configs/experiment2.yaml` for different setups
