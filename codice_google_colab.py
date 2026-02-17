"""
Verifoto Google Colab - Code Cells Reference
============================================

This file contains the Python code cells used in the Colab notebooks.
For the actual notebooks, see:
- scripts/Verifoto_Training_V2.ipynb (main training)
- scripts/Verifoto_Recovery.ipynb (session recovery)

For detailed documentation, see: docs/COLAB_WORKFLOW.md
"""

# ==============================================================================
# TRAINING NOTEBOOK - Main Cells
# ==============================================================================

# --- CELL 1: Experiment Configuration ---
EXPERIMENT_NAME = "2026-02-17_baseline_test"
DATASET_NAME = "exp_3_augmented_v6.2_noK"
CONFIG_FILE = "baseline.yaml"
GITHUB_TOKEN = ""

DATASET_ROOT = f"/content/drive/MyDrive/DatasetVerifoto/images/{DATASET_NAME}"
CHECKPOINT_DIR = "/content/drive/MyDrive/verifoto_checkpoints"
BACKUP_DIR = f"/content/drive/MyDrive/verifoto_results/{EXPERIMENT_NAME}"
OUTPUT_DIR = f"outputs/runs/{EXPERIMENT_NAME}"
CONFIG_PATH = f"configs/{CONFIG_FILE}"

# --- CELL 2: Check GPU ---
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# --- CELL 3: Setup Environment ---
# %cd /content
# !git clone https://github.com/valerio-lorenti-ai/verifoto-dl.git
# %cd verifoto-dl
# !git pull
# !pip install -q -r requirements.txt

# --- CELL 4: Mount Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- CELL 5: Verify Dataset ---
import os
from pathlib import Path

print(f"Dataset exists: {os.path.exists(DATASET_ROOT)}")
if os.path.exists(DATASET_ROOT):
    has_originali = os.path.exists(os.path.join(DATASET_ROOT, "originali"))
    has_modificate = os.path.exists(os.path.join(DATASET_ROOT, "modificate"))
    print(f"✓ originali/: {has_originali}")
    print(f"✓ modificate/: {has_modificate}")

# --- CELL 6: Update Config ---
import yaml

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
config['dataset_root'] = DATASET_ROOT
with open(CONFIG_PATH, 'w') as f:
    yaml.dump(config, f)
print("✓ Config updated")

# --- CELL 7: Train ---
# !python -m src.train \
#   --config {CONFIG_PATH} \
#   --run_name {EXPERIMENT_NAME} \
#   --checkpoint_dir {CHECKPOINT_DIR}

# --- CELL 8: Analyze Results ---
# !python scripts/analyze_results.py {EXPERIMENT_NAME}

# --- CELL 9: Display Results Inline ---
import json
import pandas as pd
from IPython.display import Image, display

results_dir = Path(OUTPUT_DIR)

with open(results_dir / "metrics.json") as f:
    metrics = json.load(f)

print("="*80)
print("TEST METRICS")
print("="*80)
for k, v in metrics['test_metrics'].items():
    if v is not None:
        print(f"{k:>15}: {v:.4f}")

# Display confusion matrix image
display(Image(filename=str(results_dir / "cm.png")))

# --- CELL 10: Backup to Drive ---
# !mkdir -p {BACKUP_DIR}
# !cp -r {OUTPUT_DIR}/* {BACKUP_DIR}/

# --- CELL 11: Push to GitHub ---
from getpass import getpass

if not GITHUB_TOKEN:
    GITHUB_TOKEN = getpass("Enter GitHub token (or press Enter to skip): ")

if GITHUB_TOKEN:
    # %cd /content/verifoto-dl
    # !git config user.email "colab@verifoto.ai"
    # !git config user.name "Colab Training"
    # !git add -f {OUTPUT_DIR}
    # !git commit -m "Add training results: {EXPERIMENT_NAME}"
    repo_url = f"https://{GITHUB_TOKEN}@github.com/valerio-lorenti-ai/verifoto-dl.git"
    # !git push {repo_url} main


# ==============================================================================
# RECOVERY NOTEBOOK - Main Cells
# ==============================================================================

# --- CELL 1: Recovery Configuration ---
ORIGINAL_RUN_NAME = "2026-02-16_noK"
RECOVERY_RUN_NAME = "2026-02-16_noK_recovered"
DATASET_NAME = "exp_3_augmented_v6.2_noK"
CONFIG_FILE = "baseline.yaml"
THRESHOLD = 0.5

DATASET_ROOT = f"/content/drive/MyDrive/DatasetVerifoto/images/{DATASET_NAME}"
CHECKPOINT_PATH = f"/content/drive/MyDrive/verifoto_checkpoints/{ORIGINAL_RUN_NAME}/best.pt"
OUTPUT_DIR = f"outputs/runs/{RECOVERY_RUN_NAME}"
CONFIG_PATH = f"configs/{CONFIG_FILE}"

# --- CELL 2: Setup (same as training) ---
# (Clone repo, install deps, mount drive)

# --- CELL 3: Verify Checkpoint ---
if os.path.exists(CHECKPOINT_PATH):
    size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
    print(f"✓ Checkpoint found: {size_mb:.2f} MB")
else:
    print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")

# --- CELL 4: Regenerate Results ---
# !python -m src.eval \
#   --config {CONFIG_PATH} \
#   --run_name {RECOVERY_RUN_NAME} \
#   --checkpoint_path {CHECKPOINT_PATH} \
#   --threshold {THRESHOLD}

# --- CELL 5: Analysis and Push (same as training) ---
# (Analyze, backup, push to GitHub)
