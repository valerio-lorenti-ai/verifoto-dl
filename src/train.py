import os
import sys
import time
import json
import random
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.data import (
    find_class_dirs, list_images_in_dir, compute_hashes,
    group_near_duplicates, label_of_path, stratified_group_split,
    build_transforms, ImageBinaryDataset
)
from utils.model import build_model, set_backbone_trainable
from utils.metrics import predict_proba, compute_metrics_from_probs, EarlyStopping
from utils.visualization import plot_prob_distributions, plot_roc_pr, plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"


def train_one_epoch(model, loader, optimizer, criterion, scheduler=None, max_grad_norm=1.0, device="cuda"):
    model.train()
    losses = []
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan


@torch.no_grad()
def validate(model, loader, threshold=0.5, device="cuda"):
    probs, y_true, _ = predict_proba(model, loader, device)
    return compute_metrics_from_probs(probs, y_true, threshold=threshold)


def save_checkpoint(model, path: Path, best_metric: float = None, cfg: dict = None):
    payload = {
        "state_dict": model.state_dict(),
        "best_metric": float(best_metric) if best_metric is not None else None,
        "cfg": cfg if cfg is not None else None,
    }
    torch.save(payload, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run_name", type=str, required=True, help="Run name (e.g., 2026-02-16_baseline1)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint dir (for Drive)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    set_seed(config.get('seed', 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    dataset_root = config['dataset_root']
    checkpoint_dir = args.checkpoint_dir or config.get('checkpoint_dir', 'checkpoints')
    output_dir = Path(config.get('output_dir', 'outputs/runs')) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = Path(checkpoint_dir) / args.run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_root}")
    print(f"Output: {output_dir}")
    print(f"Checkpoints: {checkpoint_path}")

    # Load data
    print("\n=== Loading dataset ===")
    non_dir, frode_dir = find_class_dirs(dataset_root)
    if non_dir is None or frode_dir is None:
        raise RuntimeError("Cannot find NON_FRODE and FRODE directories")

    non_paths = list_images_in_dir(non_dir)
    fro_paths = list_images_in_dir(frode_dir)
    print(f"NON_FRODE: {len(non_paths)} | FRODE: {len(fro_paths)}")

    # Deduplication
    all_paths = non_paths + fro_paths
    hashes = compute_hashes(all_paths, hash_size=8)
    groups = group_near_duplicates(all_paths, hashes, max_hamming=4)

    group_items = []
    for gid, paths in groups.items():
        ys = [label_of_path(p, non_dir, frode_dir) for p in paths]
        y = int(np.round(np.mean(ys)))
        group_items.append((gid, y, paths))

    print(f"Groups: {len(group_items)}")

    # Split
    (train_paths, train_y, _), (val_paths, val_y, _), (test_paths, test_y, _) = stratified_group_split(
        group_items, 0.70, 0.15, 0.15, seed=config.get('seed', 42)
    )
    print(f"Split: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    # Datasets
    img_size = config.get('img_size', 224)
    batch_size = config.get('batch_size', 16)
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = ImageBinaryDataset(train_paths, train_y, transform=train_tf, img_size=img_size)
    val_ds = ImageBinaryDataset(val_paths, val_y, transform=eval_tf, img_size=img_size)
    test_ds = ImageBinaryDataset(test_paths, test_y, transform=eval_tf, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model_name = config.get('model_name', 'efficientnet_b0')
    drop_rate = config.get('drop_rate', 0.2)
    model = build_model(model_name, pretrained=True, drop_rate=drop_rate).to(device)

    # Loss
    train_pos = sum(train_y)
    train_neg = len(train_y) - train_pos
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"pos_weight: {pos_weight.item():.2f}")

    # Training config
    epochs_head = config.get('epochs_head', 5)
    epochs_finetune = config.get('epochs_finetune', 25)
    lr_head = config.get('lr_head', 3e-4)
    lr_finetune = config.get('lr_finetune', 1e-4)
    weight_decay = config.get('weight_decay', 1e-3)
    patience = config.get('patience', 6)
    monitor = config.get('monitor', 'pr_auc')
    max_grad_norm = config.get('max_grad_norm', 1.0)

    history = []
    best_metric = -1e9
    best_ckpt_path = checkpoint_path / "best.pt"

    # Phase 1: Head only
    print("\n=== Phase 1: Head-only ===")
    set_backbone_trainable(model, trainable=False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr_head, weight_decay=weight_decay)
    total_steps = max(epochs_head * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    es = EarlyStopping(patience=patience, min_delta=1e-4, mode="max")

    for epoch in range(1, epochs_head + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, max_grad_norm, device)
        val_m = validate(model, val_loader, threshold=0.5, device=device)
        monitor_val = val_m[monitor]
        stop, improved = es.step(monitor_val)

        history.append({"phase": "head", "epoch": epoch, "train_loss": tr_loss, **val_m})
        print(f"[Head {epoch}/{epochs_head}] loss={tr_loss:.4f} val_{monitor}={monitor_val:.4f} val_f1={val_m['f1']:.4f}")

        if improved and monitor_val > best_metric:
            best_metric = monitor_val
            save_checkpoint(model, best_ckpt_path, best_metric=best_metric, cfg=config)

        if stop:
            print("Early stopping (head)")
            break

    # Phase 2: Finetune
    print("\n=== Phase 2: Finetune ===")
    set_backbone_trainable(model, trainable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_finetune, weight_decay=weight_decay)
    total_steps = max(epochs_finetune * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    es = EarlyStopping(patience=patience, min_delta=1e-4, mode="max")

    for epoch in range(1, epochs_finetune + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, max_grad_norm, device)
        val_m = validate(model, val_loader, threshold=0.5, device=device)
        monitor_val = val_m[monitor]
        stop, improved = es.step(monitor_val)

        history.append({"phase": "finetune", "epoch": epoch, "train_loss": tr_loss, **val_m})
        print(f"[FT {epoch}/{epochs_finetune}] loss={tr_loss:.4f} val_{monitor}={monitor_val:.4f} val_f1={val_m['f1']:.4f}")

        if improved and monitor_val > best_metric:
            best_metric = monitor_val
            save_checkpoint(model, best_ckpt_path, best_metric=best_metric, cfg=config)

        if stop:
            print("Early stopping (finetune)")
            break

    # Load best and evaluate on test
    print("\n=== Test Evaluation ===")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_probs, test_true, test_paths = predict_proba(model, test_loader, device)
    test_metrics = compute_metrics_from_probs(test_probs, test_true, threshold=0.5)
    test_cm = confusion_matrix(test_true, (test_probs >= 0.5).astype(int), labels=[0, 1])

    print(f"Test metrics: {test_metrics}")

    # Save outputs
    plot_prob_distributions(test_probs, test_true, save_path=output_dir / "prob_dist.png")
    plot_roc_pr(test_probs, test_true, save_path_roc=output_dir / "roc_curve.png", save_path_pr=output_dir / "pr_curve.png")
    plot_confusion_matrix(test_cm, save_path=output_dir / "cm.png")

    # Save metrics JSON
    metrics_output = {
        "run_name": args.run_name,
        "git_commit": get_git_commit(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": 0.5,
        "test_metrics": {k: float(v) if not np.isnan(v) else None for k, v in test_metrics.items()},
        "confusion_matrix": test_cm.tolist(),
        "config": config
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics_output, f, indent=2)

    # Save notes
    notes = f"""# Run: {args.run_name}

## Config
- Model: {model_name}
- Image size: {img_size}
- Batch size: {batch_size}
- Epochs: {epochs_head} (head) + {epochs_finetune} (finetune)

## Results (threshold=0.5)
- Accuracy: {test_metrics['acc']:.4f}
- Precision: {test_metrics['prec']:.4f}
- Recall: {test_metrics['rec']:.4f}
- F1: {test_metrics['f1']:.4f}
- PR-AUC: {test_metrics['pr_auc']:.4f}
- ROC-AUC: {test_metrics['roc_auc']:.4f}

## Confusion Matrix
```
{test_cm}
```

Git commit: {get_git_commit()}
"""

    with open(output_dir / "notes.md", 'w') as f:
        f.write(notes)

    print(f"\n✓ Training complete. Results saved to {output_dir}")
    print(f"✓ Best checkpoint saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()
