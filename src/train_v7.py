"""
Training script V7 con ottimizzazioni avanzate:
- ConvNeXt-Tiny
- Focal Loss / Weighted Focal Loss
- Augmentation differenziata per real vs generated
- Threshold cost-sensitive
- Più peso su errori su immagini reali
"""

import os
import sys
import time
import json
import random
import argparse
import subprocess
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.data import (
    parse_augmented_v6_dataset, group_based_split_v6, domain_aware_group_split_v1,
    build_transforms, ImageBinaryDataset
)
from src.utils.model import build_model, set_backbone_trainable
from src.utils.losses import build_loss_function
from src.utils.metrics import (
    predict_proba, compute_metrics_from_probs, compute_group_metrics,
    get_top_errors, EarlyStopping, find_optimal_threshold,
    find_cost_sensitive_threshold, find_threshold_with_max_fp_rate
)
from src.utils.visualization import plot_prob_distributions, plot_roc_pr, plot_confusion_matrix
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


def collate_with_metadata(batch):
    """Custom collate function per gestire metadati con valori None."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    
    metadata = {}
    if len(batch) > 0 and len(batch[0]) > 2:
        meta_keys = batch[0][2].keys()
        for key in meta_keys:
            metadata[key] = [item[2][key] for item in batch]
    
    return images, labels, metadata


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
    parser.add_argument("--run_name", type=str, required=True, help="Run name")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint dir")
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
    df = parse_augmented_v6_dataset(dataset_root)
    
    # Split strategy
    split_strategy = config.get('split_strategy', 'group_v6')
    split_include_food = config.get('split_include_food', False)
    
    if split_strategy == 'domain_aware':
        print("\n⚠️  Using DOMAIN-AWARE GROUP-BASED split")
        print("   (prevents data leakage + balances source/generator across splits)")
        train_df, val_df, test_df = domain_aware_group_split_v1(
            df, 0.70, 0.15, 0.15, seed=config.get('seed', 42), include_food=split_include_food
        )
    else:
        print("\n⚠️  Using GROUP-BASED split to prevent data leakage")
        print("   (photos with multiple versions stay in same split)")
        train_df, val_df, test_df = group_based_split_v6(
            df, 0.70, 0.15, 0.15, seed=config.get('seed', 42)
        )
    
    # Datasets with differential augmentation
    img_size = config.get('img_size', 224)
    batch_size = config.get('batch_size', 16)
    aug_strength = config.get('augmentation_strength', 'normal')
    
    train_tf, eval_tf = build_transforms(img_size, augmentation_strength=aug_strength)
    
    # Augmentation più forte per immagini reali (opzionale)
    use_differential_aug = config.get('real_augmentation_multiplier', 1.0) > 1.0
    real_train_tf = None
    if use_differential_aug:
        real_train_tf, _ = build_transforms(img_size, augmentation_strength='strong')
        print(f"✓ Using differential augmentation (stronger for real images)")

    train_ds = ImageBinaryDataset(train_df, transform=train_tf, img_size=img_size,
                                   real_transform=real_train_tf, use_differential_aug=use_differential_aug)
    val_ds = ImageBinaryDataset(val_df, transform=eval_tf, img_size=img_size)
    test_ds = ImageBinaryDataset(test_df, transform=eval_tf, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, 
                               pin_memory=True, collate_fn=collate_with_metadata)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, 
                            pin_memory=True, collate_fn=collate_with_metadata)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, 
                             pin_memory=True, collate_fn=collate_with_metadata)

    # Model
    model_name = config.get('model_name', 'efficientnet_b0')
    drop_rate = config.get('drop_rate', 0.2)
    model = build_model(model_name, pretrained=True, drop_rate=drop_rate).to(device)
    
    print(f"\n=== Model: {model_name} ===")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Loss function
    loss_type = config.get('loss_type', 'bce')
    train_pos = (train_df['label'] == 1).sum()
    train_neg = (train_df['label'] == 0).sum()
    pos_weight = train_neg / max(train_pos, 1)
    
    loss_kwargs = {
        'pos_weight': pos_weight,
        'focal_alpha': config.get('focal_alpha', 0.25),
        'focal_gamma': config.get('focal_gamma', 2.0),
        'real_weight': config.get('real_weight', 2.0),
        'fp_cost': config.get('fp_cost', 2.0),
        'fn_cost': config.get('fn_cost', 1.0),
    }
    
    criterion = build_loss_function(loss_type, **loss_kwargs)
    print(f"\n=== Loss: {loss_type} ===")
    print(f"pos_weight: {pos_weight:.2f}")
    if loss_type in ['focal', 'weighted_focal']:
        print(f"focal_alpha: {loss_kwargs['focal_alpha']}")
        print(f"focal_gamma: {loss_kwargs['focal_gamma']}")
    if loss_type == 'weighted_focal':
        print(f"real_weight: {loss_kwargs['real_weight']} (penalizza {loss_kwargs['real_weight']}x errori su real)")
    if loss_type == 'cost_sensitive':
        print(f"fp_cost: {loss_kwargs['fp_cost']}, fn_cost: {loss_kwargs['fn_cost']}")

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

    # Load best and evaluate
    print("\n=== Test Evaluation ===")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Find optimal threshold on validation
    print("\n=== Finding Optimal Threshold on Validation Set ===")
    val_probs, val_true, val_metadata = predict_proba(model, val_loader, device)
    
    threshold_strategy = config.get('threshold_strategy', 'f1')
    
    if threshold_strategy == 'cost_sensitive':
        # Cost-sensitive threshold
        fp_cost = config.get('fp_cost', 2.0)
        fn_cost = config.get('fn_cost', 1.0)
        optimal_threshold, best_cost, threshold_info = find_cost_sensitive_threshold(
            val_probs, val_true, fp_cost=fp_cost, fn_cost=fn_cost
        )
        print(f"Optimal threshold (cost-sensitive): {optimal_threshold:.3f} (cost={best_cost:.1f})")
        print(f"  FP cost: {fp_cost}, FN cost: {fn_cost}")
    
    elif threshold_strategy == 'max_fp_rate':
        # Max FP rate constraint
        max_fp_rate = config.get('max_fp_rate', 0.10)
        optimal_threshold, best_f1, threshold_info = find_threshold_with_max_fp_rate(
            val_probs, val_true, max_fp_rate=max_fp_rate
        )
        print(f"Optimal threshold (max FP rate {max_fp_rate:.1%}): {optimal_threshold:.3f} (F1={best_f1:.4f})")
    
    else:
        # Standard F1 optimization
        optimal_threshold, best_f1, threshold_info = find_optimal_threshold(
            val_probs, val_true, metric='f1'
        )
        print(f"Optimal threshold (F1): {optimal_threshold:.3f} (F1={best_f1:.4f})")
    
    # Save validation predictions
    val_predictions_df = pd.DataFrame({
        'path': [m['path'] for m in val_metadata],
        'y_true': val_true,
        'y_prob': val_probs,
        'y_pred': (val_probs >= optimal_threshold).astype(int),
        'source': [m.get('source') for m in val_metadata],
        'quality': [m.get('quality') for m in val_metadata],
        'food_category': [m.get('food_category') for m in val_metadata],
        'defect_type': [m.get('defect_type') for m in val_metadata],
        'generator': [m.get('generator') for m in val_metadata]
    })
    val_predictions_df.to_csv(output_dir / "validation_predictions.csv", index=False)
    
    # Save validation logits for calibration
    val_probs_clipped = np.clip(val_probs, 1e-7, 1 - 1e-7)
    val_logits = np.log(val_probs_clipped / (1 - val_probs_clipped))
    np.save(output_dir / "validation_logits.npy", val_logits)
    
    # Test evaluation
    test_threshold = optimal_threshold
    print(f"\nUsing threshold={test_threshold:.3f} for test evaluation")

    test_probs, test_true, test_metadata = predict_proba(model, test_loader, device)
    test_metrics = compute_metrics_from_probs(test_probs, test_true, threshold=test_threshold)
    test_cm = confusion_matrix(test_true, (test_probs >= test_threshold).astype(int), labels=[0, 1])

    print(f"Test metrics: {test_metrics}")

    # Save predictions and metrics
    predictions_df = pd.DataFrame({
        'path': [m['path'] for m in test_metadata],
        'y_true': test_true,
        'y_prob': test_probs,
        'y_pred': (test_probs >= test_threshold).astype(int),
        'source': [m.get('source') for m in test_metadata],
        'quality': [m.get('quality') for m in test_metadata],
        'food_category': [m.get('food_category') for m in test_metadata],
        'defect_type': [m.get('defect_type') for m in test_metadata],
        'generator': [m.get('generator') for m in test_metadata]
    })
    
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Group metrics
    print("\n=== Computing group metrics ===")
    for group_col in ['food_category', 'defect_type', 'generator', 'quality']:
        group_metrics = compute_group_metrics(predictions_df, group_col, threshold=test_threshold)
        group_metrics.to_csv(output_dir / f"group_metrics_{group_col}.csv", index=False)
        print(f"✓ Saved group_metrics_{group_col}.csv")
    
    # Top errors
    top_fp = get_top_errors(predictions_df, error_type='fp', top_n=50)
    top_fp.to_csv(output_dir / "top_false_positives.csv", index=False)
    
    top_fn = get_top_errors(predictions_df, error_type='fn', top_n=50)
    top_fn.to_csv(output_dir / "top_false_negatives.csv", index=False)

    # Visualizations
    plot_prob_distributions(test_probs, test_true, save_path=output_dir / "prob_dist.png")
    plot_roc_pr(test_probs, test_true, save_path_roc=output_dir / "roc_curve.png", 
                save_path_pr=output_dir / "pr_curve.png")
    plot_confusion_matrix(test_cm, save_path=output_dir / "cm.png")

    # Save metrics JSON
    metrics_output = {
        "run_name": args.run_name,
        "git_commit": get_git_commit(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": float(test_threshold),
        "threshold_strategy": threshold_strategy,
        "threshold_info": threshold_info,
        "test_metrics": {k: float(v) if not np.isnan(v) else None for k, v in test_metrics.items()},
        "confusion_matrix": test_cm.tolist(),
        "config": config
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics_output, f, indent=2)

    # Save notes
    notes = f"""# Run: {args.run_name}

## Config
- Model: {model_name} ({n_params/1e6:.1f}M params)
- Image size: {img_size}
- Batch size: {batch_size}
- Epochs: {epochs_head} (head) + {epochs_finetune} (finetune)
- Loss: {loss_type}
- Augmentation: {aug_strength}
- Differential augmentation: {use_differential_aug}

## Threshold Selection
- Strategy: {threshold_strategy}
- Optimal threshold: {test_threshold:.3f}
- Validation score: {best_f1 if threshold_strategy != 'cost_sensitive' else best_cost:.4f}

## Results (threshold={test_threshold:.3f})
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
