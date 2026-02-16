import os
import sys
import time
import json
import random
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data import (
    find_class_dirs, list_images_in_dir, compute_hashes,
    group_near_duplicates, label_of_path, stratified_group_split,
    build_transforms, ImageBinaryDataset
)
from utils.model import build_model
from utils.metrics import predict_proba, compute_metrics_from_probs
from utils.visualization import plot_prob_distributions, plot_roc_pr, plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run_name", type=str, required=True, help="Run name for output")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Paths
    dataset_root = config['dataset_root']
    output_dir = Path(config.get('output_dir', 'outputs/runs')) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_root}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output: {output_dir}")

    # Load data
    print("\n=== Loading dataset ===")
    non_dir, frode_dir = find_class_dirs(dataset_root)
    if non_dir is None or frode_dir is None:
        raise RuntimeError("Cannot find NON_FRODE and FRODE directories")

    non_paths = list_images_in_dir(non_dir)
    fro_paths = list_images_in_dir(frode_dir)
    print(f"NON_FRODE: {len(non_paths)} | FRODE: {len(fro_paths)}")

    all_paths = non_paths + fro_paths
    hashes = compute_hashes(all_paths, hash_size=8)
    groups = group_near_duplicates(all_paths, hashes, max_hamming=4)

    group_items = []
    for gid, paths in groups.items():
        ys = [label_of_path(p, non_dir, frode_dir) for p in paths]
        y = int(np.round(np.mean(ys)))
        group_items.append((gid, y, paths))

    # Split (use same seed for reproducibility)
    (_, _, _), (_, _, _), (test_paths, test_y, _) = stratified_group_split(
        group_items, 0.70, 0.15, 0.15, seed=config.get('seed', 42)
    )
    print(f"Test set: {len(test_paths)}")

    # Dataset
    img_size = config.get('img_size', 224)
    batch_size = config.get('batch_size', 16)
    _, eval_tf = build_transforms(img_size)

    test_ds = ImageBinaryDataset(test_paths, test_y, transform=eval_tf, img_size=img_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Load model
    model_name = config.get('model_name', 'efficientnet_b0')
    drop_rate = config.get('drop_rate', 0.2)
    model = build_model(model_name, pretrained=False, drop_rate=drop_rate).to(device)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint_path}")

    # Evaluate
    print(f"\n=== Evaluation (threshold={args.threshold}) ===")
    test_probs, test_true, test_paths = predict_proba(model, test_loader, device)
    test_metrics = compute_metrics_from_probs(test_probs, test_true, threshold=args.threshold)
    test_cm = confusion_matrix(test_true, (test_probs >= args.threshold).astype(int), labels=[0, 1])

    print(f"Metrics: {test_metrics}")
    print(f"Confusion Matrix:\n{test_cm}")

    # Save outputs
    plot_prob_distributions(test_probs, test_true, save_path=output_dir / "prob_dist.png")
    plot_roc_pr(test_probs, test_true, save_path_roc=output_dir / "roc_curve.png", save_path_pr=output_dir / "pr_curve.png")
    plot_confusion_matrix(test_cm, save_path=output_dir / "cm.png")

    # Save metrics
    metrics_output = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": args.threshold,
        "test_metrics": {k: float(v) if not np.isnan(v) else None for k, v in test_metrics.items()},
        "confusion_matrix": test_cm.tolist(),
        "config": config
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics_output, f, indent=2)

    # Notes
    notes = f"""# Evaluation: {args.run_name}

## Results (threshold={args.threshold})
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

Checkpoint: {args.checkpoint_path}
"""

    with open(output_dir / "notes.md", 'w') as f:
        f.write(notes)

    print(f"\n✓ Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
