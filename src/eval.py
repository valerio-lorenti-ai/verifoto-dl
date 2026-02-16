import os
import sys
import time
import json
import random
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.data import (
    parse_augmented_v6_dataset, stratified_group_split_v6,
    build_transforms, ImageBinaryDataset
)
from src.utils.model import build_model
from src.utils.metrics import (
    predict_proba, compute_metrics_from_probs, compute_group_metrics,
    get_top_errors
)
from src.utils.visualization import plot_prob_distributions, plot_roc_pr, plot_confusion_matrix
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
    df = parse_augmented_v6_dataset(dataset_root)
    
    # Split (use same seed for reproducibility)
    _, _, test_df = stratified_group_split_v6(
        df, 0.70, 0.15, 0.15, seed=config.get('seed', 42)
    )
    print(f"Test set: {len(test_df)}")

    # Dataset
    img_size = config.get('img_size', 224)
    batch_size = config.get('batch_size', 16)
    _, eval_tf = build_transforms(img_size)

    test_ds = ImageBinaryDataset(test_df, transform=eval_tf, img_size=img_size)
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
    test_probs, test_true, test_metadata = predict_proba(model, test_loader, device)
    test_metrics = compute_metrics_from_probs(test_probs, test_true, threshold=args.threshold)
    test_cm = confusion_matrix(test_true, (test_probs >= args.threshold).astype(int), labels=[0, 1])

    print(f"Metrics: {test_metrics}")
    print(f"Confusion Matrix:\n{test_cm}")

    # Crea DataFrame con predictions e metadati
    predictions_df = pd.DataFrame({
        'path': [m['path'] for m in test_metadata],
        'y_true': test_true,
        'y_prob': test_probs,
        'y_pred': (test_probs >= args.threshold).astype(int),
        'source': [m.get('source') for m in test_metadata],
        'quality': [m.get('quality') for m in test_metadata],
        'food_category': [m.get('food_category') for m in test_metadata],
        'defect_type': [m.get('defect_type') for m in test_metadata],
        'generator': [m.get('generator') for m in test_metadata]
    })
    
    # Salva predictions
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"✓ Saved predictions.csv ({len(predictions_df)} samples)")
    
    # Calcola e salva group metrics
    print("\n=== Computing group metrics ===")
    
    group_food = compute_group_metrics(predictions_df, 'food_category', threshold=args.threshold)
    group_food.to_csv(output_dir / "group_metrics_food.csv", index=False)
    print(f"✓ Saved group_metrics_food.csv ({len(group_food)} groups)")
    
    group_defect = compute_group_metrics(predictions_df, 'defect_type', threshold=args.threshold)
    group_defect.to_csv(output_dir / "group_metrics_defect.csv", index=False)
    print(f"✓ Saved group_metrics_defect.csv ({len(group_defect)} groups)")
    
    group_generator = compute_group_metrics(predictions_df, 'generator', threshold=args.threshold)
    group_generator.to_csv(output_dir / "group_metrics_generator.csv", index=False)
    print(f"✓ Saved group_metrics_generator.csv ({len(group_generator)} groups)")
    
    group_quality = compute_group_metrics(predictions_df, 'quality', threshold=args.threshold)
    group_quality.to_csv(output_dir / "group_metrics_quality.csv", index=False)
    print(f"✓ Saved group_metrics_quality.csv ({len(group_quality)} groups)")
    
    # Top errors
    top_fp = get_top_errors(predictions_df, error_type='fp', top_n=50)
    top_fp.to_csv(output_dir / "top_false_positives.csv", index=False)
    print(f"✓ Saved top_false_positives.csv ({len(top_fp)} errors)")
    
    top_fn = get_top_errors(predictions_df, error_type='fn', top_n=50)
    top_fn.to_csv(output_dir / "top_false_negatives.csv", index=False)
    print(f"✓ Saved top_false_negatives.csv ({len(top_fn)} errors)")

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
