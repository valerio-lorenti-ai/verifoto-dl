#!/usr/bin/env python3
"""
Fine-tune model with hard negative mining.

Uses photo_hard_fp.csv to identify hard negative samples (false positives)
and increases their weight during fine-tuning to improve precision.

Usage:
    python scripts/hard_negative_finetune.py \
        --run outputs/runs/2026-02-17_noK2_noLeakage \
        --config configs/baseline.yaml \
        --epochs 5 \
        --lr 1e-5 \
        --repeat_factor 3
"""

import sys
import argparse
from pathlib import Path

import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data import (
    parse_augmented_v6_dataset, group_based_split_v6,
    build_transforms, ImageBinaryDataset
)
from src.utils.model import build_model
from src.utils.metrics import predict_proba, compute_metrics_from_probs
from src.train import set_seed, train_one_epoch, validate, save_checkpoint


def extract_photo_id(path: str) -> str:
    """Extract photo ID from path (first 4 characters)."""
    return Path(path).stem[:4]


def create_hard_negative_sampler(dataset, hard_fp_ids: set, repeat_factor: float = 3.0):
    """
    Create weighted sampler that oversamples hard negative examples.
    
    Args:
        dataset: ImageBinaryDataset
        hard_fp_ids: set of photo_ids that are hard false positives
        repeat_factor: how much to oversample hard negatives
    
    Returns:
        WeightedRandomSampler
    """
    weights = []
    
    for idx in range(len(dataset)):
        path = dataset.df.iloc[idx]['path']
        photo_id = extract_photo_id(path)
        label = dataset.df.iloc[idx]['label']
        
        # Hard negatives: originali (label=0) che sono FP
        if label == 0 and photo_id in hard_fp_ids:
            weights.append(repeat_factor)
        else:
            weights.append(1.0)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with hard negative mining")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tune epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--repeat_factor", type=float, default=3.0, 
                       help="Oversampling factor for hard negatives")
    parser.add_argument("--output_suffix", type=str, default="_hn",
                       help="Suffix for output run name")
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    
    print("="*80)
    print("HARD NEGATIVE MINING FINE-TUNE")
    print("="*80)
    print(f"Base run: {run_dir.name}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Repeat factor: {args.repeat_factor}")
    
    # Load hard FP photo IDs
    hard_fp_file = run_dir / "photo_hard_fp.csv"
    if not hard_fp_file.exists():
        print(f"\n❌ Error: {hard_fp_file} not found")
        print(f"   Run analyze_by_photo.py first to generate hard FP list")
        return 1
    
    hard_fp_df = pd.read_csv(hard_fp_file)
    hard_fp_ids = set(hard_fp_df['photo_id'].values)
    
    print(f"\nHard false positives: {len(hard_fp_ids)} photos")
    print(f"  Examples: {list(hard_fp_ids)[:10]}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load dataset
    print("\n=== Loading dataset ===")
    df = parse_augmented_v6_dataset(config['dataset_root'])
    train_df, val_df, test_df = group_based_split_v6(
        df, 0.70, 0.15, 0.15, seed=config.get('seed', 42)
    )
    
    # Count hard negatives in training set
    train_df['photo_id'] = train_df['path'].apply(extract_photo_id)
    hard_neg_in_train = train_df[
        (train_df['label'] == 0) & (train_df['photo_id'].isin(hard_fp_ids))
    ]
    
    print(f"\nHard negatives in training set:")
    print(f"  Photos: {hard_neg_in_train['photo_id'].nunique()}")
    print(f"  Images: {len(hard_neg_in_train)}")
    print(f"  (will be oversampled by {args.repeat_factor}x)")
    
    # Build datasets
    img_size = config.get('img_size', 224)
    batch_size = config.get('batch_size', 16)
    train_tf, eval_tf = build_transforms(img_size)
    
    train_ds = ImageBinaryDataset(train_df, transform=train_tf, img_size=img_size)
    val_ds = ImageBinaryDataset(val_df, transform=eval_tf, img_size=img_size)
    test_ds = ImageBinaryDataset(test_df, transform=eval_tf, img_size=img_size)
    
    # Create hard negative sampler
    print("\n=== Creating hard negative sampler ===")
    hard_neg_sampler = create_hard_negative_sampler(
        train_ds, hard_fp_ids, repeat_factor=args.repeat_factor
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=hard_neg_sampler,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Load model from checkpoint
    print("\n=== Loading model ===")
    checkpoint_path = run_dir.parent.parent / "checkpoints" / run_dir.name / "best.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    model_name = config.get('model_name', 'efficientnet_b0')
    drop_rate = config.get('drop_rate', 0.2)
    model = build_model(model_name, pretrained=False, drop_rate=drop_rate).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    
    # Setup training
    train_pos = (train_df['label'] == 1).sum()
    train_neg = (train_df['label'] == 0).sum()
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                   weight_decay=config.get('weight_decay', 1e-3))
    
    # Fine-tune
    print(f"\n=== Fine-tuning with hard negative mining ===")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    
    best_val_f1 = 0.0
    output_dir = run_dir.parent / f"{run_dir.name}{args.output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = run_dir.parent.parent / "checkpoints" / f"{run_dir.name}{args.output_suffix}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best.pt"
    
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scheduler=None, max_grad_norm=config.get('max_grad_norm', 1.0),
            device=device
        )
        
        val_m = validate(model, val_loader, threshold=0.5, device=device)
        
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"loss={tr_loss:.4f} val_f1={val_m['f1']:.4f} "
              f"val_prec={val_m['prec']:.4f} val_rec={val_m['rec']:.4f}")
        
        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            save_checkpoint(model, best_ckpt_path, best_metric=best_val_f1, cfg=config)
            print(f"  ✓ Saved best checkpoint (F1={best_val_f1:.4f})")
    
    # Evaluate on test set
    print("\n=== Test Evaluation ===")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    test_probs, test_true, test_metadata = predict_proba(model, test_loader, device)
    
    # Use threshold from original run
    threshold_file = run_dir / "chosen_threshold.json"
    if threshold_file.exists():
        import json
        with open(threshold_file, 'r') as f:
            threshold_data = json.load(f)
            test_threshold = threshold_data.get('recommendation', 0.5)
    else:
        test_threshold = 0.5
    
    test_metrics = compute_metrics_from_probs(test_probs, test_true, threshold=test_threshold)
    
    print(f"\nTest metrics (threshold={test_threshold:.3f}):")
    print(f"  Accuracy:  {test_metrics['acc']:.4f}")
    print(f"  Precision: {test_metrics['prec']:.4f}")
    print(f"  Recall:    {test_metrics['rec']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    
    # Save predictions
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
    print(f"\n✓ Saved: {output_dir}/predictions.csv")
    
    # Run photo-level analysis
    print("\n=== Running photo-level analysis ===")
    import subprocess
    result = subprocess.run([
        sys.executable,
        "scripts/analyze_by_photo.py",
        "--run", str(output_dir),
        "--min-recall", "0.90"
    ], capture_output=False)
    
    print("\n" + "="*80)
    print("HARD NEGATIVE FINE-TUNE COMPLETE")
    print("="*80)
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"✓ Checkpoint saved to: {best_ckpt_path}")
    print(f"\n📊 Compare with original run:")
    print(f"  python scripts/compare_runs.py \\")
    print(f"    --run1 {run_dir} \\")
    print(f"    --run2 {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
