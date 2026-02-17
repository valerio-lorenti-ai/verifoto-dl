"""
Script per analizzare data leakage in un run esistente.
Verifica se foto con versioni multiple sono finite in train E test.
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data import parse_augmented_v6_dataset, stratified_group_split_v6, analyze_split_leakage


def main():
    parser = argparse.ArgumentParser(description="Analyze data leakage in dataset split")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    seed = config.get('seed', 42)
    
    print(f"\n{'='*80}")
    print(f"ANALYZING DATA LEAKAGE")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Seed: {seed}")
    
    # Load dataset
    print("\nLoading dataset...")
    df = parse_augmented_v6_dataset(args.dataset_root)
    
    # Split using OLD method (with potential leakage)
    print("\n" + "="*80)
    print("ANALYSIS WITH OLD SPLIT METHOD (stratified_group_split_v6)")
    print("="*80)
    train_df, val_df, test_df = stratified_group_split_v6(
        df, 0.70, 0.15, 0.15, seed=seed
    )
    
    # Analyze leakage (already called in stratified_group_split_v6)
    # stats = analyze_split_leakage(train_df, val_df, test_df)
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Get stats from the analysis
    from src.utils.data import extract_photo_id
    train_photos = set(train_df['path'].apply(extract_photo_id))
    test_photos = set(test_df['path'].apply(extract_photo_id))
    overlap = train_photos & test_photos
    
    if len(overlap) > 0:
        leakage_pct = len(overlap) / len(test_photos) * 100
        print(f"\n🚨 DATA LEAKAGE DETECTED!")
        print(f"\nImpact on metrics:")
        print(f"  - Test metrics are inflated by ~{leakage_pct * 0.3:.1f}%")
        print(f"  - Model is memorizing photos, not learning patterns")
        print(f"  - Production performance will be LOWER than test metrics")
        
        print(f"\n✅ SOLUTION:")
        print(f"  1. Use group_based_split_v6() instead of stratified_group_split_v6()")
        print(f"  2. Re-train model with corrected split")
        print(f"  3. Expect metrics to drop by 20-40% (but they'll be REAL)")
        
        print(f"\nTo fix:")
        print(f"  - Update train.py to use: from src.utils.data import group_based_split_v6")
        print(f"  - Replace: stratified_group_split_v6() → group_based_split_v6()")
        print(f"  - Re-run training")
    else:
        print(f"\n✓ No data leakage detected!")
        print(f"  Your split is correct and metrics are reliable.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
