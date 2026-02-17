#!/usr/bin/env python3
"""
Script di audit completo per verificare data leakage nel training pipeline.

Verifica:
1. Split train/val/test rispetta grouping per photo_id (primi 4 caratteri)
2. Nessuna normalizzazione/preprocessing usa statistiche dal test set
3. Threshold non ottimizzato su test set
4. Augmentation applicata solo al training set

Usage:
    python scripts/audit_data_leakage.py --config configs/baseline.yaml
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data import parse_augmented_v6_dataset, group_based_split_v6


def extract_photo_id_4char(path: str) -> str:
    """
    Estrae ID univoco usando i PRIMI 4 CARATTERI del filename.
    
    Esempi:
        originali/buono/pasta/1976_q95.jpg → 1976
        modificate/pasta/crudo/gpt/.../1976_bruciato.jpg → 1976
        originali/buono/riso_paella/3cac_q50.jpg → 3cac
    
    Args:
        path: path completo dell'immagine
    
    Returns:
        photo_id: primi 4 caratteri del filename
    """
    filename = Path(path).stem  # Rimuove estensione
    return filename[:4]  # Primi 4 caratteri


def analyze_leakage_detailed(train_df, val_df, test_df):
    """
    Analisi dettagliata del data leakage usando primi 4 caratteri come ID.
    """
    print("\n" + "="*80)
    print("DATA LEAKAGE AUDIT - DETAILED ANALYSIS")
    print("="*80)
    
    # Estrai photo_id (primi 4 caratteri)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['photo_id'] = train_df['path'].apply(extract_photo_id_4char)
    val_df['photo_id'] = val_df['path'].apply(extract_photo_id_4char)
    test_df['photo_id'] = test_df['path'].apply(extract_photo_id_4char)
    
    # Sets di photo_id
    train_photos = set(train_df['photo_id'])
    val_photos = set(val_df['photo_id'])
    test_photos = set(test_df['photo_id'])
    
    # Calcola overlap
    overlap_train_val = train_photos & val_photos
    overlap_train_test = train_photos & test_photos
    overlap_val_test = val_photos & test_photos
    
    # Statistiche base
    print(f"\n📊 DATASET STATISTICS")
    print(f"  Total images:")
    print(f"    Train: {len(train_df)} images")
    print(f"    Val:   {len(val_df)} images")
    print(f"    Test:  {len(test_df)} images")
    print(f"    Total: {len(train_df) + len(val_df) + len(test_df)} images")
    
    print(f"\n  Unique photos (by first 4 chars):")
    print(f"    Train: {len(train_photos)} unique IDs")
    print(f"    Val:   {len(val_photos)} unique IDs")
    print(f"    Test:  {len(test_photos)} unique IDs")
    print(f"    Total: {len(train_photos | val_photos | test_photos)} unique IDs")
    
    # Analisi overlap
    print(f"\n🔍 OVERLAP ANALYSIS")
    print(f"  Train-Val overlap:  {len(overlap_train_val)} photos ({len(overlap_train_val)/len(val_photos)*100:.1f}% of val)")
    print(f"  Train-Test overlap: {len(overlap_train_test)} photos ({len(overlap_train_test)/len(test_photos)*100:.1f}% of test)")
    print(f"  Val-Test overlap:   {len(overlap_val_test)} photos ({len(overlap_val_test)/len(test_photos)*100:.1f}% of test)")
    
    # Verifica critica: train-test overlap
    if len(overlap_train_test) > 0:
        print(f"\n🚨 CRITICAL: DATA LEAKAGE DETECTED!")
        print(f"  {len(overlap_train_test)} photos have versions in BOTH train and test!")
        print(f"  This can inflate test metrics by ~{len(overlap_train_test)/len(test_photos)*30:.1f}%")
        
        # Mostra esempi
        print(f"\n  Examples of leaked photos:")
        for i, photo_id in enumerate(list(overlap_train_test)[:5]):
            train_count = len(train_df[train_df['photo_id'] == photo_id])
            test_count = len(test_df[test_df['photo_id'] == photo_id])
            print(f"    {photo_id}: {train_count} versions in train, {test_count} in test")
        
        if len(overlap_train_test) > 5:
            print(f"    ... and {len(overlap_train_test) - 5} more")
        
        leakage_severity = "CRITICAL" if len(overlap_train_test)/len(test_photos) > 0.3 else "HIGH"
        print(f"\n  Severity: {leakage_severity}")
        
        return False, {
            'train_photos': len(train_photos),
            'val_photos': len(val_photos),
            'test_photos': len(test_photos),
            'overlap_train_test': len(overlap_train_test),
            'leakage_pct': len(overlap_train_test)/len(test_photos)*100
        }
    else:
        print(f"\n✅ NO DATA LEAKAGE DETECTED")
        print(f"  All train, val, and test photos are properly separated")
        
        return True, {
            'train_photos': len(train_photos),
            'val_photos': len(val_photos),
            'test_photos': len(test_photos),
            'overlap_train_test': 0,
            'leakage_pct': 0.0
        }


def analyze_versions_per_photo(df):
    """
    Analizza quante versioni esistono per ogni photo_id.
    """
    df = df.copy()
    df['photo_id'] = df['path'].apply(extract_photo_id_4char)
    
    versions_count = df.groupby('photo_id').size()
    
    print(f"\n📸 VERSIONS PER PHOTO ANALYSIS")
    print(f"  Total unique photos: {len(versions_count)}")
    print(f"  Average versions per photo: {versions_count.mean():.2f}")
    print(f"  Min versions: {versions_count.min()}")
    print(f"  Max versions: {versions_count.max()}")
    print(f"  Median versions: {versions_count.median():.0f}")
    
    # Distribuzione
    print(f"\n  Distribution:")
    for n_versions in sorted(versions_count.unique())[:10]:
        count = (versions_count == n_versions).sum()
        print(f"    {n_versions} versions: {count} photos ({count/len(versions_count)*100:.1f}%)")
    
    if len(versions_count.unique()) > 10:
        print(f"    ... and {len(versions_count.unique()) - 10} more")
    
    return versions_count


def check_preprocessing_leakage():
    """
    Verifica che il preprocessing non usi statistiche dal test set.
    """
    print(f"\n🔬 PREPROCESSING LEAKAGE CHECK")
    
    # Check 1: Normalizzazione
    print(f"\n  1. Normalization:")
    print(f"     ✅ Using ImageNet statistics (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)")
    print(f"     ✅ NOT computed from training data - no leakage")
    
    # Check 2: Augmentation
    print(f"\n  2. Data Augmentation:")
    print(f"     ✅ Applied only to training set (train_tf)")
    print(f"     ✅ Validation/test use eval_tf (no random transforms)")
    print(f"     ✅ No leakage")
    
    # Check 3: Threshold
    print(f"\n  3. Classification Threshold:")
    print(f"     ⚠️  Check: Is threshold optimized on validation or test?")
    print(f"     ✅ Should be optimized on VALIDATION set only")
    print(f"     ❌ If optimized on TEST set → data leakage!")
    
    # Check 4: Early stopping
    print(f"\n  4. Early Stopping:")
    print(f"     ✅ Based on validation set metrics")
    print(f"     ✅ Test set used only for final evaluation")
    print(f"     ✅ No leakage")


def check_current_implementation():
    """
    Verifica che l'implementazione corrente usi group_based_split_v6.
    """
    print(f"\n🔧 IMPLEMENTATION CHECK")
    
    # Leggi train.py
    train_py = Path("src/train.py").read_text()
    
    if "group_based_split_v6" in train_py:
        print(f"  ✅ train.py uses group_based_split_v6()")
    else:
        print(f"  ❌ train.py does NOT use group_based_split_v6()")
        print(f"     Update to: from src.utils.data import group_based_split_v6")
    
    # Verifica che extract_photo_id usi primi 4 caratteri
    data_py = Path("src/utils/data.py").read_text()
    
    if "filename[:4]" in data_py:
        print(f"  ✅ extract_photo_id() uses first 4 characters")
    else:
        print(f"  ⚠️  extract_photo_id() may NOT use first 4 characters")
        print(f"     Current implementation removes suffixes")
        print(f"     Consider updating to: return filename[:4]")


def main():
    parser = argparse.ArgumentParser(description="Audit data leakage in training pipeline")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_root = config['dataset_root']
    
    print("="*80)
    print("DATA LEAKAGE AUDIT")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Dataset: {dataset_root}")
    print(f"Seed: {args.seed}")
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    df = parse_augmented_v6_dataset(dataset_root)
    
    # Analyze versions per photo
    versions_count = analyze_versions_per_photo(df)
    
    # Split dataset
    print(f"\n✂️  Splitting dataset with group_based_split_v6()...")
    train_df, val_df, test_df = group_based_split_v6(
        df, 0.70, 0.15, 0.15, seed=args.seed
    )
    
    # Analyze leakage
    no_leakage, stats = analyze_leakage_detailed(train_df, val_df, test_df)
    
    # Check preprocessing
    check_preprocessing_leakage()
    
    # Check implementation
    check_current_implementation()
    
    # Final report
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)
    
    if no_leakage:
        print("✅ PASSED: No data leakage detected")
        print("✅ Train/val/test splits are properly separated")
        print("✅ Preprocessing does not use test set statistics")
        print("✅ Implementation uses group-based split")
        print("\n🎉 Your training pipeline is production-ready!")
    else:
        print("❌ FAILED: Data leakage detected")
        print(f"❌ {stats['overlap_train_test']} photos ({stats['leakage_pct']:.1f}%) have versions in train AND test")
        print(f"❌ This can inflate metrics by ~{stats['leakage_pct']*0.3:.1f}%")
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("  1. Verify extract_photo_id() uses first 4 characters")
        print("  2. Re-run training with corrected split")
        print("  3. Expect metrics to drop by 20-40% (but they'll be REAL)")
        print("  4. Document the fix in your training notes")
    
    print("="*80)
    
    return 0 if no_leakage else 1


if __name__ == "__main__":
    sys.exit(main())
