#!/usr/bin/env python3
"""
Quick test script per validare domain_aware_group_split_v1().
Usa un dataset sintetico per verificare che lo split funzioni correttamente.
"""

import sys
import random
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data import domain_aware_group_split_v1, extract_photo_id


def create_synthetic_dataset(n_photos=100, seed=42):
    """
    Crea dataset sintetico per test.
    
    Simula:
    - 40% originali your_original
    - 30% originali kaggle_vale
    - 30% modificate (15% GPT-1-mini, 15% GPT-1.5)
    """
    rnd = random.Random(seed)
    
    records = []
    photo_id = 1000
    
    # Originali your_original (40%)
    n_your = int(n_photos * 0.40)
    for i in range(n_your):
        pid = f"{photo_id:04d}"
        # Ogni foto ha 2-4 versioni (crop, resize, etc.)
        n_versions = rnd.randint(2, 4)
        for v in range(n_versions):
            records.append({
                'path': f'/fake/originali/buono/pasta/{pid}_v{v}.jpg',
                'label': 0,
                'source': 'your_original',
                'generator': None,
                'food_category': rnd.choice(['pasta', 'riso', 'carne']),
                'quality': 'buono',
                'defect_type': None
            })
        photo_id += 1
    
    # Originali kaggle_vale (30%)
    n_kaggle = int(n_photos * 0.30)
    for i in range(n_kaggle):
        pid = f"{photo_id:04d}"
        n_versions = rnd.randint(2, 4)
        for v in range(n_versions):
            records.append({
                'path': f'/fake/originali/buono/pasta/{pid}_v{v}.jpg',
                'label': 0,
                'source': 'kaggle_vale',
                'generator': None,
                'food_category': rnd.choice(['pasta', 'riso', 'carne']),
                'quality': 'buono',
                'defect_type': None
            })
        photo_id += 1
    
    # Modificate GPT-1-mini (15%)
    n_gpt_mini = int(n_photos * 0.15)
    for i in range(n_gpt_mini):
        pid = f"{photo_id:04d}"
        n_versions = rnd.randint(2, 4)
        for v in range(n_versions):
            records.append({
                'path': f'/fake/modificate/pasta/bruciato/gpt_image_1_mini/{pid}_v{v}.jpg',
                'label': 1,
                'source': 'your_ai',
                'generator': 'gpt_image_1_mini',
                'food_category': rnd.choice(['pasta', 'riso', 'carne']),
                'quality': None,
                'defect_type': 'bruciato'
            })
        photo_id += 1
    
    # Modificate GPT-1.5 (15%)
    n_gpt_15 = int(n_photos * 0.15)
    for i in range(n_gpt_15):
        pid = f"{photo_id:04d}"
        n_versions = rnd.randint(2, 4)
        for v in range(n_versions):
            records.append({
                'path': f'/fake/modificate/pasta/crudo/gpt_image_1_5/{pid}_v{v}.jpg',
                'label': 1,
                'source': 'your_ai',
                'generator': 'gpt_image_1_5',
                'food_category': rnd.choice(['pasta', 'riso', 'carne']),
                'quality': None,
                'defect_type': 'crudo'
            })
        photo_id += 1
    
    df = pd.DataFrame(records)
    return df


def validate_split(train_df, val_df, test_df):
    """Valida che lo split sia corretto."""
    
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    # 1. No overlap di photo_id
    train_photos = set(train_df['path'].apply(extract_photo_id))
    val_photos = set(val_df['path'].apply(extract_photo_id))
    test_photos = set(test_df['path'].apply(extract_photo_id))
    
    overlap_train_val = train_photos & val_photos
    overlap_train_test = train_photos & test_photos
    overlap_val_test = val_photos & test_photos
    
    print(f"\n1) NO OVERLAP CHECK:")
    print(f"   Train-Val overlap: {len(overlap_train_val)} {'✓' if len(overlap_train_val) == 0 else '✗'}")
    print(f"   Train-Test overlap: {len(overlap_train_test)} {'✓' if len(overlap_train_test) == 0 else '✗'}")
    print(f"   Val-Test overlap: {len(overlap_val_test)} {'✓' if len(overlap_val_test) == 0 else '✗'}")
    
    # 2. Proporzioni split
    total = len(train_df) + len(val_df) + len(test_df)
    train_pct = len(train_df) / total * 100
    val_pct = len(val_df) / total * 100
    test_pct = len(test_df) / total * 100
    
    print(f"\n2) SPLIT PROPORTIONS:")
    print(f"   Train: {train_pct:.1f}% (target: 70%)")
    print(f"   Val:   {val_pct:.1f}% (target: 15%)")
    print(f"   Test:  {test_pct:.1f}% (target: 15%)")
    
    # 3. Domain balance
    print(f"\n3) DOMAIN BALANCE:")
    
    # Source balance
    for source in ['your_original', 'kaggle_vale', 'your_ai']:
        train_count = (train_df['source'] == source).sum()
        val_count = (val_df['source'] == source).sum()
        test_count = (test_df['source'] == source).sum()
        total_source = train_count + val_count + test_count
        
        if total_source > 0:
            train_pct_src = train_count / total_source * 100
            val_pct_src = val_count / total_source * 100
            test_pct_src = test_count / total_source * 100
            
            # Check if within ±5% of target
            train_ok = abs(train_pct_src - 70) < 5
            val_ok = abs(val_pct_src - 15) < 5
            test_ok = abs(test_pct_src - 15) < 5
            
            status = '✓' if (train_ok and val_ok and test_ok) else '⚠️'
            
            print(f"   {source:20s}: Train={train_pct_src:5.1f}%  Val={val_pct_src:5.1f}%  Test={test_pct_src:5.1f}%  {status}")
    
    # Generator balance (solo AI)
    ai_train = train_df[train_df['label'] == 1]
    ai_val = val_df[val_df['label'] == 1]
    ai_test = test_df[test_df['label'] == 1]
    
    for gen in ['gpt_image_1_mini', 'gpt_image_1_5']:
        train_count = (ai_train['generator'] == gen).sum()
        val_count = (ai_val['generator'] == gen).sum()
        test_count = (ai_test['generator'] == gen).sum()
        total_gen = train_count + val_count + test_count
        
        if total_gen > 0:
            train_pct_gen = train_count / total_gen * 100
            val_pct_gen = val_count / total_gen * 100
            test_pct_gen = test_count / total_gen * 100
            
            train_ok = abs(train_pct_gen - 70) < 5
            val_ok = abs(val_pct_gen - 15) < 5
            test_ok = abs(test_pct_gen - 15) < 5
            
            status = '✓' if (train_ok and val_ok and test_ok) else '⚠️'
            
            print(f"   {gen:20s}: Train={train_pct_gen:5.1f}%  Val={val_pct_gen:5.1f}%  Test={test_pct_gen:5.1f}%  {status}")
    
    # 4. Label balance
    print(f"\n4) LABEL BALANCE:")
    train_pos_rate = train_df['label'].mean()
    val_pos_rate = val_df['label'].mean()
    test_pos_rate = test_df['label'].mean()
    
    print(f"   Train pos rate: {train_pos_rate:.3f}")
    print(f"   Val pos rate:   {val_pos_rate:.3f}")
    print(f"   Test pos rate:  {test_pos_rate:.3f}")
    
    # Check if similar (within 5%)
    pos_rate_ok = (abs(train_pos_rate - val_pos_rate) < 0.05 and 
                   abs(train_pos_rate - test_pos_rate) < 0.05)
    print(f"   Status: {'✓' if pos_rate_ok else '⚠️'}")
    
    print("\n" + "="*80)
    
    # Overall status
    all_ok = (len(overlap_train_val) == 0 and 
              len(overlap_train_test) == 0 and 
              len(overlap_val_test) == 0 and
              pos_rate_ok)
    
    if all_ok:
        print("✅ ALL CHECKS PASSED!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review output above")
    
    print("="*80)


def main():
    print("="*80)
    print("DOMAIN-AWARE SPLIT TEST")
    print("="*80)
    
    # Create synthetic dataset
    print("\n1) Creating synthetic dataset...")
    df = create_synthetic_dataset(n_photos=100, seed=42)
    
    print(f"   Total images: {len(df)}")
    print(f"   Unique photos: {df['path'].apply(extract_photo_id).nunique()}")
    print(f"   Sources: {df['source'].unique().tolist()}")
    print(f"   Generators: {df['generator'].dropna().unique().tolist()}")
    
    # Test domain_aware_group_split_v1
    print("\n2) Testing domain_aware_group_split_v1()...")
    train_df, val_df, test_df = domain_aware_group_split_v1(
        df, 
        train_ratio=0.70, 
        val_ratio=0.15, 
        test_ratio=0.15, 
        seed=42,
        include_food=False
    )
    
    # Validate
    print("\n3) Validating split...")
    validate_split(train_df, val_df, test_df)
    
    # Test with include_food=True
    print("\n" + "="*80)
    print("TESTING WITH include_food=True")
    print("="*80)
    
    train_df2, val_df2, test_df2 = domain_aware_group_split_v1(
        df, 
        train_ratio=0.70, 
        val_ratio=0.15, 
        test_ratio=0.15, 
        seed=42,
        include_food=True
    )
    
    validate_split(train_df2, val_df2, test_df2)
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
