#!/usr/bin/env python3
"""
Verify if there's data leakage in hard negative fine-tuning.
Checks if hard FP photos from original test set ended up in hard negative training set.
"""
import sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data import parse_augmented_v6_dataset, group_based_split_v6, domain_aware_group_split_v1, extract_photo_id

# Paths
ORIGINAL_RUN = "outputs/runs/2026-02-20_convnext_v8_domaiAware"
HARD_FP_FILE = f"{ORIGINAL_RUN}/photo_hard_fp.csv"
CONFIG_FILE = "configs/convnext_v8.yaml"

print("=" * 80)
print("HARD NEGATIVE DATA LEAKAGE VERIFICATION")
print("=" * 80)

# Load hard FP photos from original run
hard_fp_df = pd.read_csv(HARD_FP_FILE)
hard_fp_ids = set(hard_fp_df['photo_id'].values)
print(f"\nHard FP photos from ORIGINAL test set: {len(hard_fp_ids)}")
print(f"  Examples: {list(hard_fp_ids)[:10]}")

# Load config
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

# Load dataset
df = parse_augmented_v6_dataset(config['dataset_root'])
df['photo_id'] = df['path'].apply(extract_photo_id)

print(f"\n=== ORIGINAL SPLIT (domain_aware) ===")
# Original split (domain_aware)
train_df_orig, val_df_orig, test_df_orig = domain_aware_group_split_v1(
    df, 0.70, 0.15, 0.15, seed=config.get('seed', 42), include_food=False
)

train_photos_orig = set(train_df_orig['photo_id'].unique())
val_photos_orig = set(val_df_orig['photo_id'].unique())
test_photos_orig = set(test_df_orig['photo_id'].unique())

print(f"Train photos: {len(train_photos_orig)}")
print(f"Val photos:   {len(val_photos_orig)}")
print(f"Test photos:  {len(test_photos_orig)}")

# Check where hard FP photos are in original split
hard_fp_in_train_orig = hard_fp_ids & train_photos_orig
hard_fp_in_val_orig = hard_fp_ids & val_photos_orig
hard_fp_in_test_orig = hard_fp_ids & test_photos_orig

print(f"\nHard FP photos location in ORIGINAL split:")
print(f"  In train: {len(hard_fp_in_train_orig)}")
print(f"  In val:   {len(hard_fp_in_val_orig)}")
print(f"  In test:  {len(hard_fp_in_test_orig)} ← Should be ALL of them")

print(f"\n=== HARD NEGATIVE SPLIT (group_based_v6) ===")
# Hard negative split (group_based_v6 - WRONG!)
train_df_hn, val_df_hn, test_df_hn = group_based_split_v6(
    df, 0.70, 0.15, 0.15, seed=config.get('seed', 42)
)

train_photos_hn = set(train_df_hn['photo_id'].unique())
val_photos_hn = set(val_df_hn['photo_id'].unique())
test_photos_hn = set(test_df_hn['photo_id'].unique())

print(f"Train photos: {len(train_photos_hn)}")
print(f"Val photos:   {len(val_photos_hn)}")
print(f"Test photos:  {len(test_photos_hn)}")

# Check where hard FP photos are in hard negative split
hard_fp_in_train_hn = hard_fp_ids & train_photos_hn
hard_fp_in_val_hn = hard_fp_ids & val_photos_hn
hard_fp_in_test_hn = hard_fp_ids & test_photos_hn

print(f"\nHard FP photos location in HARD NEGATIVE split:")
print(f"  In train: {len(hard_fp_in_train_hn)} ← DATA LEAKAGE if > 0!")
print(f"  In val:   {len(hard_fp_in_val_hn)}")
print(f"  In test:  {len(hard_fp_in_test_hn)}")

print("\n" + "=" * 80)
print("LEAKAGE ANALYSIS")
print("=" * 80)

# Photos that moved from test to train
leaked_photos = hard_fp_in_test_orig & hard_fp_in_train_hn

if len(leaked_photos) > 0:
    print(f"\n🚨 DATA LEAKAGE DETECTED!")
    print(f"\n{len(leaked_photos)} hard FP photos moved from ORIGINAL test → HARD NEGATIVE train:")
    print(f"  {list(leaked_photos)}")
    
    # Count images
    leaked_images = train_df_hn[train_df_hn['photo_id'].isin(leaked_photos)]
    print(f"\n  Total leaked images: {len(leaked_images)}")
    print(f"  These photos were in original TEST set (used to identify hard FPs)")
    print(f"  But are now in HARD NEGATIVE TRAIN set (model sees them during training)")
    print(f"\n  This explains the unrealistic performance improvement!")
    
    print("\n" + "=" * 80)
    print("IMPACT")
    print("=" * 80)
    print(f"\nThe model is being trained on photos it will be tested on.")
    print(f"This is why precision jumped from 74.2% to 97.8% (+23.6%)")
    print(f"\nThe results are NOT valid - this is data leakage.")
    
else:
    print(f"\n✅ No data leakage detected")
    print(f"All hard FP photos remained in their original splits")

print("\n" + "=" * 80)
print("SOLUTION")
print("=" * 80)
print(f"\nhard_negative_finetune.py must use the SAME split strategy as training:")
print(f"  Current: group_based_split_v6 (WRONG)")
print(f"  Should be: domain_aware_group_split_v1 (from config)")
print(f"\nFix: Update hard_negative_finetune.py to read split_strategy from config")
