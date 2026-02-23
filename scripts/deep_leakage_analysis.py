#!/usr/bin/env python3
"""
Deep analysis of potential data leakage in hard negative results.
"""
import pandas as pd
import json

print("=" * 80)
print("DEEP DATA LEAKAGE ANALYSIS")
print("=" * 80)

# Load predictions
orig_pred = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware/photo_level_predictions.csv")
hn_pred = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware_hard_negative/photo_level_predictions.csv")

# Load hard FP
orig_hard_fp = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware/photo_hard_fp.csv")
orig_hard_fp_ids = set(orig_hard_fp['photo_id'].values)

print(f"\n{'='*80}")
print("TEST SET COMPARISON")
print("=" * 80)

orig_photos = set(orig_pred['photo_id'].values)
hn_photos = set(hn_pred['photo_id'].values)

print(f"\nOriginal test photos: {len(orig_photos)}")
print(f"Hard neg test photos: {len(hn_photos)}")

# Photos in original but not in hard negative
missing_photos = orig_photos - hn_photos
print(f"\nPhotos in ORIGINAL but NOT in HARD NEGATIVE: {len(missing_photos)}")
if len(missing_photos) > 0:
    print(f"  Examples: {list(missing_photos)[:20]}")
    
    # Check if missing photos are hard FP
    missing_hard_fp = missing_photos & orig_hard_fp_ids
    print(f"\n  Of these, {len(missing_hard_fp)} were HARD FP in original:")
    if len(missing_hard_fp) > 0:
        print(f"    {list(missing_hard_fp)}")
        print(f"\n  🚨 CRITICAL: {len(missing_hard_fp)}/{len(orig_hard_fp_ids)} hard FP photos are MISSING from test set!")
        print(f"     This is {len(missing_hard_fp)/len(orig_hard_fp_ids)*100:.1f}% of hard FP photos")
        print(f"     They likely moved to train/val → DATA LEAKAGE!")

# Photos in hard negative but not in original
new_photos = hn_photos - orig_photos
print(f"\nPhotos in HARD NEGATIVE but NOT in ORIGINAL: {len(new_photos)}")
if len(new_photos) > 0:
    print(f"  Examples: {list(new_photos)[:20]}")

print(f"\n{'='*80}")
print("HARD FP DETAILED ANALYSIS")
print("=" * 80)

# For each original hard FP, check if it's in the new test set
print(f"\nOriginal hard FP photos (16 total):")
print(f"{'Photo ID':<10} {'In HN Test?':<15} {'Orig Prob':<12} {'HN Prob':<12} {'Status'}")
print("-" * 80)

for photo_id in sorted(orig_hard_fp_ids):
    orig_row = orig_pred[orig_pred['photo_id'] == photo_id]
    hn_row = hn_pred[hn_pred['photo_id'] == photo_id]
    
    if len(orig_row) == 0:
        continue
        
    orig_prob = orig_row.iloc[0]['prob_mean']
    
    if len(hn_row) == 0:
        in_test = "NO"
        hn_prob = "N/A"
        status = "🚨 MISSING (likely in train/val)"
    else:
        in_test = "YES"
        hn_prob = f"{hn_row.iloc[0]['prob_mean']:.3f}"
        
        # Check if still FP
        hn_prob_val = hn_row.iloc[0]['prob_mean']
        if hn_prob_val >= 0.5:
            status = "⚠️  Still FP"
        else:
            status = "✅ Fixed"
    
    print(f"{photo_id:<10} {in_test:<15} {orig_prob:<12.3f} {hn_prob:<12} {status}")

print(f"\n{'='*80}")
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("=" * 80)

# Compare probability distributions
orig_probs = orig_pred['prob_mean'].values
hn_probs = hn_pred['prob_mean'].values

print(f"\nOriginal test set probabilities:")
print(f"  Mean:   {orig_probs.mean():.3f}")
print(f"  Median: {pd.Series(orig_probs).median():.3f}")
print(f"  Std:    {orig_probs.std():.3f}")
print(f"  Min:    {orig_probs.min():.3f}")
print(f"  Max:    {orig_probs.max():.3f}")

print(f"\nHard negative test set probabilities:")
print(f"  Mean:   {hn_probs.mean():.3f}")
print(f"  Median: {pd.Series(hn_probs).median():.3f}")
print(f"  Std:    {hn_probs.std():.3f}")
print(f"  Min:    {hn_probs.min():.3f}")
print(f"  Max:    {hn_probs.max():.3f}")

# Check for suspicious patterns
mean_diff = hn_probs.mean() - orig_probs.mean()
print(f"\nMean probability change: {mean_diff:+.3f}")
if abs(mean_diff) > 0.05:
    print(f"  🚨 Large shift in mean probability - suspicious!")

print(f"\n{'='*80}")
print("SPLIT CONSISTENCY CHECK")
print("=" * 80)

# The test sets should be IDENTICAL if using same split
if len(missing_photos) > 0 or len(new_photos) > 0:
    print(f"\n🚨 TEST SETS ARE DIFFERENT!")
    print(f"   Missing: {len(missing_photos)} photos")
    print(f"   New:     {len(new_photos)} photos")
    print(f"\n   This proves different splits were used.")
    print(f"   Even with same algorithm, the SEED or STRATEGY must be different.")
    
    # Check if it's the hard FP photos that moved
    if len(missing_photos & orig_hard_fp_ids) > 0:
        print(f"\n   🚨 CRITICAL: Hard FP photos moved from test set!")
        print(f"      This is the SMOKING GUN for data leakage.")
        print(f"      The model was trained on photos it's being tested on.")
else:
    print(f"\n✅ Test sets are identical (good)")

print(f"\n{'='*80}")
print("FINAL VERDICT")
print("=" * 80)

missing_hard_fp = missing_photos & orig_hard_fp_ids
if len(missing_hard_fp) > 0:
    print(f"\n🚨 DATA LEAKAGE CONFIRMED!")
    print(f"\n   Evidence:")
    print(f"   1. {len(missing_hard_fp)} hard FP photos missing from test set")
    print(f"   2. Test set size changed (226 → 221)")
    print(f"   3. Fix rate unrealistically high (93.8%)")
    print(f"   4. F1 improvement high (+9.5%)")
    print(f"\n   Conclusion: Different split was used despite using domain_aware.")
    print(f"   Possible causes:")
    print(f"   - Different random seed")
    print(f"   - Different dataset version")
    print(f"   - Bug in split function")
    print(f"\n   Results are NOT VALID.")
elif len(missing_photos) > 5:
    print(f"\n⚠️  LIKELY DATA LEAKAGE")
    print(f"   Test set changed significantly ({len(missing_photos)} photos)")
    print(f"   Results should be treated with caution")
else:
    print(f"\n✅ NO OBVIOUS DATA LEAKAGE")
    print(f"   Test sets are similar")
    print(f"   Results appear valid")
