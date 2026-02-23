#!/usr/bin/env python3
"""
Analyze hard negative results to detect data leakage.
"""
import pandas as pd
import json

print("=" * 80)
print("HARD NEGATIVE RESULTS ANALYSIS")
print("=" * 80)

# Load original results
orig_metrics = json.load(open("outputs/runs/2026-02-20_convnext_v8_domaiAware/photo_level_metrics.json"))
orig_hard_fp = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware/photo_hard_fp.csv")
orig_predictions = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware/photo_level_predictions.csv")

# Load hard negative results
hn_metrics = json.load(open("outputs/runs/2026-02-20_convnext_v8_domaiAware_hard_negative/photo_level_metrics.json"))
hn_hard_fp = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware_hard_negative/photo_hard_fp.csv")
hn_predictions = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware_hard_negative/photo_level_predictions.csv")

print("\n" + "=" * 80)
print("METRICS COMPARISON")
print("=" * 80)

# Compare at optimal F1 threshold
orig_opt = orig_metrics['optimal_f1_threshold']
hn_opt = hn_metrics['optimal_f1_threshold']

print(f"\nOriginal (threshold={orig_opt['threshold']}):")
print(f"  Precision: {orig_opt['precision']:.1%}")
print(f"  Recall:    {orig_opt['recall']:.1%}")
print(f"  F1:        {orig_opt['f1']:.1%}")
print(f"  FP: {orig_opt['fp']}, FN: {orig_opt['fn']}")

print(f"\nHard Negative (threshold={hn_opt['threshold']}):")
print(f"  Precision: {hn_opt['precision']:.1%}")
print(f"  Recall:    {hn_opt['recall']:.1%}")
print(f"  F1:        {hn_opt['f1']:.1%}")
print(f"  FP: {hn_opt['fp']}, FN: {hn_opt['fn']}")

print(f"\nImprovement:")
print(f"  Precision: {(hn_opt['precision'] - orig_opt['precision'])*100:+.1f}%")
print(f"  Recall:    {(hn_opt['recall'] - orig_opt['recall'])*100:+.1f}%")
print(f"  F1:        {(hn_opt['f1'] - orig_opt['f1'])*100:+.1f}%")

print("\n" + "=" * 80)
print("HARD FALSE POSITIVES ANALYSIS")
print("=" * 80)

orig_hard_fp_ids = set(orig_hard_fp['photo_id'].values)
hn_hard_fp_ids = set(hn_hard_fp['photo_id'].values)

print(f"\nOriginal hard FP: {len(orig_hard_fp_ids)} photos")
print(f"  Examples: {list(orig_hard_fp_ids)[:10]}")

print(f"\nHard Negative hard FP: {len(hn_hard_fp_ids)} photos")
print(f"  Examples: {list(hn_hard_fp_ids)}")

# Photos that were fixed
fixed_photos = orig_hard_fp_ids - hn_hard_fp_ids
print(f"\n✅ Fixed (no longer FP): {len(fixed_photos)} photos")
if len(fixed_photos) > 0:
    print(f"  Examples: {list(fixed_photos)[:10]}")

# Photos still problematic
still_fp = orig_hard_fp_ids & hn_hard_fp_ids
print(f"\n⚠️  Still FP: {len(still_fp)} photos")
if len(still_fp) > 0:
    print(f"  Examples: {list(still_fp)}")

# New FP
new_fp = hn_hard_fp_ids - orig_hard_fp_ids
print(f"\n🆕 New FP: {len(new_fp)} photos")
if len(new_fp) > 0:
    print(f"  Examples: {list(new_fp)}")

print("\n" + "=" * 80)
print("DETAILED PHOTO COMPARISON")
print("=" * 80)

# Merge predictions
orig_pred = orig_predictions[['photo_id', 'y_true', 'prob_mean']].copy()
orig_pred.columns = ['photo_id', 'y_true', 'prob_orig']

hn_pred = hn_predictions[['photo_id', 'y_true', 'prob_mean']].copy()
hn_pred.columns = ['photo_id', 'y_true', 'prob_hn']

comparison = orig_pred.merge(hn_pred, on=['photo_id', 'y_true'], how='outer')
comparison['prob_diff'] = comparison['prob_hn'] - comparison['prob_orig']

# Focus on original hard FP
hard_fp_comparison = comparison[comparison['photo_id'].isin(orig_hard_fp_ids)].copy()
hard_fp_comparison = hard_fp_comparison.sort_values('prob_diff', ascending=False)

print(f"\nOriginal hard FP photos - probability changes:")
print(f"\nTop 10 most improved (prob decreased = good):")
for _, row in hard_fp_comparison.head(10).iterrows():
    print(f"  {row['photo_id']}: {row['prob_orig']:.3f} → {row['prob_hn']:.3f} ({row['prob_diff']:+.3f})")

print("\n" + "=" * 80)
print("DATA LEAKAGE INDICATORS")
print("=" * 80)

# Check for suspicious patterns
n_photos_orig = orig_metrics['n_photos']
n_photos_hn = hn_metrics['n_photos']

print(f"\nTest set size:")
print(f"  Original: {n_photos_orig} photos")
print(f"  Hard Neg: {n_photos_hn} photos")

if n_photos_orig != n_photos_hn:
    print(f"\n🚨 WARNING: Test set size changed!")
    print(f"  Difference: {n_photos_hn - n_photos_orig:+d} photos")
    print(f"  This suggests different split was used → DATA LEAKAGE!")
else:
    print(f"\n✅ Test set size unchanged (good sign)")

# Check if improvement is realistic
f1_improvement = (hn_opt['f1'] - orig_opt['f1']) * 100
if f1_improvement > 10:
    print(f"\n🚨 WARNING: F1 improvement is very high ({f1_improvement:+.1f}%)")
    print(f"  Improvements >10% are suspicious and may indicate data leakage")
elif f1_improvement > 5:
    print(f"\n⚠️  CAUTION: F1 improvement is moderate ({f1_improvement:+.1f}%)")
    print(f"  This is plausible but should be verified")
else:
    print(f"\n✅ F1 improvement is realistic ({f1_improvement:+.1f}%)")

# Check if hard FP were actually fixed
fix_rate = len(fixed_photos) / len(orig_hard_fp_ids) * 100
print(f"\nHard FP fix rate: {fix_rate:.1f}%")
if fix_rate > 80:
    print(f"  🚨 Very high fix rate - suspicious!")
elif fix_rate > 60:
    print(f"  ⚠️  High fix rate - verify manually")
else:
    print(f"  ✅ Moderate fix rate - plausible")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if n_photos_orig != n_photos_hn:
    print("\n🚨 LIKELY DATA LEAKAGE: Test set size changed")
    print("   → Different split was used")
    print("   → Results are NOT valid")
elif f1_improvement > 10 and fix_rate > 80:
    print("\n🚨 LIKELY DATA LEAKAGE: Improvement too high + fix rate too high")
    print("   → Results seem unrealistic")
    print("   → Verify split strategy was consistent")
elif f1_improvement > 5:
    print("\n⚠️  POSSIBLE DATA LEAKAGE: Improvement is high")
    print("   → Results should be verified manually")
    print("   → Check if split strategy was consistent")
else:
    print("\n✅ LIKELY NO DATA LEAKAGE: Improvement is realistic")
    print("   → Results appear valid")
