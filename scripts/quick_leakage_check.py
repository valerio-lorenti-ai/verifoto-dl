#!/usr/bin/env python3
"""Quick check for data leakage"""
import pandas as pd
import json

# Paths - update these to match your runs
ORIG_RUN = "outputs/runs/2026-02-23_convnext_v8.1_domainAware"
HN_RUN = "outputs/runs/2026-02-23_convnext_v8.1_domainAware_hard_negative"

print("="*80)
print("QUICK DATA LEAKAGE CHECK")
print("="*80)

# Load predictions
try:
    orig_pred = pd.read_csv(f"{ORIG_RUN}/photo_level_predictions.csv")
    hn_pred = pd.read_csv(f"{HN_RUN}/photo_level_predictions.csv")
    
    orig_photos = set(orig_pred['photo_id'].values)
    hn_photos = set(hn_pred['photo_id'].values)
    
    print(f"\nOriginal test: {len(orig_photos)} photos")
    print(f"Hard neg test: {len(hn_photos)} photos")
    
    overlap = orig_photos & hn_photos
    missing = orig_photos - hn_photos
    new = hn_photos - orig_photos
    
    print(f"\nOverlap: {len(overlap)}/{len(orig_photos)} ({len(overlap)/len(orig_photos)*100:.1f}%)")
    print(f"Missing: {len(missing)}")
    print(f"New: {len(new)}")
    
    # Load metrics
    with open(f"{ORIG_RUN}/photo_level_metrics.json") as f:
        orig_metrics = json.load(f)
    with open(f"{HN_RUN}/photo_level_metrics.json") as f:
        hn_metrics = json.load(f)
    
    orig_opt = orig_metrics['optimal_f1_threshold']
    hn_opt = hn_metrics['optimal_f1_threshold']
    
    print(f"\n{'='*80}")
    print("METRICS COMPARISON")
    print("="*80)
    
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
    
    print(f"\n{'='*80}")
    print("VERDICT")
    print("="*80)
    
    if len(overlap) == len(orig_photos):
        print("\nOK: Test sets are IDENTICAL (100% overlap)")
        print("   No data leakage detected")
        
        f1_improvement = (hn_opt['f1'] - orig_opt['f1']) * 100
        if f1_improvement > 10:
            print(f"\n   WARNING: F1 improvement is high ({f1_improvement:+.1f}%)")
            print("   This is suspicious but test sets are identical")
        elif f1_improvement > 5:
            print(f"\n   OK: F1 improvement is moderate ({f1_improvement:+.1f}%)")
            print("   This is plausible")
        else:
            print(f"\n   OK: F1 improvement is realistic ({f1_improvement:+.1f}%)")
    else:
        print(f"\nERROR: Test sets are DIFFERENT ({len(overlap)/len(orig_photos)*100:.1f}% overlap)")
        print("   DATA LEAKAGE DETECTED!")
        print(f"   {len(missing)} photos moved from test set")
        
except FileNotFoundError as e:
    print(f"\nERROR: File not found: {e}")
    print("Make sure the run directories exist")
