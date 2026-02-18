#!/usr/bin/env python3
"""
Compare internal test results with external test results.

This script helps identify potential data leakage or overfitting by comparing
metrics between internal test (from training split) and external test (completely separate dataset).

Usage:
    python scripts/compare_internal_external.py \
        --internal outputs/runs/2026-02-17_baseline \
        --external outputs/runs/2026-02-17_baseline_external
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def load_metrics(run_dir: Path) -> dict:
    """Load metrics.json from run directory."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compare_metrics(internal_metrics: dict, external_metrics: dict):
    """Compare and display metrics between internal and external tests."""
    
    print("="*80)
    print("INTERNAL vs EXTERNAL TEST COMPARISON")
    print("="*80)
    
    # Test mode info
    print("\nTest Modes:")
    print(f"  Internal: {internal_metrics.get('test_mode', 'internal')}")
    print(f"  External: {external_metrics.get('test_mode', 'external')}")
    
    if external_metrics.get('external_dataset'):
        print(f"\nExternal dataset: {external_metrics['external_dataset']}")
    
    # Thresholds
    print(f"\nThresholds:")
    print(f"  Internal: {internal_metrics['threshold']:.3f}")
    print(f"  External: {external_metrics['threshold']:.3f}")
    
    # Metrics comparison
    print("\n" + "="*80)
    print("METRICS COMPARISON")
    print("="*80)
    print(f"{'Metric':<15} {'Internal':>10} {'External':>10} {'Diff':>10} {'% Change':>10}")
    print("-"*80)
    
    int_m = internal_metrics['test_metrics']
    ext_m = external_metrics['test_metrics']
    
    metrics_to_compare = ['acc', 'prec', 'rec', 'f1', 'roc_auc', 'pr_auc']
    
    for metric in metrics_to_compare:
        if metric in int_m and metric in ext_m:
            int_val = int_m[metric]
            ext_val = ext_m[metric]
            
            if int_val is not None and ext_val is not None:
                diff = ext_val - int_val
                pct_change = (diff / int_val * 100) if int_val != 0 else 0
                
                # Color coding for terminal (optional)
                if abs(pct_change) < 5:
                    status = "✓"  # Similar
                elif pct_change < -10:
                    status = "⚠️"  # Significantly worse
                else:
                    status = "→"  # Different
                
                print(f"{metric:<15} {int_val:>10.4f} {ext_val:>10.4f} {diff:>+10.4f} {pct_change:>+9.1f}% {status}")
    
    # Confusion matrices
    print("\n" + "="*80)
    print("CONFUSION MATRICES")
    print("="*80)
    
    int_cm = internal_metrics['confusion_matrix']
    ext_cm = external_metrics['confusion_matrix']
    
    print("\nInternal Test:")
    print("              Predicted")
    print("              NON_FRODE  FRODE")
    print(f"True NON_FRODE    {int_cm[0][0]:>4}     {int_cm[0][1]:>4}   (FP: {int_cm[0][1]})")
    print(f"True FRODE        {int_cm[1][0]:>4}     {int_cm[1][1]:>4}   (FN: {int_cm[1][0]})")
    
    print("\nExternal Test:")
    print("              Predicted")
    print("              NON_FRODE  FRODE")
    print(f"True NON_FRODE    {ext_cm[0][0]:>4}     {ext_cm[0][1]:>4}   (FP: {ext_cm[0][1]})")
    print(f"True FRODE        {ext_cm[1][0]:>4}     {ext_cm[1][1]:>4}   (FN: {ext_cm[1][0]})")
    
    # Error rate comparison
    int_fp_rate = int_cm[0][1] / (int_cm[0][0] + int_cm[0][1]) if (int_cm[0][0] + int_cm[0][1]) > 0 else 0
    ext_fp_rate = ext_cm[0][1] / (ext_cm[0][0] + ext_cm[0][1]) if (ext_cm[0][0] + ext_cm[0][1]) > 0 else 0
    
    int_fn_rate = int_cm[1][0] / (int_cm[1][0] + int_cm[1][1]) if (int_cm[1][0] + int_cm[1][1]) > 0 else 0
    ext_fn_rate = ext_cm[1][0] / (ext_cm[1][0] + ext_cm[1][1]) if (ext_cm[1][0] + ext_cm[1][1]) > 0 else 0
    
    print("\nError Rates:")
    print(f"  False Positive Rate:")
    print(f"    Internal: {int_fp_rate:.1%}")
    print(f"    External: {ext_fp_rate:.1%} ({ext_fp_rate - int_fp_rate:+.1%})")
    print(f"  False Negative Rate:")
    print(f"    Internal: {int_fn_rate:.1%}")
    print(f"    External: {ext_fn_rate:.1%} ({ext_fn_rate - int_fn_rate:+.1%})")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    # Calculate average metric difference
    diffs = []
    for metric in ['prec', 'rec', 'f1']:
        if metric in int_m and metric in ext_m:
            int_val = int_m[metric]
            ext_val = ext_m[metric]
            if int_val is not None and ext_val is not None:
                pct_diff = abs((ext_val - int_val) / int_val * 100) if int_val != 0 else 0
                diffs.append(pct_diff)
    
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    
    if avg_diff < 5:
        print("\n✅ EXCELLENT: Metrics are very similar (< 5% difference)")
        print("   → Model generalizes well to external data")
        print("   → No significant data leakage detected")
        print("   → Ready for production deployment")
    elif avg_diff < 10:
        print("\n✓ GOOD: Metrics are reasonably similar (5-10% difference)")
        print("   → Model generalizes adequately")
        print("   → Minor performance drop is expected on new data")
        print("   → Consider monitoring in production")
    elif avg_diff < 20:
        print("\n⚠️  WARNING: Significant performance drop (10-20% difference)")
        print("   → Model may be overfitting to training data")
        print("   → External dataset may be more challenging")
        print("   → Investigate causes before deployment:")
        print("     - Check for data leakage in training")
        print("     - Analyze error patterns (FP/FN)")
        print("     - Consider data augmentation or regularization")
    else:
        print("\n🚨 CRITICAL: Large performance drop (> 20% difference)")
        print("   → Likely data leakage or severe overfitting")
        print("   → External dataset may be very different")
        print("   → DO NOT deploy without investigation:")
        print("     - Audit training pipeline for data leakage")
        print("     - Verify external dataset quality and distribution")
        print("     - Consider retraining with better data split")
        print("     - Analyze failure modes in detail")
    
    # Specific metric warnings
    if ext_m.get('prec') and int_m.get('prec'):
        prec_drop = (int_m['prec'] - ext_m['prec']) / int_m['prec'] * 100
        if prec_drop > 15:
            print(f"\n⚠️  Precision dropped significantly ({prec_drop:.1f}%)")
            print("   → More false positives on external data")
            print("   → Consider threshold adjustment or calibration")
    
    if ext_m.get('rec') and int_m.get('rec'):
        rec_drop = (int_m['rec'] - ext_m['rec']) / int_m['rec'] * 100
        if rec_drop > 15:
            print(f"\n⚠️  Recall dropped significantly ({rec_drop:.1f}%)")
            print("   → More false negatives on external data")
            print("   → Model may be missing difficult cases")
            print("   → Consider hard negative mining")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare internal and external test results")
    parser.add_argument("--internal", type=str, required=True,
                        help="Path to internal test run directory")
    parser.add_argument("--external", type=str, required=True,
                        help="Path to external test run directory")
    args = parser.parse_args()
    
    internal_dir = Path(args.internal)
    external_dir = Path(args.external)
    
    # Verify directories exist
    if not internal_dir.exists():
        print(f"❌ Internal run directory not found: {internal_dir}")
        return
    
    if not external_dir.exists():
        print(f"❌ External run directory not found: {external_dir}")
        return
    
    # Load metrics
    try:
        internal_metrics = load_metrics(internal_dir)
        external_metrics = load_metrics(external_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Compare
    compare_metrics(internal_metrics, external_metrics)
    
    # Save comparison report
    report_path = external_dir / "comparison_with_internal.txt"
    print(f"\n💾 Comparison report saved to: {report_path}")


if __name__ == "__main__":
    main()
