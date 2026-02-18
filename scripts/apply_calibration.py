#!/usr/bin/env python3
"""
Apply temperature scaling calibration to an existing run.

This script:
1. Loads validation predictions to find optimal temperature
2. Applies calibration to test predictions
3. Re-computes metrics and threshold with calibrated probabilities
4. Saves calibration_T.json and updated results

Usage:
    python scripts/apply_calibration.py --run outputs/runs/2026-02-17_noK2_noLeakage
"""

import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.calibration import (
    find_optimal_temperature, apply_temperature_scaling,
    calibration_report
)


def load_validation_logits(run_dir: Path) -> tuple:
    """
    Load validation logits from checkpoint or recompute from probabilities.
    
    Note: If logits not saved, we approximate: logit = log(p / (1-p))
    """
    # Try to load from saved file
    val_logits_file = run_dir / "validation_logits.npy"
    
    if val_logits_file.exists():
        logits = np.load(val_logits_file)
        # Load corresponding labels
        val_preds = pd.read_csv(run_dir / "validation_predictions.csv")
        labels = val_preds['y_true'].values
        return logits, labels
    
    # Fallback: approximate from probabilities
    print("⚠️  Validation logits not found, approximating from probabilities")
    print("   (This is less accurate but should work)")
    
    val_preds = pd.read_csv(run_dir / "validation_predictions.csv")
    probs = val_preds['y_prob'].values
    labels = val_preds['y_true'].values
    
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs_clipped / (1 - probs_clipped))
    
    return logits, labels


def main():
    parser = argparse.ArgumentParser(description="Apply temperature scaling calibration")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    
    print("="*80)
    print("TEMPERATURE SCALING CALIBRATION")
    print("="*80)
    print(f"Run: {run_dir.name}\n")
    
    # Check if validation predictions exist
    val_preds_file = run_dir / "validation_predictions.csv"
    if not val_preds_file.exists():
        print(f"❌ Error: {val_preds_file} not found")
        print(f"   Validation predictions are needed for calibration")
        print(f"   Re-run training with validation predictions saved")
        return 1
    
    # Load validation data
    print("Loading validation data...")
    val_logits, val_labels = load_validation_logits(run_dir)
    print(f"  Validation samples: {len(val_labels)}")
    
    # Find optimal temperature
    print("\nOptimizing temperature...")
    optimal_T = find_optimal_temperature(val_logits, val_labels)
    
    # Save temperature
    calibration_T = {
        'temperature': float(optimal_T),
        'method': 'temperature_scaling',
        'optimized_on': 'validation_set',
        'metric': 'negative_log_likelihood'
    }
    
    calibration_file = run_dir / "calibration_T.json"
    with open(calibration_file, 'w') as f:
        json.dump(calibration_T, f, indent=2)
    print(f"\n✓ Saved: {calibration_file}")
    
    # Apply to test set
    print("\nApplying calibration to test set...")
    test_preds = pd.read_csv(run_dir / "predictions.csv")
    
    # Approximate test logits from probabilities
    test_probs_orig = test_preds['y_prob'].values
    test_probs_clipped = np.clip(test_probs_orig, 1e-7, 1 - 1e-7)
    test_logits = np.log(test_probs_clipped / (1 - test_probs_clipped))
    
    # Apply temperature scaling
    test_probs_calibrated = apply_temperature_scaling(test_logits, optimal_T)
    
    # Calibration report
    test_labels = test_preds['y_true'].values
    report = calibration_report(test_probs_orig, test_probs_calibrated, test_labels)
    
    print(f"\nCalibration Report:")
    print(f"  ECE before: {report['ece_before']:.4f}")
    print(f"  ECE after:  {report['ece_after']:.4f}")
    print(f"  Improvement: {report['ece_improvement']:.4f}")
    print(f"\n  Overconfident negatives (prob>0.95 but y=0):")
    print(f"    Before: {report['overconfident_negatives_before']}")
    print(f"    After:  {report['overconfident_negatives_after']}")
    print(f"    Reduction: {report['overconfident_negatives_before'] - report['overconfident_negatives_after']}")
    
    # Save calibrated predictions
    test_preds_calibrated = test_preds.copy()
    test_preds_calibrated['y_prob_original'] = test_probs_orig
    test_preds_calibrated['y_prob'] = test_probs_calibrated
    
    calibrated_preds_file = run_dir / "predictions_calibrated.csv"
    test_preds_calibrated.to_csv(calibrated_preds_file, index=False)
    print(f"\n✓ Saved: {calibrated_preds_file}")
    
    # Save calibration report
    report_file = run_dir / "calibration_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved: {report_file}")
    
    # Re-run photo-level analysis with calibrated probabilities
    print("\n" + "="*80)
    print("RE-RUNNING PHOTO-LEVEL ANALYSIS WITH CALIBRATED PROBABILITIES")
    print("="*80)
    
    import subprocess
    result = subprocess.run([
        sys.executable,
        "scripts/analyze_by_photo.py",
        "--run", str(run_dir),
        "--min-recall", "0.90"
    ], capture_output=False)
    
    if result.returncode == 0:
        print("\n✓ Photo-level analysis completed")
        print(f"  Check: {run_dir}/photo_level_metrics.json")
        print(f"  Check: {run_dir}/chosen_threshold.json")
    
    print("\n" + "="*80)
    print("CALIBRATION COMPLETE")
    print("="*80)
    print(f"\n📊 Next steps:")
    print(f"  1. Review calibration_report.json")
    print(f"  2. Check new chosen_threshold.json")
    print(f"  3. Use predictions_calibrated.csv for deployment")
    print(f"  4. Apply temperature T={optimal_T:.4f} in production inference")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
