"""
Script per testare rapidamente diversi threshold su un modello già addestrato.
Utile per trovare il threshold ottimale senza ri-addestrare.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score


def compute_metrics_at_threshold(probs, y_true, threshold):
    """Calcola metriche a un dato threshold."""
    y_pred = (probs >= threshold).astype(int)
    
    acc = (y_pred == y_true).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate
    }


def main():
    parser = argparse.ArgumentParser(description="Test multiple thresholds on predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.csv")
    parser.add_argument("--thresholds", type=str, default="0.5,0.55,0.6,0.65,0.7,0.75,0.8", 
                       help="Comma-separated thresholds to test")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}")
    df = pd.read_csv(args.predictions)
    
    if 'y_true' not in df.columns or 'y_prob' not in df.columns:
        raise ValueError("predictions.csv must contain 'y_true' and 'y_prob' columns")
    
    y_true = df['y_true'].values
    y_prob = df['y_prob'].values
    
    # Parse thresholds
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    
    print(f"\nTesting {len(thresholds)} thresholds: {thresholds}")
    print(f"Dataset: {len(y_true)} samples ({(y_true == 1).sum()} positive, {(y_true == 0).sum()} negative)")
    
    # Compute metrics for each threshold
    results = []
    for thresh in thresholds:
        metrics = compute_metrics_at_threshold(y_prob, y_true, thresh)
        results.append(metrics)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\n" + "="*100)
    print("THRESHOLD COMPARISON")
    print("="*100)
    print(results_df.to_string(index=False))
    
    # Find best thresholds
    print("\n" + "="*100)
    print("BEST THRESHOLDS")
    print("="*100)
    
    best_f1_idx = results_df['f1'].idxmax()
    best_prec_idx = results_df['precision'].idxmax()
    best_rec_idx = results_df['recall'].idxmax()
    
    print(f"\nBest F1: {results_df.loc[best_f1_idx, 'f1']:.4f} at threshold {results_df.loc[best_f1_idx, 'threshold']:.2f}")
    print(f"  Precision: {results_df.loc[best_f1_idx, 'precision']:.4f}")
    print(f"  Recall: {results_df.loc[best_f1_idx, 'recall']:.4f}")
    print(f"  FP: {results_df.loc[best_f1_idx, 'fp']:.0f}, FN: {results_df.loc[best_f1_idx, 'fn']:.0f}")
    
    print(f"\nBest Precision: {results_df.loc[best_prec_idx, 'precision']:.4f} at threshold {results_df.loc[best_prec_idx, 'threshold']:.2f}")
    print(f"  F1: {results_df.loc[best_prec_idx, 'f1']:.4f}")
    print(f"  Recall: {results_df.loc[best_prec_idx, 'recall']:.4f}")
    print(f"  FP: {results_df.loc[best_prec_idx, 'fp']:.0f}, FN: {results_df.loc[best_prec_idx, 'fn']:.0f}")
    
    print(f"\nBest Recall: {results_df.loc[best_rec_idx, 'recall']:.4f} at threshold {results_df.loc[best_rec_idx, 'threshold']:.2f}")
    print(f"  F1: {results_df.loc[best_rec_idx, 'f1']:.4f}")
    print(f"  Precision: {results_df.loc[best_rec_idx, 'precision']:.4f}")
    print(f"  FP: {results_df.loc[best_rec_idx, 'fp']:.0f}, FN: {results_df.loc[best_rec_idx, 'fn']:.0f}")
    
    # Recommendations
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)
    
    # Find threshold with F1 > 0.65 and best precision
    good_f1 = results_df[results_df['f1'] >= 0.65]
    if len(good_f1) > 0:
        best_prec_with_good_f1 = good_f1.loc[good_f1['precision'].idxmax()]
        print(f"\n✅ Best threshold with F1 ≥ 0.65: {best_prec_with_good_f1['threshold']:.2f}")
        print(f"   F1: {best_prec_with_good_f1['f1']:.4f}")
        print(f"   Precision: {best_prec_with_good_f1['precision']:.4f}")
        print(f"   Recall: {best_prec_with_good_f1['recall']:.4f}")
        print(f"   FP: {best_prec_with_good_f1['fp']:.0f}, FN: {best_prec_with_good_f1['fn']:.0f}")
    else:
        print(f"\n⚠️  No threshold achieves F1 ≥ 0.65")
        print(f"   Best F1: {results_df.loc[best_f1_idx, 'f1']:.4f} at threshold {results_df.loc[best_f1_idx, 'threshold']:.2f}")
    
    # Find threshold with FP rate < 15%
    low_fp = results_df[results_df['fp_rate'] <= 0.15]
    if len(low_fp) > 0:
        best_f1_with_low_fp = low_fp.loc[low_fp['f1'].idxmax()]
        print(f"\n✅ Best threshold with FP rate ≤ 15%: {best_f1_with_low_fp['threshold']:.2f}")
        print(f"   F1: {best_f1_with_low_fp['f1']:.4f}")
        print(f"   Precision: {best_f1_with_low_fp['precision']:.4f}")
        print(f"   Recall: {best_f1_with_low_fp['recall']:.4f}")
        print(f"   FP rate: {best_f1_with_low_fp['fp_rate']:.2%}")
        print(f"   FP: {best_f1_with_low_fp['fp']:.0f}, FN: {best_f1_with_low_fp['fn']:.0f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
