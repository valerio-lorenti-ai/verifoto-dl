"""
Analizza performance GPT-1.5 con threshold specifico.
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json


def analyze_gpt15_with_threshold(run_dir: Path, threshold: float):
    """Analizza GPT-1.5 con threshold specifico"""
    
    # Carica predictions
    pred_file = run_dir / "predictions.csv"
    if not pred_file.exists():
        print(f"❌ File not found: {pred_file}")
        return
    
    pred = pd.read_csv(pred_file)
    
    # Filtra solo GPT-1.5
    gpt15 = pred[pred['generator'] == 'gpt_image_1_5'].copy()
    
    if len(gpt15) == 0:
        print("❌ No GPT-1.5 samples found")
        return
    
    print("="*80)
    print(f"GPT-1.5 ANALYSIS WITH THRESHOLD {threshold:.2f}")
    print("="*80)
    
    # Applica threshold
    gpt15['y_pred_new'] = (gpt15['y_prob'] >= threshold).astype(int)
    
    # Calcola metriche
    tp = ((gpt15['y_true'] == 1) & (gpt15['y_pred_new'] == 1)).sum()
    fp = ((gpt15['y_true'] == 0) & (gpt15['y_pred_new'] == 1)).sum()
    tn = ((gpt15['y_true'] == 0) & (gpt15['y_pred_new'] == 0)).sum()
    fn = ((gpt15['y_true'] == 1) & (gpt15['y_pred_new'] == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(gpt15)
    
    print(f"\nTotal GPT-1.5 samples: {len(gpt15)}")
    print(f"  Real (y=0): {(gpt15['y_true'] == 0).sum()}")
    print(f"  AI (y=1):   {(gpt15['y_true'] == 1).sum()}")
    
    print(f"\nMetrics with threshold {threshold:.2f}:")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1:        {f1:.1%}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:>3}  FP={fp:>3}")
    print(f"  FN={fn:>3}  TP={tp:>3}")
    
    # Confronto con threshold originale
    if 'y_pred' in gpt15.columns:
        tp_orig = ((gpt15['y_true'] == 1) & (gpt15['y_pred'] == 1)).sum()
        fn_orig = ((gpt15['y_true'] == 1) & (gpt15['y_pred'] == 0)).sum()
        recall_orig = tp_orig / (tp_orig + fn_orig) if (tp_orig + fn_orig) > 0 else 0
        
        print(f"\nComparison with original threshold:")
        print(f"  Original recall: {recall_orig:.1%}")
        print(f"  New recall:      {recall:.1%}")
        print(f"  Improvement:     {recall - recall_orig:+.1%}")
    
    # Falsi negativi
    fn_samples = gpt15[(gpt15['y_true'] == 1) & (gpt15['y_pred_new'] == 0)]
    
    if len(fn_samples) > 0:
        print(f"\n⚠️  False Negatives: {len(fn_samples)}")
        print(f"\nTop 10 False Negatives (lowest probability):")
        print(f"{'Prob':>6} {'Defect':<15} {'Food':<20} {'Path'}")
        print("-"*80)
        
        fn_sorted = fn_samples.sort_values('y_prob')
        for _, row in fn_sorted.head(10).iterrows():
            prob = row['y_prob']
            defect = str(row.get('defect_type', 'N/A'))[:14]
            food = str(row.get('food_category', 'N/A'))[:19]
            path = Path(row['path']).name
            print(f"{prob:>6.3f} {defect:<15} {food:<20} {path}")
        
        # Statistiche FN
        print(f"\nFalse Negative Statistics:")
        print(f"  Mean probability: {fn_samples['y_prob'].mean():.3f}")
        print(f"  Median probability: {fn_samples['y_prob'].median():.3f}")
        print(f"  Min probability: {fn_samples['y_prob'].min():.3f}")
        print(f"  Max probability: {fn_samples['y_prob'].max():.3f}")
        
        # Breakdown per defect
        if 'defect_type' in fn_samples.columns:
            print(f"\nFalse Negatives by Defect Type:")
            defect_counts = fn_samples['defect_type'].value_counts()
            for defect, count in defect_counts.items():
                pct = count / len(fn_samples) * 100
                print(f"  {defect}: {count} ({pct:.1f}%)")
    else:
        print(f"\n✅ No False Negatives! Perfect recall!")
    
    # Raccomandazione
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if recall >= 0.85:
        print(f"✅ Recall {recall:.1%} is GOOD (≥85%)")
        print(f"   No fine-tuning needed for GPT-1.5")
        print(f"   Ready for production with threshold {threshold:.2f}")
    elif recall >= 0.80:
        print(f"⚠️  Recall {recall:.1%} is ACCEPTABLE (80-85%)")
        print(f"   Consider fine-tuning if you need higher recall")
        print(f"   Or accept current performance")
    else:
        print(f"❌ Recall {recall:.1%} is LOW (<80%)")
        print(f"   Fine-tuning on GPT-1.5 is RECOMMENDED")
        print(f"   Expected improvement: {recall:.1%} → 90-95%")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze GPT-1.5 with specific threshold")
    parser.add_argument("run_name", type=str, help="Run name (e.g., 2026-02-20_convnext_v8_domaiAware)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Threshold to use (default: 0.55)")
    args = parser.parse_args()
    
    # Trova run directory
    run_dir = Path("outputs/runs") / args.run_name
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)
    
    print(f"\nAnalyzing run: {args.run_name}")
    print(f"Directory: {run_dir}\n")
    
    analyze_gpt15_with_threshold(run_dir, args.threshold)


if __name__ == "__main__":
    main()
