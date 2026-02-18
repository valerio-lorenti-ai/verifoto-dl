#!/usr/bin/env python3
"""
Analisi delle performance a livello di FOTO (photo_id) invece che per singola immagine.

Con 8 versioni per foto, le metriche per-immagine amplificano gli errori.
Questa analisi aggrega le versioni per ottenere metriche più realistiche.

Usage:
    python scripts/analyze_by_photo.py --run outputs/runs/2026-02-17_noK2_noLeakage
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, average_precision_score
)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available, skipping plots")


def extract_photo_id(path: str) -> str:
    """Estrae ID univoco usando i PRIMI 4 CARATTERI del filename."""
    filename = Path(path).stem
    return filename[:4]


def aggregate_by_photo(predictions_df: pd.DataFrame, threshold: float = 0.5, 
                       aggregation: str = 'mean') -> pd.DataFrame:
    """
    Aggrega le predizioni per photo_id.
    
    Args:
        predictions_df: DataFrame con colonne path, y_true, y_prob, y_pred
        threshold: soglia per classificazione
        aggregation: 'mean', 'median', 'max', 'voting'
    
    Returns:
        DataFrame aggregato per photo_id
    """
    df = predictions_df.copy()
    df['photo_id'] = df['path'].apply(extract_photo_id)
    
    # Verifica label consistency
    label_check = df.groupby('photo_id')['y_true'].nunique()
    inconsistent = label_check[label_check > 1]
    
    if len(inconsistent) > 0:
        print(f"\n⚠️  WARNING: {len(inconsistent)} photos have inconsistent labels!")
        print(f"   This indicates a bug in dataset construction.")
        print(f"   Examples: {list(inconsistent.index[:5])}")
    
    # Aggregazione
    if aggregation == 'mean':
        agg_func = 'mean'
    elif aggregation == 'median':
        agg_func = 'median'
    elif aggregation == 'max':
        agg_func = 'max'
    elif aggregation == 'voting':
        # Majority voting sulle predizioni
        agg_func = lambda x: (x >= threshold).sum() / len(x)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # Aggrega per photo_id
    photo_df = df.groupby('photo_id').agg({
        'y_true': 'first',  # Assume label consistente
        'y_prob': agg_func if aggregation != 'voting' else 'mean',
        'y_pred': 'mean' if aggregation == 'voting' else 'first',  # Placeholder
        'source': 'first',
        'quality': 'first',
        'food_category': 'first',
        'defect_type': 'first',
        'generator': 'first'
    }).reset_index()
    
    # Calcola statistiche per versione
    version_stats = df.groupby('photo_id')['y_prob'].agg([
        ('prob_mean', 'mean'),
        ('prob_std', 'std'),
        ('prob_min', 'min'),
        ('prob_max', 'max'),
        ('n_versions', 'size')
    ]).reset_index()
    
    photo_df = photo_df.merge(version_stats, on='photo_id')
    
    # Ricalcola predizione con threshold
    if aggregation == 'voting':
        # Voting: maggioranza delle versioni
        voting_results = df.groupby('photo_id')['y_pred'].apply(
            lambda x: 1 if x.sum() > len(x) / 2 else 0
        ).reset_index()
        photo_df['y_pred'] = voting_results['y_pred']
    else:
        # Threshold sulla probabilità aggregata
        photo_df['y_pred'] = (photo_df['y_prob'] >= threshold).astype(int)
    
    return photo_df


def compute_photo_metrics(photo_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Calcola metriche a livello di foto."""
    y_true = photo_df['y_true'].values
    y_prob = photo_df['y_prob'].values
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'n_photos': len(photo_df),
        'n_positive': (y_true == 1).sum(),
        'n_negative': (y_true == 0).sum(),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None,
        'pr_auc': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics['confusion_matrix'] = cm.tolist()
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])
    
    return metrics


def find_threshold_optimal(photo_df: pd.DataFrame, metric: str = 'f1', 
                          min_recall: float = None) -> tuple:
    """
    Trova threshold ottimale per massimizzare una metrica.
    
    Args:
        photo_df: DataFrame aggregato per photo
        metric: 'f1', 'precision', 'recall', 'accuracy'
        min_recall: se specificato, vincolo minimo su recall
    
    Returns:
        (best_threshold, best_score, sweep_results)
    """
    y_true = photo_df['y_true'].values
    y_prob = photo_df['y_prob'].values
    
    # Sweep thresholds
    thresholds = np.linspace(0.1, 0.9, 17)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        
        results.append({
            'threshold': thresh,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'fp': fp,
            'fn': fn
        })
    
    results_df = pd.DataFrame(results)
    
    # Applica vincolo recall se specificato
    if min_recall is not None:
        valid = results_df[results_df['recall'] >= min_recall]
        if len(valid) == 0:
            print(f"⚠️  No threshold achieves recall >= {min_recall}")
            valid = results_df
    else:
        valid = results_df
    
    # Trova best threshold
    best_idx = valid[metric].idxmax()
    best_row = valid.loc[best_idx]
    
    return best_row['threshold'], best_row[metric], results_df


def analyze_consistency(photo_df: pd.DataFrame) -> dict:
    """Analizza consistency delle predizioni tra versioni."""
    stats = {
        'avg_std': photo_df['prob_std'].mean(),
        'median_std': photo_df['prob_std'].median(),
        'max_std': photo_df['prob_std'].max(),
        'photos_std_lt_0.01': (photo_df['prob_std'] < 0.01).sum(),
        'photos_std_lt_0.02': (photo_df['prob_std'] < 0.02).sum(),
        'photos_std_lt_0.05': (photo_df['prob_std'] < 0.05).sum(),
        'pct_std_lt_0.05': (photo_df['prob_std'] < 0.05).sum() / len(photo_df) * 100
    }
    
    return stats


def find_hard_cases(photo_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Identifica foto con errori sistematici."""
    photo_df = photo_df.copy()
    photo_df['y_pred'] = (photo_df['y_prob'] >= threshold).astype(int)
    
    # Falsi positivi
    fp_photos = photo_df[(photo_df['y_true'] == 0) & (photo_df['y_pred'] == 1)]
    fp_hard = fp_photos.sort_values('y_prob', ascending=False)
    
    # Falsi negativi
    fn_photos = photo_df[(photo_df['y_true'] == 1) & (photo_df['y_pred'] == 0)]
    fn_hard = fn_photos.sort_values('y_prob', ascending=True)
    
    # Foto con alta varianza (inconsistent predictions)
    high_variance = photo_df.nlargest(20, 'prob_std')
    
    # Foto con bassa varianza (perfect consistency)
    low_variance = photo_df.nsmallest(20, 'prob_std')
    
    return {
        'false_positives': fp_hard,
        'false_negatives': fn_hard,
        'high_variance': high_variance,
        'low_variance': low_variance
    }


def plot_threshold_sweep(sweep_df: pd.DataFrame, save_path: Path):
    """Plot threshold sweep results."""
    if not HAS_MATPLOTLIB:
        print("⚠️  Skipping plot (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Precision, Recall, F1
    ax = axes[0, 0]
    ax.plot(sweep_df['threshold'], sweep_df['precision'], label='Precision', marker='o')
    ax.plot(sweep_df['threshold'], sweep_df['recall'], label='Recall', marker='s')
    ax.plot(sweep_df['threshold'], sweep_df['f1'], label='F1', marker='^')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, F1 vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(sweep_df['threshold'], sweep_df['accuracy'], marker='o', color='green')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Threshold')
    ax.grid(True, alpha=0.3)
    
    # FP, FN
    ax = axes[1, 0]
    ax.plot(sweep_df['threshold'], sweep_df['fp'], label='False Positives', marker='o', color='red')
    ax.plot(sweep_df['threshold'], sweep_df['fn'], label='False Negatives', marker='s', color='orange')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Count')
    ax.set_title('False Positives and False Negatives vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision-Recall curve
    ax = axes[1, 1]
    ax.plot(sweep_df['recall'], sweep_df['precision'], marker='o')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Trade-off')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved threshold sweep plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze predictions at photo level")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    parser.add_argument("--aggregation", type=str, default="mean", 
                       choices=['mean', 'median', 'max', 'voting'],
                       help="Aggregation method for probabilities")
    parser.add_argument("--min-recall", type=float, default=None,
                       help="Minimum recall constraint for threshold selection")
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    predictions_csv = run_dir / "predictions.csv"
    
    if not predictions_csv.exists():
        print(f"Error: {predictions_csv} not found")
        return 1
    
    print("="*80)
    print("PHOTO-LEVEL ANALYSIS")
    print("="*80)
    print(f"Run: {run_dir.name}")
    print(f"Aggregation: {args.aggregation}")
    
    # Load predictions
    df = pd.read_csv(predictions_csv)
    print(f"\nImage-level: {len(df)} images")
    
    # Get original threshold from metrics.json
    metrics_json = run_dir / "metrics.json"
    if metrics_json.exists():
        with open(metrics_json, 'r') as f:
            metrics = json.load(f)
            original_threshold = metrics.get('threshold', 0.5)
    else:
        original_threshold = 0.5
    
    print(f"Original threshold: {original_threshold:.3f}")
    
    # Aggregate by photo
    print(f"\n{'='*80}")
    print("AGGREGATING BY PHOTO")
    print("="*80)
    
    photo_df = aggregate_by_photo(df, threshold=original_threshold, 
                                   aggregation=args.aggregation)
    
    print(f"Photo-level: {len(photo_df)} unique photos")
    print(f"  Positive: {(photo_df['y_true'] == 1).sum()}")
    print(f"  Negative: {(photo_df['y_true'] == 0).sum()}")
    print(f"  Avg versions per photo: {photo_df['n_versions'].mean():.2f}")
    
    # Compute metrics with original threshold
    print(f"\n{'='*80}")
    print(f"METRICS AT ORIGINAL THRESHOLD ({original_threshold:.3f})")
    print("="*80)
    
    metrics_original = compute_photo_metrics(photo_df, threshold=original_threshold)
    
    print(f"\nPhoto-level metrics:")
    print(f"  Accuracy:  {metrics_original['accuracy']:.4f}")
    print(f"  Precision: {metrics_original['precision']:.4f}")
    print(f"  Recall:    {metrics_original['recall']:.4f}")
    print(f"  F1:        {metrics_original['f1']:.4f}")
    if metrics_original['roc_auc']:
        print(f"  ROC-AUC:   {metrics_original['roc_auc']:.4f}")
    if metrics_original['pr_auc']:
        print(f"  PR-AUC:    {metrics_original['pr_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN={metrics_original['tn']}  FP={metrics_original['fp']}")
    print(f"  FN={metrics_original['fn']}  TP={metrics_original['tp']}")
    
    # Threshold sweep
    print(f"\n{'='*80}")
    print("THRESHOLD OPTIMIZATION")
    print("="*80)
    
    best_f1_thresh, best_f1, sweep_df = find_threshold_optimal(
        photo_df, metric='f1', min_recall=args.min_recall
    )
    
    print(f"\nOptimal threshold (F1): {best_f1_thresh:.3f}")
    print(f"  F1: {best_f1:.4f}")
    
    # Compute metrics at optimal threshold
    metrics_optimal = compute_photo_metrics(photo_df, threshold=best_f1_thresh)
    print(f"  Accuracy:  {metrics_optimal['accuracy']:.4f}")
    print(f"  Precision: {metrics_optimal['precision']:.4f}")
    print(f"  Recall:    {metrics_optimal['recall']:.4f}")
    print(f"  FP: {metrics_optimal['fp']}, FN: {metrics_optimal['fn']}")
    
    # Try precision-optimized threshold
    best_prec_thresh, best_prec, _ = find_threshold_optimal(
        photo_df, metric='precision', min_recall=0.90
    )
    
    print(f"\nPrecision-optimized threshold (recall >= 0.90): {best_prec_thresh:.3f}")
    metrics_prec = compute_photo_metrics(photo_df, threshold=best_prec_thresh)
    print(f"  Precision: {metrics_prec['precision']:.4f}")
    print(f"  Recall:    {metrics_prec['recall']:.4f}")
    print(f"  F1:        {metrics_prec['f1']:.4f}")
    print(f"  FP: {metrics_prec['fp']}, FN: {metrics_prec['fn']}")
    
    # Consistency analysis
    print(f"\n{'='*80}")
    print("CONSISTENCY ANALYSIS")
    print("="*80)
    
    consistency = analyze_consistency(photo_df)
    print(f"\nProbability std across versions:")
    print(f"  Mean:   {consistency['avg_std']:.4f}")
    print(f"  Median: {consistency['median_std']:.4f}")
    print(f"  Max:    {consistency['max_std']:.4f}")
    print(f"\nPhotos with low variance:")
    print(f"  std < 0.01: {consistency['photos_std_lt_0.01']} ({consistency['photos_std_lt_0.01']/len(photo_df)*100:.1f}%)")
    print(f"  std < 0.02: {consistency['photos_std_lt_0.02']} ({consistency['photos_std_lt_0.02']/len(photo_df)*100:.1f}%)")
    print(f"  std < 0.05: {consistency['photos_std_lt_0.05']} ({consistency['pct_std_lt_0.05']:.1f}%)")
    
    # Hard cases
    print(f"\n{'='*80}")
    print("HARD CASES")
    print("="*80)
    
    hard_cases = find_hard_cases(photo_df, threshold=best_f1_thresh)
    
    print(f"\nFalse Positives (top 10):")
    for i, row in hard_cases['false_positives'].head(10).iterrows():
        print(f"  {row['photo_id']}: prob={row['y_prob']:.3f}, std={row['prob_std']:.4f}, "
              f"food={row['food_category']}, quality={row['quality']}")
    
    print(f"\nFalse Negatives (top 10):")
    for i, row in hard_cases['false_negatives'].head(10).iterrows():
        print(f"  {row['photo_id']}: prob={row['y_prob']:.3f}, std={row['prob_std']:.4f}, "
              f"defect={row['defect_type']}, gen={row['generator']}")
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("="*80)
    
    # Photo-level predictions
    photo_predictions_path = run_dir / "photo_level_predictions.csv"
    photo_df.to_csv(photo_predictions_path, index=False)
    print(f"✓ Saved: {photo_predictions_path}")
    
    # Photo-level metrics
    photo_metrics = {
        'aggregation': args.aggregation,
        'n_photos': int(len(photo_df)),
        'original_threshold': {
            'threshold': float(original_threshold),
            **{k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
               for k, v in metrics_original.items() if k != 'confusion_matrix'}
        },
        'optimal_f1_threshold': {
            'threshold': float(best_f1_thresh),
            **{k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
               for k, v in metrics_optimal.items() if k != 'confusion_matrix'}
        },
        'optimal_precision_threshold': {
            'threshold': float(best_prec_thresh),
            **{k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
               for k, v in metrics_prec.items() if k != 'confusion_matrix'}
        },
        'consistency': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                       for k, v in consistency.items()}
    }
    
    photo_metrics_path = run_dir / "photo_level_metrics.json"
    with open(photo_metrics_path, 'w') as f:
        json.dump(photo_metrics, f, indent=2)
    print(f"✓ Saved: {photo_metrics_path}")
    
    # Threshold sweep
    sweep_path = run_dir / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"✓ Saved: {sweep_path}")
    
    # Chosen threshold
    chosen_threshold = {
        'f1_optimal': best_f1_thresh,
        'precision_optimal_recall_90': best_prec_thresh,
        'recommendation': best_prec_thresh if metrics_prec['precision'] > 0.75 else best_f1_thresh,
        'rationale': 'Use precision-optimal if precision > 0.75, otherwise use F1-optimal'
    }
    
    chosen_path = run_dir / "chosen_threshold.json"
    with open(chosen_path, 'w') as f:
        json.dump(chosen_threshold, f, indent=2)
    print(f"✓ Saved: {chosen_path}")
    
    # Plot
    plot_path = run_dir / "threshold_sweep.png"
    plot_threshold_sweep(sweep_df, plot_path)
    
    # Hard cases CSVs
    hard_cases['false_positives'].to_csv(run_dir / "photo_hard_fp.csv", index=False)
    hard_cases['false_negatives'].to_csv(run_dir / "photo_hard_fn.csv", index=False)
    print(f"✓ Saved: photo_hard_fp.csv, photo_hard_fn.csv")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n📊 Photo-level metrics are more realistic than image-level")
    print(f"   (8 versions per photo amplify errors in image-level metrics)")
    
    print(f"\n🎯 Recommended threshold: {chosen_threshold['recommendation']:.3f}")
    print(f"   Expected performance:")
    
    rec_thresh = chosen_threshold['recommendation']
    rec_metrics = metrics_prec if rec_thresh == best_prec_thresh else metrics_optimal
    print(f"     Precision: {rec_metrics['precision']:.1%}")
    print(f"     Recall:    {rec_metrics['recall']:.1%}")
    print(f"     F1:        {rec_metrics['f1']:.1%}")
    print(f"     FP: {rec_metrics['fp']}, FN: {rec_metrics['fn']}")
    
    print(f"\n📈 Improvement from original threshold:")
    prec_improvement = (rec_metrics['precision'] - metrics_original['precision']) * 100
    f1_improvement = (rec_metrics['f1'] - metrics_original['f1']) * 100
    print(f"     Precision: {prec_improvement:+.1f}%")
    print(f"     F1:        {f1_improvement:+.1f}%")
    
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
