#!/usr/bin/env python3
"""
Analizza data leakage dai risultati di un run già completato.
Verifica se le immagini nel test set hanno versioni nel training set.

Usage:
    python scripts/analyze_leakage_from_results.py --run outputs/runs/2026-02-17_noK2
"""

import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd


def extract_photo_id(path: str) -> str:
    """
    Estrae ID univoco usando i PRIMI 4 CARATTERI del filename.
    
    Esempi:
        .../1976_q95.jpg → 1976
        .../3cac_q50.jpg → 3cac
        .../dd44_highres_crop.jpg → dd44
    """
    filename = Path(path).stem
    return filename[:4]


def analyze_leakage_from_predictions(predictions_csv: Path):
    """
    Analizza data leakage dal file predictions.csv.
    
    NOTA: Questo script può solo STIMARE il leakage, perché predictions.csv
    contiene solo il test set. Per un'analisi completa serve accesso a train/val/test.
    """
    print("="*80)
    print("DATA LEAKAGE ANALYSIS FROM PREDICTIONS")
    print("="*80)
    print(f"File: {predictions_csv}")
    
    # Carica predictions
    df = pd.read_csv(predictions_csv)
    print(f"\nTest set: {len(df)} images")
    
    # Estrai photo_id
    df['photo_id'] = df['path'].apply(extract_photo_id)
    
    # Conta versioni per photo_id
    versions_per_photo = df.groupby('photo_id').size()
    
    print(f"\n📸 VERSIONS IN TEST SET")
    print(f"  Unique photos: {len(versions_per_photo)}")
    print(f"  Total images: {len(df)}")
    print(f"  Avg versions per photo: {versions_per_photo.mean():.2f}")
    print(f"  Min versions: {versions_per_photo.min()}")
    print(f"  Max versions: {versions_per_photo.max()}")
    
    # Trova foto con molte versioni (possibile indicatore di leakage)
    high_version_photos = versions_per_photo[versions_per_photo > 5]
    
    if len(high_version_photos) > 0:
        print(f"\n⚠️  Photos with >5 versions in test set: {len(high_version_photos)}")
        print(f"  This is unusual and may indicate data leakage")
        print(f"\n  Examples:")
        for photo_id, count in high_version_photos.head(10).items():
            print(f"    {photo_id}: {count} versions")
    
    # Analizza performance per photo_id
    print(f"\n📊 PERFORMANCE ANALYSIS")
    
    # Raggruppa per photo_id
    photo_stats = df.groupby('photo_id').agg({
        'y_true': 'first',  # Assume stesso label per tutte le versioni
        'y_pred': lambda x: (x.sum() / len(x)),  # % predizioni corrette
        'y_prob': 'mean'  # Probabilità media
    }).reset_index()
    
    photo_stats.columns = ['photo_id', 'y_true', 'pred_rate', 'avg_prob']
    
    # Calcola accuracy per photo
    photo_stats['all_correct'] = (photo_stats['pred_rate'] == 1.0) | (photo_stats['pred_rate'] == 0.0)
    photo_stats['all_correct'] = photo_stats.apply(
        lambda row: (row['pred_rate'] == 1.0 and row['y_true'] == 1) or 
                    (row['pred_rate'] == 0.0 and row['y_true'] == 0),
        axis=1
    )
    
    perfect_photos = photo_stats['all_correct'].sum()
    print(f"  Photos with ALL versions predicted correctly: {perfect_photos}/{len(photo_stats)} ({perfect_photos/len(photo_stats)*100:.1f}%)")
    
    # Questo è sospetto se troppo alto
    if perfect_photos / len(photo_stats) > 0.85:
        print(f"  🚨 WARNING: {perfect_photos/len(photo_stats)*100:.1f}% perfect accuracy is suspiciously high!")
        print(f"     This may indicate data leakage (model memorizing photos)")
    
    # Analizza consistency tra versioni
    print(f"\n🔍 CONSISTENCY ANALYSIS")
    
    # Per ogni foto, calcola std delle probabilità tra versioni
    consistency = df.groupby('photo_id')['y_prob'].std().fillna(0)
    
    print(f"  Avg std of probabilities across versions: {consistency.mean():.4f}")
    print(f"  Photos with std < 0.05 (very consistent): {(consistency < 0.05).sum()}/{len(consistency)} ({(consistency < 0.05).sum()/len(consistency)*100:.1f}%)")
    
    # Alta consistency può indicare che il modello riconosce la foto, non i pattern
    if (consistency < 0.05).sum() / len(consistency) > 0.7:
        print(f"  🚨 WARNING: {(consistency < 0.05).sum()/len(consistency)*100:.1f}% photos have very consistent predictions!")
        print(f"     This suggests the model is recognizing specific photos, not learning general patterns")
        print(f"     Possible data leakage!")
    
    # Trova esempi di alta consistency
    high_consistency_photos = consistency[consistency < 0.02].head(10)
    if len(high_consistency_photos) > 0:
        print(f"\n  Examples of highly consistent photos (std < 0.02):")
        for photo_id, std in high_consistency_photos.items():
            versions = df[df['photo_id'] == photo_id]
            print(f"    {photo_id}: {len(versions)} versions, std={std:.4f}, probs={versions['y_prob'].min():.3f}-{versions['y_prob'].max():.3f}")


def analyze_false_positives(predictions_csv: Path):
    """
    Analizza i falsi positivi per pattern di leakage.
    """
    print(f"\n" + "="*80)
    print("FALSE POSITIVES ANALYSIS")
    print("="*80)
    
    df = pd.read_csv(predictions_csv)
    df['photo_id'] = df['path'].apply(extract_photo_id)
    
    # Falsi positivi
    fp = df[(df['y_true'] == 0) & (df['y_pred'] == 1)]
    
    if len(fp) == 0:
        print("  No false positives found")
        return
    
    print(f"  Total false positives: {len(fp)}")
    
    # Raggruppa per photo_id
    fp_photos = fp.groupby('photo_id').size()
    
    print(f"  Unique photos with FP: {len(fp_photos)}")
    print(f"  Avg FP per photo: {fp_photos.mean():.2f}")
    
    # Foto con TUTTE le versioni FP
    all_fp_photos = []
    for photo_id in fp_photos.index:
        photo_versions = df[df['photo_id'] == photo_id]
        if len(photo_versions[photo_versions['y_pred'] == 1]) == len(photo_versions):
            all_fp_photos.append(photo_id)
    
    print(f"  Photos with ALL versions as FP: {len(all_fp_photos)}")
    
    if len(all_fp_photos) > 0:
        print(f"\n  Examples:")
        for photo_id in all_fp_photos[:5]:
            versions = df[df['photo_id'] == photo_id]
            print(f"    {photo_id}: {len(versions)} versions, all predicted as positive")
            print(f"      Prob range: {versions['y_prob'].min():.3f}-{versions['y_prob'].max():.3f}")


def analyze_false_negatives(predictions_csv: Path):
    """
    Analizza i falsi negativi per pattern di leakage.
    """
    print(f"\n" + "="*80)
    print("FALSE NEGATIVES ANALYSIS")
    print("="*80)
    
    df = pd.read_csv(predictions_csv)
    df['photo_id'] = df['path'].apply(extract_photo_id)
    
    # Falsi negativi
    fn = df[(df['y_true'] == 1) & (df['y_pred'] == 0)]
    
    if len(fn) == 0:
        print("  No false negatives found")
        return
    
    print(f"  Total false negatives: {len(fn)}")
    
    # Raggruppa per photo_id
    fn_photos = fn.groupby('photo_id').size()
    
    print(f"  Unique photos with FN: {len(fn_photos)}")
    print(f"  Avg FN per photo: {fn_photos.mean():.2f}")
    
    # Foto con TUTTE le versioni FN
    all_fn_photos = []
    for photo_id in fn_photos.index:
        photo_versions = df[df['photo_id'] == photo_id]
        if len(photo_versions[photo_versions['y_pred'] == 0]) == len(photo_versions):
            all_fn_photos.append(photo_id)
    
    print(f"  Photos with ALL versions as FN: {len(all_fn_photos)}")
    
    if len(all_fn_photos) > 0:
        print(f"\n  Examples:")
        for photo_id in all_fn_photos[:5]:
            versions = df[df['photo_id'] == photo_id]
            print(f"    {photo_id}: {len(versions)} versions, all predicted as negative")
            print(f"      Prob range: {versions['y_prob'].min():.3f}-{versions['y_prob'].max():.3f}")
            print(f"      Defect: {versions['defect_type'].iloc[0]}, Generator: {versions['generator'].iloc[0]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze data leakage from run results")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory (e.g., outputs/runs/2026-02-17_noK2)")
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    predictions_csv = run_dir / "predictions.csv"
    
    if not predictions_csv.exists():
        print(f"Error: {predictions_csv} not found")
        return 1
    
    # Analisi principale
    analyze_leakage_from_predictions(predictions_csv)
    
    # Analisi errori
    analyze_false_positives(predictions_csv)
    analyze_false_negatives(predictions_csv)
    
    # Conclusioni
    print(f"\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print(f"\n⚠️  IMPORTANT: This analysis is based ONLY on test set predictions.")
    print(f"   For a complete leakage audit, you need access to train/val/test splits.")
    print(f"\n   To perform a complete audit:")
    print(f"   1. Run: python scripts/audit_data_leakage.py --config configs/baseline.yaml")
    print(f"   2. This will verify that train and test sets have NO overlapping photos")
    print(f"\n   If you see high consistency or suspiciously high accuracy:")
    print(f"   - The model may be memorizing specific photos (data leakage)")
    print(f"   - Re-train with group_based_split_v6() to fix this")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
