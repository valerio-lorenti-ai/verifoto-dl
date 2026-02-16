"""
Script per analizzare i risultati del training con dataset augmented_v6.
Mostra metriche per gruppi e identifica aree problematiche.
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_run_data(run_dir: Path):
    """Carica tutti i file CSV di un run"""
    data = {}
    
    files = {
        'predictions': 'predictions.csv',
        'food': 'group_metrics_food.csv',
        'defect': 'group_metrics_defect.csv',
        'generator': 'group_metrics_generator.csv',
        'quality': 'group_metrics_quality.csv',
        'fp': 'top_false_positives.csv',
        'fn': 'top_false_negatives.csv'
    }
    
    for key, filename in files.items():
        filepath = run_dir / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
        else:
            data[key] = None
    
    return data


def print_summary(data):
    """Stampa summary generale"""
    pred = data['predictions']
    if pred is None:
        print("No predictions found")
        return
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    n_total = len(pred)
    n_correct = (pred['y_true'] == pred['y_pred']).sum()
    accuracy = n_correct / n_total
    
    print(f"Total samples: {n_total}")
    print(f"Correct: {n_correct} ({accuracy:.2%})")
    print(f"Errors: {n_total - n_correct} ({1-accuracy:.2%})")
    
    # Breakdown by source
    print("\nBy source:")
    for source in pred['source'].unique():
        if pd.notna(source):
            subset = pred[pred['source'] == source]
            acc = (subset['y_true'] == subset['y_pred']).mean()
            print(f"  {source}: {len(subset)} samples, {acc:.2%} accuracy")


def print_group_metrics(data, group_name, top_n=10):
    """Stampa metriche per gruppo"""
    df = data.get(group_name)
    if df is None or len(df) == 0:
        print(f"\nNo {group_name} metrics found")
        return
    
    col_name = df.columns[0]  # Prima colonna è il nome del gruppo
    
    print("\n" + "="*80)
    print(f"METRICS BY {group_name.upper()}")
    print("="*80)
    
    # Ordina per F1 decrescente
    df_sorted = df.sort_values('f1', ascending=False)
    
    print(f"\n{'Group':<25} {'Samples':>8} {'F1':>6} {'Prec':>6} {'Rec':>6} {'FP':>5} {'FN':>5}")
    print("-"*80)
    
    for _, row in df_sorted.head(top_n).iterrows():
        group_val = str(row[col_name])[:24]
        print(f"{group_val:<25} {row['n_samples']:>8} {row['f1']:>6.3f} "
              f"{row['precision']:>6.3f} {row['recall']:>6.3f} "
              f"{int(row['fp']):>5} {int(row['fn']):>5}")
    
    # Identifica problemi
    print("\n⚠️  Problematic groups (F1 < 0.80):")
    low_f1 = df_sorted[df_sorted['f1'] < 0.80]
    if len(low_f1) > 0:
        for _, row in low_f1.iterrows():
            print(f"  - {row[col_name]}: F1={row['f1']:.3f}, "
                  f"samples={row['n_samples']}, FP={int(row['fp'])}, FN={int(row['fn'])}")
    else:
        print("  None! All groups have F1 >= 0.80")


def print_top_errors(data, error_type='fp', top_n=10):
    """Stampa top errori"""
    df = data.get(error_type)
    if df is None or len(df) == 0:
        print(f"\nNo {error_type} errors found")
        return
    
    error_name = "FALSE POSITIVES" if error_type == 'fp' else "FALSE NEGATIVES"
    
    print("\n" + "="*80)
    print(f"TOP {error_name}")
    print("="*80)
    
    print(f"\n{'Food':<15} {'Defect':<15} {'Generator':<20} {'Prob':>6} {'Path'}")
    print("-"*80)
    
    for _, row in df.head(top_n).iterrows():
        food = str(row.get('food_category', 'N/A'))[:14]
        defect = str(row.get('defect_type', 'N/A'))[:14]
        gen = str(row.get('generator', 'N/A'))[:19]
        prob = row['y_prob']
        path = Path(row['path']).name
        
        print(f"{food:<15} {defect:<15} {gen:<20} {prob:>6.3f} {path}")


def compare_generators(data):
    """Confronta performance tra generatori"""
    gen = data.get('generator')
    if gen is None or len(gen) == 0:
        print("\nNo generator metrics found")
        return
    
    print("\n" + "="*80)
    print("GENERATOR COMPARISON")
    print("="*80)
    
    gen_sorted = gen.sort_values('f1', ascending=False)
    
    print(f"\n{'Generator':<25} {'Samples':>8} {'F1':>6} {'Precision':>10} {'Recall':>8}")
    print("-"*80)
    
    for _, row in gen_sorted.iterrows():
        print(f"{row['generator']:<25} {row['n_samples']:>8} {row['f1']:>6.3f} "
              f"{row['precision']:>10.3f} {row['recall']:>8.3f}")
    
    # Identifica generatore più difficile
    if len(gen_sorted) > 0:
        worst = gen_sorted.iloc[-1]
        print(f"\n⚠️  Most challenging generator: {worst['generator']} (F1={worst['f1']:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument("run_name", type=str, help="Run name (e.g., 2026-02-16_baseline)")
    parser.add_argument("--top", type=int, default=10, help="Number of top items to show")
    args = parser.parse_args()
    
    # Trova run directory
    run_dir = Path("outputs/runs") / args.run_name
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    print(f"\nAnalyzing run: {args.run_name}")
    print(f"Directory: {run_dir}")
    
    # Carica dati
    data = load_run_data(run_dir)
    
    # Stampa analisi
    print_summary(data)
    print_group_metrics(data, 'food', top_n=args.top)
    print_group_metrics(data, 'defect', top_n=args.top)
    compare_generators(data)
    print_group_metrics(data, 'quality', top_n=args.top)
    print_top_errors(data, 'fp', top_n=args.top)
    print_top_errors(data, 'fn', top_n=args.top)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
