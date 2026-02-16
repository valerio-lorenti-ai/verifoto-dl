"""Compare metrics across multiple training runs"""
import json
import sys
from pathlib import Path
from typing import List, Dict

def load_run_metrics(run_dir: Path) -> Dict:
    """Load metrics from a run directory"""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file) as f:
        data = json.load(f)
    
    return {
        "run_name": data.get("run_name", run_dir.name),
        "timestamp": data.get("timestamp", "unknown"),
        "model": data.get("config", {}).get("model_name", "unknown"),
        "threshold": data.get("threshold", 0.5),
        "git_commit": data.get("git_commit", "unknown")[:8],
        **data.get("test_metrics", {})
    }

def main():
    runs_dir = Path("outputs/runs")
    
    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist")
        sys.exit(1)
    
    # Load all runs
    runs = []
    for run_path in sorted(runs_dir.iterdir()):
        if run_path.is_dir() and run_path.name != ".gitkeep":
            metrics = load_run_metrics(run_path)
            if metrics:
                runs.append(metrics)
    
    if not runs:
        print("No runs found")
        sys.exit(0)
    
    # Sort by PR-AUC (descending)
    runs.sort(key=lambda x: x.get("pr_auc", 0) or 0, reverse=True)
    
    # Print header
    print("\n" + "="*120)
    print(f"{'Run Name':<35} {'Model':<20} {'F1':>6} {'Prec':>6} {'Rec':>6} {'PR-AUC':>7} {'ROC-AUC':>7} {'Thr':>5} {'Commit':<8}")
    print("="*120)
    
    # Print runs
    for r in runs:
        f1 = r.get("f1", 0) or 0
        prec = r.get("prec", 0) or 0
        rec = r.get("rec", 0) or 0
        pr_auc = r.get("pr_auc", 0) or 0
        roc_auc = r.get("roc_auc", 0) or 0
        
        print(f"{r['run_name']:<35} {r['model']:<20} {f1:>6.4f} {prec:>6.4f} {rec:>6.4f} {pr_auc:>7.4f} {roc_auc:>7.4f} {r['threshold']:>5.2f} {r['git_commit']:<8}")
    
    print("="*120)
    print(f"\nTotal runs: {len(runs)}")
    
    # Best run
    if runs:
        best = runs[0]
        print(f"\n🏆 Best run (by PR-AUC): {best['run_name']}")
        print(f"   Model: {best['model']}")
        print(f"   PR-AUC: {best.get('pr_auc', 0):.4f}")
        print(f"   F1: {best.get('f1', 0):.4f}")
        print(f"   Commit: {best['git_commit']}")

if __name__ == "__main__":
    main()
