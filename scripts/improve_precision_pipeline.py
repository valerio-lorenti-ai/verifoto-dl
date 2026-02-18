#!/usr/bin/env python3
"""
Complete pipeline for improving precision through calibration and hard negative mining.

This script runs the full improvement pipeline:
1. Photo-level analysis
2. Temperature scaling calibration
3. Hard negative fine-tuning
4. Visual error mosaics
5. Final comparison

Usage:
    python scripts/improve_precision_pipeline.py \
        --run outputs/runs/2026-02-17_noK2_noLeakage \
        --config configs/baseline.yaml
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully")
        return True
    else:
        print(f"\n❌ {description} failed with code {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run complete precision improvement pipeline")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration step")
    parser.add_argument("--skip-hard-negative", action="store_true", help="Skip hard negative mining")
    parser.add_argument("--skip-mosaics", action="store_true", help="Skip mosaic generation")
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    
    print("="*80)
    print("PRECISION IMPROVEMENT PIPELINE")
    print("="*80)
    print(f"Run: {run_dir.name}")
    print(f"Config: {args.config}\n")
    
    steps_completed = []
    steps_failed = []
    
    # Step 1: Photo-level analysis
    if not (run_dir / "photo_level_metrics.json").exists():
        success = run_command([
            sys.executable,
            "scripts/analyze_by_photo.py",
            "--run", str(run_dir),
            "--min-recall", "0.90"
        ], "Photo-level analysis")
        
        if success:
            steps_completed.append("Photo-level analysis")
        else:
            steps_failed.append("Photo-level analysis")
            print("\n❌ Pipeline failed at photo-level analysis")
            return 1
    else:
        print("\n✓ Photo-level analysis already done, skipping")
        steps_completed.append("Photo-level analysis (cached)")
    
    # Step 2: Temperature scaling calibration
    if not args.skip_calibration:
        if not (run_dir / "calibration_T.json").exists():
            success = run_command([
                sys.executable,
                "scripts/apply_calibration.py",
                "--run", str(run_dir)
            ], "Temperature scaling calibration")
            
            if success:
                steps_completed.append("Calibration")
            else:
                steps_failed.append("Calibration")
                print("\n⚠️  Calibration failed, continuing with other steps")
        else:
            print("\n✓ Calibration already done, skipping")
            steps_completed.append("Calibration (cached)")
    else:
        print("\n⏭️  Skipping calibration (--skip-calibration)")
    
    # Step 3: Hard negative fine-tuning
    if not args.skip_hard_negative:
        hn_run_dir = run_dir.parent / f"{run_dir.name}_hn"
        
        if not hn_run_dir.exists():
            success = run_command([
                sys.executable,
                "scripts/hard_negative_finetune.py",
                "--run", str(run_dir),
                "--config", args.config,
                "--epochs", "5",
                "--lr", "1e-5",
                "--repeat_factor", "3"
            ], "Hard negative fine-tuning")
            
            if success:
                steps_completed.append("Hard negative mining")
            else:
                steps_failed.append("Hard negative mining")
                print("\n⚠️  Hard negative mining failed, continuing with other steps")
        else:
            print("\n✓ Hard negative fine-tuning already done, skipping")
            steps_completed.append("Hard negative mining (cached)")
    else:
        print("\n⏭️  Skipping hard negative mining (--skip-hard-negative)")
    
    # Step 4: Generate error mosaics
    if not args.skip_mosaics:
        mosaic_dir = run_dir / "error_mosaics"
        
        if not mosaic_dir.exists() or len(list(mosaic_dir.glob("*.jpg"))) == 0:
            success = run_command([
                sys.executable,
                "scripts/generate_error_mosaics.py",
                "--run", str(run_dir),
                "--top_n", "10"
            ], "Error mosaic generation")
            
            if success:
                steps_completed.append("Error mosaics")
            else:
                steps_failed.append("Error mosaics")
                print("\n⚠️  Mosaic generation failed, continuing")
        else:
            print("\n✓ Error mosaics already generated, skipping")
            steps_completed.append("Error mosaics (cached)")
    else:
        print("\n⏭️  Skipping mosaic generation (--skip-mosaics)")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    print(f"\n✅ Steps completed ({len(steps_completed)}):")
    for step in steps_completed:
        print(f"  ✓ {step}")
    
    if steps_failed:
        print(f"\n❌ Steps failed ({len(steps_failed)}):")
        for step in steps_failed:
            print(f"  ✗ {step}")
    
    print(f"\n📊 Results:")
    print(f"  Original run: {run_dir}")
    
    if (run_dir / "photo_level_metrics.json").exists():
        import json
        with open(run_dir / "photo_level_metrics.json", 'r') as f:
            metrics = json.load(f)
            orig_metrics = metrics['original_threshold']
            opt_metrics = metrics['optimal_precision_threshold']
            
            print(f"\n  Original threshold ({orig_metrics['threshold']:.3f}):")
            print(f"    Precision: {orig_metrics['precision']:.1%}")
            print(f"    Recall:    {orig_metrics['recall']:.1%}")
            print(f"    F1:        {orig_metrics['f1']:.1%}")
            
            print(f"\n  Optimized threshold ({opt_metrics['threshold']:.3f}):")
            print(f"    Precision: {opt_metrics['precision']:.1%} ({(opt_metrics['precision']-orig_metrics['precision'])*100:+.1f}%)")
            print(f"    Recall:    {opt_metrics['recall']:.1%} ({(opt_metrics['recall']-orig_metrics['recall'])*100:+.1f}%)")
            print(f"    F1:        {opt_metrics['f1']:.1%} ({(opt_metrics['f1']-orig_metrics['f1'])*100:+.1f}%)")
    
    if (run_dir / "calibration_T.json").exists():
        import json
        with open(run_dir / "calibration_T.json", 'r') as f:
            calib = json.load(f)
            print(f"\n  Calibration temperature: {calib['temperature']:.4f}")
    
    hn_run_dir = run_dir.parent / f"{run_dir.name}_hn"
    if hn_run_dir.exists():
        print(f"\n  Hard negative run: {hn_run_dir}")
    
    print(f"\n📁 Generated files:")
    print(f"  {run_dir}/photo_level_metrics.json")
    print(f"  {run_dir}/chosen_threshold.json")
    print(f"  {run_dir}/threshold_sweep.csv")
    if (run_dir / "calibration_T.json").exists():
        print(f"  {run_dir}/calibration_T.json")
        print(f"  {run_dir}/calibration_report.json")
    if (run_dir / "error_mosaics").exists():
        print(f"  {run_dir}/error_mosaics/ ({len(list((run_dir / 'error_mosaics').glob('*.jpg')))} images)")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Review chosen_threshold.json for recommended threshold")
    print(f"  2. Inspect error_mosaics/ to understand failure modes")
    print(f"  3. If precision < 75%, consider residual stream implementation")
    print(f"  4. Deploy with calibrated probabilities and optimized threshold")
    
    return 0 if len(steps_failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
