#!/usr/bin/env python3
"""
Example script for testing a trained model on an external dataset.

This demonstrates how to evaluate a model on completely separate data
that was never seen during training, validation, or internal testing.

Usage:
    python scripts/external_test_example.py \
        --checkpoint checkpoints/2026-02-17_baseline/best.pt \
        --external_dataset /path/to/external/dataset \
        --run_name 2026-02-17_baseline_external_test
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test model on external dataset")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--external_dataset", type=str, required=True,
                        help="Path to external test dataset root")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name for this evaluation run")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Config file (default: configs/baseline.yaml)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    args = parser.parse_args()
    
    # Verify paths exist
    checkpoint_path = Path(args.checkpoint)
    external_dataset_path = Path(args.external_dataset)
    config_path = Path(args.config)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not external_dataset_path.exists():
        print(f"❌ External dataset not found: {external_dataset_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    print("="*80)
    print("EXTERNAL TEST EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"External dataset: {external_dataset_path}")
    print(f"Run name: {args.run_name}")
    print(f"Config: {config_path}")
    print(f"Threshold: {args.threshold}")
    print("="*80)
    
    # Run evaluation with external dataset
    cmd = [
        sys.executable,
        "src/eval.py",
        "--config", str(config_path),
        "--run_name", args.run_name,
        "--checkpoint_path", str(checkpoint_path),
        "--threshold", str(args.threshold),
        "--external_test_dataset", str(external_dataset_path)
    ]
    
    print("\nRunning evaluation...")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✓ External test evaluation complete!")
        print("="*80)
        print(f"\nResults saved to: outputs/runs/{args.run_name}/")
        print("\nNext steps:")
        print(f"  1. Review metrics: outputs/runs/{args.run_name}/metrics.json")
        print(f"  2. Analyze results: python scripts/analyze_results.py {args.run_name}")
        print(f"  3. Compare with internal test: python scripts/compare_runs.py")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
