#!/usr/bin/env python3
"""
Generate visual mosaics of hard FP/FN cases for manual inspection.

Creates image grids showing all 8 versions of problematic photos
with probabilities overlaid.

Usage:
    python scripts/generate_error_mosaics.py --run outputs/runs/2026-02-17_noK2_noLeakage
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def extract_photo_id(path: str) -> str:
    """Extract photo ID from path (first 4 characters)."""
    return Path(path).stem[:4]


def create_mosaic(image_paths: list, probs: list, labels: list, 
                  output_path: Path, grid_size: tuple = (4, 2)):
    """
    Create a mosaic of images with probabilities overlaid.
    
    Args:
        image_paths: list of image paths
        probs: list of probabilities
        labels: list of true labels
        output_path: where to save mosaic
        grid_size: (cols, rows) for grid layout
    """
    n_images = len(image_paths)
    cols, rows = grid_size
    
    # Load images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"  ⚠️  Failed to load {path}: {e}")
            # Create placeholder
            images.append(Image.new('RGB', (224, 224), color=(128, 128, 128)))
    
    if len(images) == 0:
        print(f"  ❌ No images loaded for mosaic")
        return
    
    # Resize all to same size
    target_size = (224, 224)
    images_resized = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images]
    
    # Create mosaic
    mosaic_width = cols * target_size[0]
    mosaic_height = rows * target_size[1]
    mosaic = Image.new('RGB', (mosaic_width, mosaic_height), color=(255, 255, 255))
    
    # Paste images
    for idx, (img, prob, label) in enumerate(zip(images_resized, probs, labels)):
        if idx >= cols * rows:
            break
        
        row = idx // cols
        col = idx % cols
        x = col * target_size[0]
        y = row * target_size[1]
        
        # Paste image
        mosaic.paste(img, (x, y))
        
        # Draw probability and label
        draw = ImageDraw.Draw(mosaic)
        
        # Try to use a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Background for text
        text = f"p={prob:.3f} y={label}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x + 5
        text_y = y + 5
        
        # Draw background rectangle
        draw.rectangle(
            [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
            fill=(0, 0, 0, 180)
        )
        
        # Draw text
        color = (255, 0, 0) if (prob >= 0.5 and label == 0) or (prob < 0.5 and label == 1) else (0, 255, 0)
        draw.text((text_x, text_y), text, fill=color, font=font)
    
    # Save
    mosaic.save(output_path, quality=95)
    print(f"  ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate error mosaics for visual inspection")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top errors to visualize")
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    
    print("="*80)
    print("GENERATING ERROR MOSAICS")
    print("="*80)
    print(f"Run: {run_dir.name}\n")
    
    # Load predictions
    predictions_file = run_dir / "predictions.csv"
    if not predictions_file.exists():
        print(f"❌ Error: {predictions_file} not found")
        return 1
    
    df = pd.read_csv(predictions_file)
    df['photo_id'] = df['path'].apply(extract_photo_id)
    
    # Load hard cases
    hard_fp_file = run_dir / "photo_hard_fp.csv"
    hard_fn_file = run_dir / "photo_hard_fn.csv"
    
    if not hard_fp_file.exists() or not hard_fn_file.exists():
        print(f"❌ Error: Hard case files not found")
        print(f"   Run analyze_by_photo.py first")
        return 1
    
    hard_fp_df = pd.read_csv(hard_fp_file)
    hard_fn_df = pd.read_csv(hard_fn_file)
    
    # Create output directory
    mosaic_dir = run_dir / "error_mosaics"
    mosaic_dir.mkdir(exist_ok=True)
    
    # Generate FP mosaics
    print(f"\nGenerating False Positive mosaics (top {args.top_n})...")
    for idx, row in hard_fp_df.head(args.top_n).iterrows():
        photo_id = row['photo_id']
        photo_versions = df[df['photo_id'] == photo_id].sort_values('path')
        
        if len(photo_versions) == 0:
            continue
        
        image_paths = photo_versions['path'].tolist()
        probs = photo_versions['y_prob'].tolist()
        labels = photo_versions['y_true'].tolist()
        
        output_path = mosaic_dir / f"FP_{photo_id}_prob{row['y_prob']:.3f}.jpg"
        
        print(f"  {photo_id}: {len(photo_versions)} versions, prob={row['y_prob']:.3f}")
        create_mosaic(image_paths, probs, labels, output_path)
    
    # Generate FN mosaics
    print(f"\nGenerating False Negative mosaics (top {args.top_n})...")
    for idx, row in hard_fn_df.head(args.top_n).iterrows():
        photo_id = row['photo_id']
        photo_versions = df[df['photo_id'] == photo_id].sort_values('path')
        
        if len(photo_versions) == 0:
            continue
        
        image_paths = photo_versions['path'].tolist()
        probs = photo_versions['y_prob'].tolist()
        labels = photo_versions['y_true'].tolist()
        
        defect = row.get('defect_type', 'unknown')
        gen = row.get('generator', 'unknown')
        output_path = mosaic_dir / f"FN_{photo_id}_{defect}_{gen}_prob{row['y_prob']:.3f}.jpg"
        
        print(f"  {photo_id}: {len(photo_versions)} versions, prob={row['y_prob']:.3f}, defect={defect}")
        create_mosaic(image_paths, probs, labels, output_path)
    
    print("\n" + "="*80)
    print("MOSAICS GENERATED")
    print("="*80)
    print(f"\n✓ Saved to: {mosaic_dir}")
    print(f"  False Positives: {len(list(mosaic_dir.glob('FP_*.jpg')))}")
    print(f"  False Negatives: {len(list(mosaic_dir.glob('FN_*.jpg')))}")
    print(f"\n📸 Inspect mosaics to understand:")
    print(f"  - Are FP photos visually similar to AI-generated?")
    print(f"  - Are FN defects too subtle?")
    print(f"  - Are there compression/resize artifacts?")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
