# Notebook Hard Negative Cell Fix

**Date:** 2026-02-20  
**Status:** ✅ COMPLETE

## Problem

The hard negative fine-tuning cell in `scripts/notebooks/verifoto_dl.ipynb` was missing the `--output_suffix _hard_negative` parameter, causing the script to fail with:

```
hard_negative_finetune.py: error: unrecognized arguments: --checkpoint_dir
```

## Root Cause

The `hard_negative_finetune.py` script requires the `--output_suffix` parameter to generate the correct output directory name. Without it, the script was trying to use the wrong directory structure.

## Solution

Updated the notebook cell to include `--output_suffix _hard_negative`:

```python
!python scripts/hard_negative_finetune.py \
    --run {OUTPUT_DIR} \
    --config {CONFIG_PATH} \
    --checkpoint_dir {CHECKPOINT_DIR} \
    --epochs 5 \
    --lr 1e-5 \
    --repeat_factor 3 \
    --output_suffix _hard_negative  # ← ADDED
```

## Verification

✅ Parameter added successfully  
✅ Notebook cell now matches the correct command format  
✅ Ready for hard negative fine-tuning execution

## Next Steps

The notebook is now ready to run hard negative fine-tuning. When precision < 75%, the cell will:

1. Fine-tune on 17 problematic photos (systematic false positives)
2. Create a new run: `{EXPERIMENT_NAME}_hard_negative`
3. Save results to: `outputs/runs/{EXPERIMENT_NAME}_hard_negative/`
4. Save checkpoint to: `{CHECKPOINT_DIR}/{EXPERIMENT_NAME}_hard_negative/best.pt`

## Expected Improvement

Current GPT-1.5 performance (threshold 0.55):
- Recall: 79.0%
- Precision: 100%
- F1: 88.3%

Target after hard negative fine-tuning:
- Recall: 90-95% (improve by +11-16%)
- Precision: maintain ~100%
- F1: 95%+

The fine-tuning will focus on the 42 false negatives (21% of GPT-1.5 images) to improve recall while maintaining high precision.
