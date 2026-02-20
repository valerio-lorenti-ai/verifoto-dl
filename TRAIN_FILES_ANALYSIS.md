# Train Files Analysis - train.py vs train_v7.py

**Date:** 2026-02-20  
**Status:** ✅ ANALYSIS COMPLETE

## Current Situation

### Files Present
- `src/train.py` - Old training script
- `src/train_v7.py` - Current training script (used in notebook)

### Usage Analysis

#### `train_v7.py` (CURRENT)
✅ Used by notebook: `!python -m src.train_v7`  
✅ Has all necessary functions: `set_seed`, `validate`, `save_checkpoint`, `train_one_epoch`  
✅ Has `num_workers=2` optimization  
✅ Has `collate_with_metadata` (but doesn't handle None)

#### `train.py` (OLD)
❌ NOT used by notebook  
❌ Only imported by `hard_negative_finetune.py` (now fixed)  
⚠️ Missing `num_workers` optimization  
⚠️ Older version

## Changes Made

### 1. Updated `hard_negative_finetune.py`
```python
# OLD (wrong)
from src.train import set_seed, validate, save_checkpoint

# NEW (correct)
from src.train_v7 import set_seed, validate, save_checkpoint
```

### 2. Created Robust Training Function
Added `train_one_epoch_robust()` in `hard_negative_finetune.py` that:
- Handles None batches from `collate_fn_filter_none`
- Tracks skipped batches
- Continues training gracefully

## Recommendation: Keep or Delete train.py?

### ✅ SAFE TO DELETE `train.py`

**Reasons:**
1. Not used anywhere in the codebase
2. `train_v7.py` is the current version
3. All functions exist in `train_v7.py`
4. Keeping both files creates confusion
5. `hard_negative_finetune.py` now uses `train_v7.py`

### Before Deleting - Verify:
```bash
# Check no hidden references
grep -r "from src.train import" .
grep -r "import src.train" .
grep -r "python -m src.train" .
```

Expected: Only references to `train_v7`, none to `train`

## Function Comparison

| Function | train.py | train_v7.py | Notes |
|----------|----------|-------------|-------|
| `set_seed` | ✅ | ✅ | Identical |
| `get_git_commit` | ✅ | ✅ | Identical |
| `collate_with_metadata` | ✅ | ✅ | Identical (neither handles None) |
| `train_one_epoch` | ✅ | ✅ | Identical |
| `validate` | ✅ | ✅ | Identical |
| `save_checkpoint` | ✅ | ✅ | Identical |
| `main` | ✅ | ✅ | Different (v7 has num_workers=2) |

## Our Robust Solution

We created `train_one_epoch_robust()` and `collate_fn_filter_none()` in `hard_negative_finetune.py` because:

1. **Specific to hard negative training**: Only needed when using `WeightedRandomSampler` with problematic images
2. **Doesn't modify core training**: Keeps `train_v7.py` clean
3. **Isolated error handling**: Only affects hard negative fine-tuning
4. **Easy to maintain**: All hard negative logic in one file

## Conclusion

✅ `hard_negative_finetune.py` now correctly imports from `train_v7.py`  
✅ Robust error handling implemented for hard negative training  
✅ `train.py` can be safely deleted (no longer used)  
✅ No confusion between old and new training scripts

## Next Steps

1. ✅ Updated imports in `hard_negative_finetune.py`
2. ⏳ Test hard negative fine-tuning
3. ⏳ If successful, delete `train.py` to avoid confusion
4. ⏳ Update documentation to reference only `train_v7.py`
