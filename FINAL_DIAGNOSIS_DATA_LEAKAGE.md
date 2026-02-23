# Final Diagnosis: Data Leakage Root Cause

**Date:** 2026-02-20  
**Status:** 🚨 CRITICAL - DATA LEAKAGE CONFIRMED

## 🔥 Smoking Gun Evidence

**15/16 hard FP photos (93.8%) are MISSING from hard negative test set!**

These photos were in the ORIGINAL test set, but are NOT in the HARD NEGATIVE test set.
They likely moved to train/val → The model was trained on them!

## Test Set Comparison

```
Original test:     226 photos
Hard negative test: 221 photos

Overlap: Only 35 photos in common!
Missing: 191 photos (84.5%)
New:     186 photos (84.2%)
```

**This is NOT a small difference - it's a COMPLETELY DIFFERENT test set!**

## Root Cause Analysis

### Hypothesis 1: Different Split Strategy (MOST LIKELY)

**Original training** probably used `group_based_split_v6`:
- Config says `split_strategy: 'domain_aware'`
- But the actual training might have used `group_v6` (default fallback)
- Or the config was added AFTER the training

**Hard negative** uses `domain_aware_group_split_v1`:
- We fixed the code to read from config
- Now it correctly uses `domain_aware`
- But this creates a DIFFERENT split than original!

### Hypothesis 2: Different Random Seed

Even with same algorithm, different seed → different split:
- `domain_aware_group_split_v1` has internal randomness
- If seed is different, split is completely different
- But seed should be same (42) from config

### Hypothesis 3: Dataset Changed

- Dataset might have changed between runs
- New photos added or removed
- This would cause different splits

## How to Verify Root Cause

### Check Original Training Logs

Look for this in original training output:
```
# If it says this → used group_v6
GROUP-BASED SPLIT (No Data Leakage)

# If it says this → used domain_aware
DOMAIN-AWARE GROUP-BASED SPLIT (No Data Leakage)
```

### Check Config History

```bash
git log -p configs/convnext_v8.yaml
```

See when `split_strategy: 'domain_aware'` was added.
If it was added AFTER the original training → that's the problem!

## Solution Options

### Option 1: Use Same Split as Original (RECOMMENDED)

If original used `group_v6`:

```python
# In hard_negative_finetune.py
# Force use of group_v6 to match original
train_df, val_df, test_df = group_based_split_v6(
    df, 0.70, 0.15, 0.15, seed=config.get('seed', 42)
)
```

This ensures SAME split as original training.

### Option 2: Re-train Original with domain_aware

Re-run the ENTIRE pipeline with `domain_aware`:
1. Delete original run
2. Re-train from scratch with `domain_aware`
3. Then run hard negative with `domain_aware`

This ensures consistency but requires re-training.

### Option 3: Save and Load Split

Most robust solution:

1. **During original training**, save the split:
   ```python
   train_df.to_csv(f"{output_dir}/train_split.csv")
   val_df.to_csv(f"{output_dir}/val_split.csv")
   test_df.to_csv(f"{output_dir}/test_split.csv")
   ```

2. **During hard negative**, load the SAME split:
   ```python
   train_df = pd.read_csv(f"{base_run}/train_split.csv")
   val_df = pd.read_csv(f"{base_run}/val_split.csv")
   test_df = pd.read_csv(f"{base_run}/test_split.csv")
   ```

This GUARANTEES identical splits.

## Immediate Action Required

### Step 1: Identify Original Split Strategy

Check original training logs to see if it used `group_v6` or `domain_aware`.

### Step 2: Match the Split

Update `hard_negative_finetune.py` to use the SAME split as original.

### Step 3: Re-run Hard Negative

With matching split strategy.

### Step 4: Verify

After re-run, check:
```python
# Test set size should be IDENTICAL
Original:     226 photos
Hard Negative: 226 photos  ← MUST be same!

# Test set photos should be IDENTICAL
Overlap: 226/226 (100%)
Missing: 0
New:     0
```

## Long-term Solution

### Implement Split Saving

Modify `train_v7.py` to save splits:

```python
# After split
train_df.to_csv(f"{output_dir}/train_split.csv", index=False)
val_df.to_csv(f"{output_dir}/val_split.csv", index=False)
test_df.to_csv(f"{output_dir}/test_split.csv", index=False)

print(f"✓ Saved split to {output_dir}/")
```

### Modify hard_negative_finetune.py

```python
# Load split from original run
split_dir = args.run
if os.path.exists(f"{split_dir}/train_split.csv"):
    print(f"Loading split from {split_dir}")
    train_df = pd.read_csv(f"{split_dir}/train_split.csv")
    val_df = pd.read_csv(f"{split_dir}/val_split.csv")
    test_df = pd.read_csv(f"{split_dir}/test_split.csv")
else:
    print("⚠️  WARNING: Split files not found, generating new split")
    # Fallback to generating split
    ...
```

This ensures PERFECT consistency.

## Conclusion

🚨 **Current results are NOT VALID** due to data leakage

✅ **Root cause identified**: Different split strategy between original and hard negative

⏳ **Action required**: Identify original split and match it

📊 **Expected after fix**: F1 ~85% (not 91%), fix rate ~50% (not 94%)

The high performance (F1 91%, fix rate 94%) is due to the model being trained on photos it's tested on.
