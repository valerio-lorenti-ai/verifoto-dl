# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LOCAL (Kiro/PC)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   src/       │    │  configs/    │    │   scripts/   │    │
│  │  - train.py  │    │  - *.yaml    │    │  - *.py      │    │
│  │  - eval.py   │    │              │    │              │    │
│  │  - utils/    │    │              │    │              │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                    │                    │            │
│         └────────────────────┴────────────────────┘            │
│                              │                                 │
│                         git commit                             │
│                              │                                 │
└──────────────────────────────┼─────────────────────────────────┘
                               │
                          git push
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                           GITHUB                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Repository: verifoto-dl                                 │  │
│  │  - Code (src/, configs/, scripts/)                       │  │
│  │  - Results (outputs/runs/)                               │  │
│  │  - Documentation (docs/, *.md)                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                          git clone
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GOOGLE COLAB (GPU)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Training Environment                                  │    │
│  │  - Clone repo from GitHub                              │    │
│  │  - Install dependencies                                │    │
│  │  - Mount Google Drive                                  │    │
│  │  - Run: python -m src.train                            │    │
│  └────────────────────────────────────────────────────────┘    │
│         │                                        │               │
│         │ reads                                  │ writes        │
│         ▼                                        ▼               │
│  ┌─────────────┐                         ┌─────────────┐       │
│  │   Dataset   │                         │  Results    │       │
│  │  (Drive)    │                         │ (outputs/)  │       │
│  └─────────────┘                         └─────────────┘       │
│                                                   │              │
│                                              git commit          │
│                                                   │              │
└───────────────────────────────────────────────────┼──────────────┘
                                                    │
                                               git push
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GOOGLE DRIVE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │   Dataset        │         │   Checkpoints    │            │
│  │   (images/)      │         │   (*.pt files)   │            │
│  │   - NON_FRODE/   │         │   - best.pt      │            │
│  │   - FRODE/       │         │   - by run_name/ │            │
│  └──────────────────┘         └──────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌──────────┐
│  Local   │  1. Edit code/config
│  (Kiro)  │  2. git commit + push
└────┬─────┘
     │
     ▼
┌──────────┐
│  GitHub  │  3. Store code + results
└────┬─────┘
     │
     ▼
┌──────────┐
│  Colab   │  4. git clone
│          │  5. Mount Drive
│          │  6. Run training
│          │  7. Save checkpoints to Drive
│          │  8. Save results to outputs/
│          │  9. git commit + push results
└────┬─────┘
     │
     ▼
┌──────────┐
│  GitHub  │  10. Store results
└────┬─────┘
     │
     ▼
┌──────────┐
│  Local   │  11. git pull
│  (Kiro)  │  12. Analyze results
└──────────┘
```

## Module Architecture

```
src/
├── train.py                    # Main training script
│   ├── Loads config (YAML)
│   ├── Calls utils.data for dataset
│   ├── Calls utils.model for model
│   ├── Calls utils.metrics for evaluation
│   └── Calls utils.visualization for plots
│
├── eval.py                     # Evaluation script
│   ├── Loads checkpoint
│   ├── Calls utils.metrics for evaluation
│   └── Calls utils.visualization for plots
│
└── utils/
    ├── data.py                 # Dataset handling
    │   ├── find_class_dirs()
    │   ├── compute_hashes()
    │   ├── group_near_duplicates()
    │   ├── stratified_group_split()
    │   ├── build_transforms()
    │   └── ImageBinaryDataset
    │
    ├── model.py                # Model building
    │   ├── build_model()
    │   ├── set_backbone_trainable()
    │   └── find_last_conv_layer()
    │
    ├── metrics.py              # Evaluation
    │   ├── predict_proba()
    │   ├── compute_metrics_from_probs()
    │   └── EarlyStopping
    │
    └── visualization.py        # Plotting
        ├── plot_prob_distributions()
        ├── plot_roc_pr()
        └── plot_confusion_matrix()
```

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    python -m src.train                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Load Config (YAML)                                      │
│     - Model name, hyperparameters, paths                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Load Dataset                                            │
│     - Find class directories                                │
│     - List all images                                       │
│     - Compute hashes (deduplication)                        │
│     - Group near-duplicates                                 │
│     - Stratified group split (70/15/15)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Create DataLoaders                                      │
│     - Apply augmentations (train)                           │
│     - Apply normalization (all)                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Build Model                                             │
│     - Load pretrained weights                               │
│     - Freeze backbone (phase 1)                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Phase 1: Train Head Only                                │
│     - Optimizer: AdamW (lr=3e-4)                            │
│     - Scheduler: CosineAnnealing                            │
│     - Early stopping on PR-AUC                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Phase 2: Finetune All                                   │
│     - Unfreeze backbone                                     │
│     - Optimizer: AdamW (lr=1e-4)                            │
│     - Scheduler: CosineAnnealing                            │
│     - Early stopping on PR-AUC                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Load Best Checkpoint                                    │
│     - From Drive or local                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  8. Evaluate on Test Set                                    │
│     - Compute metrics (F1, PR-AUC, etc.)                    │
│     - Generate confusion matrix                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  9. Save Results                                            │
│     - metrics.json (structured)                             │
│     - notes.md (human-readable)                             │
│     - Plots (cm.png, roc_curve.png, etc.)                   │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Flow

```
configs/baseline.yaml
        │
        ▼
┌───────────────────┐
│  Load YAML        │
│  - dataset_root   │
│  - model_name     │
│  - img_size       │
│  - batch_size     │
│  - epochs_*       │
│  - lr_*           │
│  - etc.           │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Override via CLI │
│  --checkpoint_dir │
│  --run_name       │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Use in training  │
└───────────────────┘
```

## Output Structure

```
outputs/runs/<run_name>/
├── metrics.json              # Machine-readable
│   ├── run_name
│   ├── git_commit
│   ├── timestamp
│   ├── threshold
│   ├── test_metrics
│   │   ├── acc
│   │   ├── prec
│   │   ├── rec
│   │   ├── f1
│   │   ├── pr_auc
│   │   └── roc_auc
│   ├── confusion_matrix
│   └── config
│
├── notes.md                  # Human-readable
│   ├── Config summary
│   ├── Results table
│   ├── Confusion matrix
│   └── Git commit
│
├── cm.png                    # Confusion matrix plot
├── roc_curve.png             # ROC curve
├── pr_curve.png              # Precision-Recall curve
└── prob_dist.png             # Probability distribution
```

## Storage Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                         GITHUB                              │
│  - Code (src/, configs/, scripts/)                          │
│  - Results (outputs/runs/)                                  │
│  - Documentation (docs/, *.md)                              │
│                                                             │
│  Size: ~10MB (code + results)                               │
│  Versioned: Yes                                             │
│  Shareable: Yes                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      GOOGLE DRIVE                           │
│  - Dataset (images/)                                        │
│  - Checkpoints (*.pt files)                                 │
│                                                             │
│  Size: ~10GB (dataset + checkpoints)                        │
│  Versioned: No                                              │
│  Shareable: Via Drive sharing                               │
└─────────────────────────────────────────────────────────────┘
```

## Comparison with Original

```
Original (codice_google_colab.py):
┌────────────────────────────────┐
│  Single 600-line script        │
│  - All code in one file        │
│  - Hardcoded values            │
│  - Manual execution            │
│  - No version control          │
└────────────────────────────────┘

New Structure:
┌────────────────────────────────┐
│  Modular architecture          │
│  - Code split by function      │
│  - YAML configuration          │
│  - CLI interface               │
│  - Git-based workflow          │
│  - Structured outputs          │
└────────────────────────────────┘
```

## Key Design Principles

1. **Separation of Concerns**
   - Data handling → `utils/data.py`
   - Model building → `utils/model.py`
   - Evaluation → `utils/metrics.py`
   - Visualization → `utils/visualization.py`

2. **Configuration over Code**
   - Hyperparameters in YAML
   - Easy to experiment
   - Version controlled

3. **Reproducibility**
   - Fixed seeds
   - Git commit tracking
   - Structured outputs

4. **Flexibility**
   - Works in Colab, local, or other environments
   - Configurable paths
   - Multiple entry points

5. **Maintainability**
   - Clean code
   - Well documented
   - Easy to debug

## Execution Modes

### Mode 1: Colab Notebook
```
User → Jupyter cells → Python code → Results
```
- Best for: Beginners, interactive exploration
- Pros: Visual, step-by-step
- Cons: Hard to version control

### Mode 2: CLI (Recommended)
```
User → CLI command → Python script → Results
```
- Best for: Production, automation
- Pros: Reproducible, scriptable
- Cons: Less interactive

### Mode 3: Scripts
```
User → Helper script → CLI command → Results
```
- Best for: Batch processing, automation
- Pros: Can run multiple experiments
- Cons: Requires scripting knowledge

## Summary

This architecture provides:
- ✅ Clean separation of concerns
- ✅ Easy to modify and extend
- ✅ Reproducible results
- ✅ Fast iteration cycle
- ✅ Version control friendly
- ✅ Production ready
