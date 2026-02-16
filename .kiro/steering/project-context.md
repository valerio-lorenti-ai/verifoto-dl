---
inclusion: auto
---

# Verifoto-DL Project Context

## Parent Project: Verifoto-AI
SaaS web app for restaurants to detect fraudulent food delivery images. Customers manipulate photos (fake mold, burns, AI-generated defects) to get undeserved refunds. Verifoto provides automated forensic analysis to reduce economic losses.

**Business Model**: B2B SaaS for restaurants and dark kitchens
**Value**: Specific to food delivery, optimized for low false positives, explainable reports

## This Project: Verifoto-DL
Deep learning training pipeline for the fraud detection model. Binary classification: original (0) vs modified (1) food images.

## Current State
- **Dataset**: augmented_v6 (hierarchical: originali/buono|cattivo, modificate/generator)
- **Architecture**: Two-phase training (head-only → full finetune)
- **Model**: EfficientNet-B0 baseline, ConvNeXt experiments
- **Output**: Predictions + group metrics CSV for error analysis
- **Goal**: High precision (low false positives) - better to miss fraud than accuse legitimate complaints

## Key Directories
```
src/              - Core training/eval code
configs/          - YAML configurations
scripts/          - Helper scripts + Colab notebook
docs/             - User documentation
.kiro/agent/      - Agent-maintained context (NOT for users)
outputs/runs/     - Training results (versioned on GitHub)
checkpoints/      - Model weights (Drive, NOT versioned)
```

## Critical Files
- `src/train.py` - Main training script
- `src/eval.py` - Evaluation script
- `src/utils/data.py` - Dataset parser for augmented_v6
- `configs/baseline.yaml` - Default configuration
- `docs/AUGMENTED_V6_DATASET.md` - Dataset format reference

## Workflow
1. Edit code locally (Kiro)
2. Commit + push to GitHub
3. Pull in Colab → train with GPU
4. Results saved to outputs/runs/
5. Commit results → pull locally → analyze

## Tech Stack
- PyTorch + timm for models
- Pandas for metadata tracking
- scikit-learn for metrics
- Google Colab for GPU training
- Google Drive for checkpoints/dataset

## Important Constraints
- Training is BINARY (0=original, 1=modified)
- Metadata used ONLY for analysis, NOT training
- Conservative approach: optimize for precision over recall (avoid false positives)
- GitHub: code + lightweight results only
- Drive: checkpoints + dataset
- Backward compatible with old dataset structure

## Production Context
This model will be deployed in Verifoto-AI web app where:
- Restaurants upload suspicious images
- Model provides confidence score + technical indicators
- LLM generates human-readable explanation
- Report used for internal decision-making
- False positives damage trust → precision is critical
