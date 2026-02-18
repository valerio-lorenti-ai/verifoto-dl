# Run: 2026-02-18_convnext_v7_noLeakage

## Config
- Model: convnext_tiny (27.8M params)
- Image size: 224
- Batch size: 12
- Epochs: 5 (head) + 35 (finetune)
- Loss: weighted_focal
- Augmentation: strong
- Differential augmentation: True

## Threshold Selection
- Strategy: cost_sensitive
- Optimal threshold: 0.700
- Validation score: 166.0000

## Results (threshold=0.700)
- Accuracy: 0.6948
- Precision: 0.7582
- Recall: 0.7340
- F1: 0.7459
- PR-AUC: 0.8998
- ROC-AUC: 0.8335

## Confusion Matrix
```
[[152  88]
 [100 276]]
```

Git commit: 9094aa3d003131596cf8731967e268417c55f8e0
