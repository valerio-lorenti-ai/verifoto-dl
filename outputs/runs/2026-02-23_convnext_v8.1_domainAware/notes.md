# Run: 2026-02-23_convnext_v8.1_domainAware

## Config
- Model: convnext_tiny (27.8M params)
- Image size: 224
- Batch size: 12
- Epochs: 5 (head) + 35 (finetune)
- Loss: weighted_focal
- Augmentation: strong
- Differential augmentation: True

## Threshold Selection
- Strategy: f1
- Optimal threshold: 0.650
- Validation score: 0.8699

## Results (threshold=0.650)
- Accuracy: 0.8338
- Precision: 0.8557
- Recall: 0.8300
- F1: 0.8426
- PR-AUC: 0.9318
- ROC-AUC: 0.9185

## Confusion Matrix
```
[[290  56]
 [ 68 332]]
```

Git commit: ac9ebf8f69d715d571c59fe53605bb2a8e773d43
