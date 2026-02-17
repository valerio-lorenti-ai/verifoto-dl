# Run: 2026-02-17_noK2

## Config
- Model: efficientnet_b0
- Image size: 224
- Batch size: 16
- Epochs: 5 (head) + 25 (finetune)

## Threshold Selection
- Optimal threshold (F1 on validation): 0.900
- Validation F1 at optimal threshold: 0.9501

## Results (threshold=0.900)
- Accuracy: 0.8737
- Precision: 0.8829
- Recall: 0.9195
- F1: 0.9008
- PR-AUC: 0.9717
- ROC-AUC: 0.9568

## Confusion Matrix
```
[[197  50]
 [ 33 377]]
```

Git commit: b73adc25709b49f985667226ab036669757d8b1f
