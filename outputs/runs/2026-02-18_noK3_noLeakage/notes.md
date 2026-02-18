# Run: 2026-02-18_noK3_noLeakage

## Config
- Model: efficientnet_b0
- Image size: 224
- Batch size: 16
- Epochs: 5 (head) + 25 (finetune)

## Threshold Selection
- Optimal threshold (F1 on validation): 0.100
- Validation F1 at optimal threshold: 0.8101

## Results (threshold=0.100)
- Accuracy: 0.7013
- Precision: 0.6720
- Recall: 0.9973
- F1: 0.8030
- PR-AUC: 0.7803
- ROC-AUC: 0.7589

## Confusion Matrix
```
[[ 57 183]
 [  1 375]]
```

Git commit: 1301a1ada3fc80e6e096ea13e457aef9a0696706
