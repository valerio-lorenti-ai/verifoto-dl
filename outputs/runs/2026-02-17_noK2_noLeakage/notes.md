# Run: 2026-02-17_noK2_noLeakage

## Config
- Model: efficientnet_b0
- Image size: 224
- Batch size: 16
- Epochs: 5 (head) + 25 (finetune)

## Threshold Selection
- Optimal threshold (F1 on validation): 0.550
- Validation F1 at optimal threshold: 0.8372

## Results (threshold=0.550)
- Accuracy: 0.7062
- Precision: 0.6931
- Recall: 0.9309
- F1: 0.7946
- PR-AUC: 0.7784
- ROC-AUC: 0.7511

## Confusion Matrix
```
[[ 85 155]
 [ 26 350]]
```

Git commit: f098a29219ad87ccf4c12bc8c717d7fbbbd5a9a3
