# Run: 2026-03-14_pico_plus_exp3_aug

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
- Optimal threshold: 0.200
- Validation score: 0.9210

## Results (threshold=0.200)
- Accuracy: 0.8926
- Precision: 0.8863
- Recall: 0.9300
- F1: 0.9076
- PR-AUC: 0.9779
- ROC-AUC: 0.9687

## Confusion Matrix
```
[[497  92]
 [ 54 717]]
```

Git commit: b8c2fe041303aa427b41216b91340a272f78c094
