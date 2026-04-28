# Run: 2026-04-28_pico_exp3_datasetmix_pw_pr_w_r_id

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
- Optimal threshold: 0.550
- Validation score: 0.8904

## Results (threshold=0.550)
- Accuracy: 0.8672
- Precision: 0.8778
- Recall: 0.9064
- F1: 0.8919
- PR-AUC: 0.9611
- ROC-AUC: 0.9395

## Confusion Matrix
```
[[ 615  147]
 [ 109 1056]]
```

Git commit: 5a8112a9911f8c66ad986ae8e46f1630607ec2e0
