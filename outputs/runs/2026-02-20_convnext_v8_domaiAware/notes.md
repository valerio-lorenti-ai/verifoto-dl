# Run: 2026-02-20_convnext_v8_domaiAware

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
- Optimal threshold: 0.900
- Validation score: 107.0000

## Results (threshold=0.900)
- Accuracy: 0.7465
- Precision: 0.7915
- Recall: 0.6887
- F1: 0.7366
- PR-AUC: 0.8709
- ROC-AUC: 0.8819

## Confusion Matrix
```
[[311  74]
 [127 281]]
```

Git commit: 15e296bf58095c8052be8472d7bdad8023bdd729
