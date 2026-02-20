# Domain-Aware Split: Esempio Output Atteso

## Scenario di Test

Dataset con:
- 500 foto originali "your_original"
- 463 foto originali "kaggle_vale"
- 600 foto modificate "your_ai" (300 GPT-1-mini, 300 GPT-1.5)

Totale: 1563 foto uniche

## Output Completo Atteso

```
=== Loading dataset ===
Scanning dataset: 100%|██████████| 4892/4892 [00:02<00:00, 2156.34it/s]

Dataset caricato: 4892 immagini
  - Originali (label=0): 2156
  - Modificate (label=1): 2736
  - Food categories: 45
  - Defect types: 12
  - Generators: 2

⚠️  Using DOMAIN-AWARE GROUP-BASED split
   (prevents data leakage + balances source/generator across splits)

================================================================================
DOMAIN-AWARE GROUP-BASED SPLIT (No Data Leakage)
================================================================================
Strategy: Stratified by label|source|generator
Unique photos: 1563
Unique strata: 4

1) SPLIT COUNTS:
  Train: 1094 photos (3425 images, 0.560 pos rate)
  Val:   234 photos (733 images, 0.558 pos rate)
  Test:  235 photos (734 images, 0.559 pos rate)

2) SPLIT x LABEL:
  Train: label=0: 1507 (44.0%)  label=1: 1918 (56.0%)
  Val  : label=0:  324 (44.2%)  label=1:  409 (55.8%)
  Test : label=0:  325 (44.3%)  label=1:  409 (55.7%)

3) SPLIT x SOURCE:
  originali           : Train= 697 (70.1%)  Val= 149 (15.0%)  Test= 148 (14.9%)
  kaggle_vale         : Train= 324 (70.0%)  Val=  69 (14.9%)  Test=  70 (15.1%)
  modificate          : Train=1916 (70.0%)  Val= 410 (15.0%)  Test= 410 (15.0%)

4) SPLIT x GENERATOR:
  gpt_image_1_mini    : Train= 630 (70.0%)  Val= 135 (15.0%)  Test= 135 (15.0%)
  gpt_image_1_5       : Train= 630 (70.1%)  Val= 134 (14.9%)  Test= 136 (15.1%)

6) STRATA PROPORTION VALIDATION:
  Large strata (n≥10): 4
  Max deviation from target: Train=0.1%  Val=0.1%  Test=0.1%
  Small strata (n<10): 0

✓ No overlap verified - data leakage prevented
✓ Domain balance verified - all sources/generators present in all splits
================================================================================

=== Model: efficientnet_b0 ===
Parameters: 4,011,391 (4.0M)

=== Loss: bce ===
pos_weight: 0.79

=== Phase 1: Head-only ===
train: 100%|██████████| 214/214 [02:15<00:00,  1.58it/s]
[Head 1/5] loss=0.5234 val_pr_auc=0.8456 val_f1=0.7823
...
```

## Interpretazione Output

### 1) SPLIT COUNTS
- ✅ Train ~70%, Val ~15%, Test ~15% (come atteso)
- ✅ Positive rate simile tra split (~56%)

### 2) SPLIT x LABEL
- ✅ Proporzione label=0/label=1 simile tra split
- ✅ Nessun bias di label tra train/val/test

### 3) SPLIT x SOURCE
- ✅ **CRITICO:** Ogni source presente in tutti gli split con ~70/15/15
- ✅ Kaggle NON concentrato solo in train o test
- ✅ Originali e modificate bilanciate

### 4) SPLIT x GENERATOR
- ✅ **CRITICO:** Ogni generator presente in tutti gli split con ~70/15/15
- ✅ GPT-1-mini e GPT-1.5 bilanciati
- ✅ Modello vede entrambi i generatori in training

### 6) STRATA VALIDATION
- ✅ Deviazione <1% per strati grandi (ottimo!)
- ✅ Nessuno strato piccolo (<10 foto)

## Confronto con Split Precedente (group_v6)

### PRIMA (group_based_split_v6):
```
3) SPLIT x SOURCE:
  originali           : Train= 850 (85.4%)  Val=  72 ( 7.2%)  Test=  72 ( 7.2%)  ❌
  kaggle_vale         : Train= 100 (21.6%)  Val= 181 (39.1%)  Test= 182 (39.3%)  ❌
  modificate          : Train=1980 (72.3%)  Val= 378 (13.8%)  Test= 378 (13.8%)  ✓

4) SPLIT x GENERATOR:
  gpt_image_1_mini    : Train= 850 (94.4%)  Val=  25 ( 2.8%)  Test=  25 ( 2.8%)  ❌
  gpt_image_1_5       : Train= 100 (11.1%)  Val= 400 (44.4%)  Test= 400 (44.4%)  ❌
```

**Problemi:**
- ❌ Kaggle quasi tutto in val/test (39% ciascuno!)
- ❌ GPT-1-mini quasi tutto in train (94%)
- ❌ GPT-1.5 quasi tutto in val/test (44% ciascuno)
- ❌ Modello impara "segnali Kaggle" e "segnali GPT-1-mini"

### DOPO (domain_aware_group_split_v1):
```
3) SPLIT x SOURCE:
  originali           : Train= 697 (70.1%)  Val= 149 (15.0%)  Test= 148 (14.9%)  ✓
  kaggle_vale         : Train= 324 (70.0%)  Val=  69 (14.9%)  Test=  70 (15.1%)  ✓
  modificate          : Train=1916 (70.0%)  Val= 410 (15.0%)  Test= 410 (15.0%)  ✓

4) SPLIT x GENERATOR:
  gpt_image_1_mini    : Train= 630 (70.0%)  Val= 135 (15.0%)  Test= 135 (15.0%)  ✓
  gpt_image_1_5       : Train= 630 (70.1%)  Val= 134 (14.9%)  Test= 136 (15.1%)  ✓
```

**Risultati:**
- ✅ Tutti i domini bilanciati 70/15/15
- ✅ Modello vede tutti i generatori in training
- ✅ Test set rappresentativo di tutti i domini
- ✅ Metriche più realistiche e generalizzabili

## Warning Attesi

### Small Strata Warning (se include_food=True)
```
⚠️  WARNING: 12 strata have <10 photos - proportions may be imprecise
   0|kaggle_vale|none|pasta_carbonara: n=3 (train=2, val=1, test=0)
   1|your_ai|gpt_image_1_mini|riso_paella: n=5 (train=4, val=1, test=0)
   ...
```

**Soluzione:** Usa `split_include_food: false` (default) per evitare strati troppo piccoli.

### Large Deviation Warning (raro)
```
⚠️  WARNING: Some large strata deviate >5% from target proportions
```

**Causa:** Strato con numero di foto non divisibile per 70/15/15 (es: 13 foto → 9/2/2)
**Soluzione:** Normale per strati con 10-20 foto, ignorabile se <5 strati

## Validazione Manuale

Dopo il training, verifica manualmente:

1. **Controlla predictions.csv:**
```python
import pandas as pd
pred = pd.read_csv("outputs/runs/YOUR_RUN/predictions.csv")

# Verifica distribuzione source
print(pred['source'].value_counts())

# Verifica distribuzione generator (solo label=1)
print(pred[pred['y_true']==1]['generator'].value_counts())
```

2. **Controlla group_metrics_generator.csv:**
```python
gen = pd.read_csv("outputs/runs/YOUR_RUN/group_metrics_generator.csv")
print(gen[['generator', 'n_samples', 'f1', 'precision', 'recall']])
```

Se F1 simile tra generatori → modello generalizza bene!

## Metriche Attese

Con domain-aware split, aspettati:

- **F1 leggermente più basso** (~2-3%) rispetto a group_v6
  - Motivo: Test set più difficile (include tutti i domini)
  - Questo è POSITIVO: metriche più realistiche

- **Precision/Recall più bilanciati** tra source/generator
  - group_v6: F1=0.95 su Kaggle, F1=0.75 su your_original
  - domain_aware: F1=0.88 su entrambi

- **Generalizzazione migliore** su dati esterni
  - Modello non overfitta su segnali di dominio
  - Performance più stabile su nuovi dataset
