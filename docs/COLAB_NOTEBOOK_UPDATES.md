# Modifiche Necessarie per Notebook Colab

## Overview

Per usare il nuovo split `domain_aware`, devi modificare **UNA SOLA CELLA** nel notebook Colab.

## File da Modificare

`scripts/Verifoto_Training_V2.ipynb`

## Modifiche Richieste

### ✅ CELLA 1: EXPERIMENT CONFIGURATION

**PRIMA (linee 8-12):**
```python
EXPERIMENT_NAME = "2026-02-17_baseline_test"  # Unique name for this experiment
DATASET_NAME = "exp_3_augmented_v6.2_noK"     # Dataset folder name in Drive
CONFIG_FILE = "baseline.yaml"                  # Config file to use (in configs/)
```

**DOPO (aggiungi una riga):**
```python
EXPERIMENT_NAME = "2026-02-20_domain_aware_test"  # Unique name for this experiment
DATASET_NAME = "exp_3_augmented_v6.1_categorized"  # Dataset folder name in Drive
CONFIG_FILE = "domain_aware_baseline.yaml"         # Config file to use (in configs/)
```

**OPPURE** se vuoi usare un config esistente, modifica solo il nome:
```python
EXPERIMENT_NAME = "2026-02-20_domain_aware_test"
DATASET_NAME = "exp_3_augmented_v6.1_categorized"
CONFIG_FILE = "baseline.yaml"  # Useremo questo e aggiungeremo split_strategy
```

### ✅ CELLA 7: UPDATE CONFIG (OPZIONALE - solo se usi config esistente)

**PRIMA:**
```python
# Update config file with correct dataset path
import yaml

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

config['dataset_root'] = DATASET_ROOT

with open(CONFIG_PATH, 'w') as f:
    yaml.dump(config, f)

print(f"✓ Config updated: {CONFIG_PATH}")
print(f"  dataset_root: {config['dataset_root']}")
```

**DOPO (aggiungi split_strategy):**
```python
# Update config file with correct dataset path and split strategy
import yaml

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

config['dataset_root'] = DATASET_ROOT
config['split_strategy'] = 'domain_aware'  # ← NUOVA RIGA
config['split_include_food'] = False       # ← NUOVA RIGA

with open(CONFIG_PATH, 'w') as f:
    yaml.dump(config, f)

print(f"✓ Config updated: {CONFIG_PATH}")
print(f"  dataset_root: {config['dataset_root']}")
print(f"  split_strategy: {config['split_strategy']}")  # ← NUOVA RIGA
print(f"  split_include_food: {config['split_include_food']}")  # ← NUOVA RIGA
```

## Opzione Più Semplice: Usa Config Dedicato

**RACCOMANDATO:** Usa il nuovo config `domain_aware_baseline.yaml` che ha già tutto configurato.

Nella CELLA 1, imposta:
```python
CONFIG_FILE = "domain_aware_baseline.yaml"
```

Poi **NON SERVE MODIFICARE** la cella 7 (update config).

## Verifica Output

Quando esegui il training, nella cella di training vedrai:

```
================================================================================
DOMAIN-AWARE GROUP-BASED SPLIT (No Data Leakage)
================================================================================
Strategy: Stratified by label|source|generator
Unique photos: 1234
Unique strata: 8

1) SPLIT COUNTS:
  Train: 864 photos (2156 images, 0.523 pos rate)
  Val:   185 photos (463 images, 0.518 pos rate)
  Test:  185 photos (467 images, 0.521 pos rate)

2) SPLIT x LABEL:
  Train: label=0: 1028 (47.7%)  label=1: 1128 (52.3%)
  Val  : label=0:  223 (48.2%)  label=1:  240 (51.8%)
  Test : label=0:  224 (48.0%)  label=1:  243 (52.0%)

3) SPLIT x SOURCE:
  originali           : Train= 512 (70.2%)  Val= 108 (14.8%)  Test= 109 (15.0%)
  kaggle_vale         : Train= 516 (69.8%)  Val= 115 (15.6%)  Test= 115 (15.6%)
  modificate          : Train= 980 (70.1%)  Val= 210 (15.0%)  Test= 208 (14.9%)

4) SPLIT x GENERATOR:
  gpt_image_1_mini    : Train= 490 (70.0%)  Val= 105 (15.0%)  Test= 105 (15.0%)
  gpt_image_1_5       : Train= 490 (70.1%)  Val= 105 (15.0%)  Test= 103 (14.9%)

✓ No overlap verified - data leakage prevented
✓ Domain balance verified - all sources/generators present in all splits
================================================================================
```

Se vedi questo output, significa che il domain-aware split funziona correttamente!

## Riepilogo Modifiche

### Opzione A: Usa Config Dedicato (RACCOMANDATO)
1. Cella 1: `CONFIG_FILE = "domain_aware_baseline.yaml"`
2. Nessun'altra modifica necessaria

### Opzione B: Modifica Config Esistente
1. Cella 1: Imposta nome esperimento e dataset
2. Cella 7: Aggiungi `config['split_strategy'] = 'domain_aware'`

## Test Rapido

Per testare che tutto funzioni, esegui un quick test:

```python
# In una nuova cella dopo il setup
from src.utils.data import parse_augmented_v6_dataset, domain_aware_group_split_v1

df = parse_augmented_v6_dataset(DATASET_ROOT)
train_df, val_df, test_df = domain_aware_group_split_v1(df, seed=42)

print(f"✓ Split test successful!")
print(f"  Train: {len(train_df)} images")
print(f"  Val: {len(val_df)} images")
print(f"  Test: {len(test_df)} images")
```

Se questo funziona senza errori, sei pronto per il training completo!
