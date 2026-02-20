# Domain-Aware Split Implementation

## Overview

Implementato nuovo split strategy `domain_aware_group_split_v1()` che previene sia data leakage che domain bias.

## Problema Risolto

**Prima (group_based_split_v6):**
- Stratificazione: `label + food_category`
- ❌ Kaggle poteva finire 90% in train, 10% in test
- ❌ GPT-1.5 e GPT-mini non bilanciati tra split
- ❌ Modello impara "segnali di dominio" invece di artefatti AI

**Dopo (domain_aware_group_split_v1):**
- Stratificazione: `label + source + generator` (opzionalmente + food_category)
- ✅ Ogni source presente in train/val/test con proporzioni simili
- ✅ Ogni generator presente in train/val/test con proporzioni simili
- ✅ Modello impara artefatti AI, non bias di dominio

## Implementazione

### Nuova Funzione

**File:** `src/utils/data.py`

```python
domain_aware_group_split_v1(
    df, 
    train_ratio=0.70, 
    val_ratio=0.15, 
    test_ratio=0.15, 
    seed=42, 
    include_food=False
)
```

**Features:**
1. Group-based per photo_id (no data leakage)
2. Stratificazione per label|source|generator
3. RNG robusto (seed + hash(key) per evitare pattern)
4. Fallback per valori mancanti:
   - source missing → "unknown"
   - generator missing → "none" (se label=0) o "unknown_generator"
5. Validazioni obbligatorie:
   - Assert no overlap tra split
   - Crosstab split x label/source/generator/food_category
   - Verifica proporzioni per strato (warning se deviazione >5%)

### Config YAML

Aggiungi al tuo config:

```yaml
# Split strategy
split_strategy: domain_aware  # "group_v6" (default) o "domain_aware"
split_include_food: false     # Se true, stratifica anche per food_category
```

### File Modificati

1. `src/utils/data.py` - Aggiunta `domain_aware_group_split_v1()`
2. `src/train.py` - Supporto per split_strategy config
3. `src/train_v7.py` - Supporto per split_strategy config
4. `src/eval.py` - Supporto per split_strategy config
5. `configs/domain_aware_baseline.yaml` - Config di esempio

## Output Atteso

Quando esegui training con `split_strategy: domain_aware`, vedrai:

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

## Uso

### Training

```bash
python src/train.py \
  --config configs/domain_aware_baseline.yaml \
  --run_name 2026-02-20_domain_aware_test
```

### Evaluation

```bash
python src/eval.py \
  --config configs/domain_aware_baseline.yaml \
  --run_name eval_domain_aware \
  --checkpoint_path checkpoints/2026-02-20_domain_aware_test/best.pt \
  --threshold 0.5
```

## Validazioni

La funzione esegue automaticamente:

1. **No Overlap:** Assert che train/val/test non condividano photo_id
2. **Crosstab:** Stampa distribuzione di label/source/generator per split
3. **Proportion Check:** Verifica che ogni strato rispetti ~70/15/15
4. **Small Strata Warning:** Avvisa se strati con <10 foto (proporzioni imprecise)

## Retrocompatibilità

`group_based_split_v6()` rimane invariato. Per usarlo:

```yaml
split_strategy: group_v6  # o ometti (default)
```

## Note

- `include_food=True` può creare molti strati piccoli (es: "0|kaggle|none|pasta" con 3 foto)
- Per dataset piccoli, usa `include_food=False` (default)
- Per dataset grandi (>5000 foto), `include_food=True` può migliorare bilanciamento
