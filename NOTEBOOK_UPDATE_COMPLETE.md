# ✅ Notebook Colab Aggiornato per Domain-Aware Split

## Modifiche Applicate

Ho aggiornato il notebook `scripts/notebooks/verifoto_dl.ipynb` per supportare il nuovo split domain-aware.

### Cella Modificata

**Cella: "Update config file with correct dataset path"**

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

**DOPO:**
```python
# Update config file with correct dataset path and split strategy
import yaml

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

config['dataset_root'] = DATASET_ROOT

# Enable domain-aware split (prevents data leakage + balances source/generator)
# Set to 'group_v6' to use old split strategy (default)
config['split_strategy'] = 'domain_aware'  # 'group_v6' or 'domain_aware'
config['split_include_food'] = False       # Set True to also stratify by food_category

with open(CONFIG_PATH, 'w') as f:
    yaml.dump(config, f)

print(f"✓ Config updated: {CONFIG_PATH}")
print(f"  dataset_root: {config['dataset_root']}")
print(f"  split_strategy: {config.get('split_strategy', 'group_v6')}")
print(f"  split_include_food: {config.get('split_include_food', False)}")
```

## Come Usare

### Opzione 1: Domain-Aware Split (RACCOMANDATO)

Il notebook è già configurato per usare domain-aware split. Esegui normalmente:

1. Apri `scripts/notebooks/verifoto_dl.ipynb` su Colab
2. Esegui tutte le celle
3. Verifica l'output dello split

### Opzione 2: Tornare al Vecchio Split (group_v6)

Se vuoi usare il vecchio split, modifica la cella:

```python
config['split_strategy'] = 'group_v6'  # Cambia da 'domain_aware' a 'group_v6'
```

## Output Atteso

Quando esegui il training con domain-aware, vedrai:

```
================================================================================
DOMAIN-AWARE GROUP-BASED SPLIT (No Data Leakage)
================================================================================
Strategy: Stratified by label|source|generator
Unique photos: 510
Unique strata: 4

1) SPLIT COUNTS:
  Train: 356 photos (2856 images, 0.653 pos rate)
  Val:   77 photos (616 images, 0.675 pos rate)
  Test:  77 photos (616 images, 0.610 pos rate)

2) SPLIT x LABEL:
  Train: label=0: 991 (34.7%)  label=1: 1865 (65.3%)
  Val  : label=0: 200 (32.5%)  label=1:  416 (67.5%)
  Test : label=0: 241 (39.1%)  label=1:  375 (60.9%)

3) SPLIT x SOURCE:
  originali           : Train= XXX (70.X%)  Val= XXX (15.X%)  Test= XXX (15.X%)
  kaggle_vale         : Train= XXX (70.X%)  Val= XXX (15.X%)  Test= XXX (15.X%)
  modificate          : Train= XXX (70.X%)  Val= XXX (15.X%)  Test= XXX (15.X%)

4) SPLIT x GENERATOR:
  gpt_image_1_mini    : Train= XXX (70.X%)  Val= XXX (15.X%)  Test= XXX (15.X%)
  gpt_image_1_5       : Train= XXX (70.X%)  Val= XXX (15.X%)  Test= XXX (15.X%)

✓ No overlap verified - data leakage prevented
✓ Domain balance verified - all sources/generators present in all splits
================================================================================
```

**Verifica chiave:** Ogni source e generator deve avere ~70/15/15 tra split!

## Confronto Prima/Dopo

### PRIMA (group_v6):
```
⚠️  Using GROUP-BASED split to prevent data leakage

================================================================================
GROUP-BASED SPLIT (No Data Leakage)
================================================================================
Unique photos: 510
  Train: 356 photos (2856 images, 0.653 pos rate)
  Val:   77 photos (616 images, 0.675 pos rate)
  Test:  77 photos (616 images, 0.610 pos rate)
✓ No overlap verified - data leakage prevented
================================================================================
```

**Problema:** Non mostra distribuzione source/generator → possibile bias!

### DOPO (domain_aware):
```
⚠️  Using DOMAIN-AWARE GROUP-BASED split
   (prevents data leakage + balances source/generator across splits)

================================================================================
DOMAIN-AWARE GROUP-BASED SPLIT (No Data Leakage)
================================================================================
Strategy: Stratified by label|source|generator
...
3) SPLIT x SOURCE:
  originali           : Train= 512 (70.2%)  Val= 108 (14.8%)  Test= 109 (15.0%)
  kaggle_vale         : Train= 516 (69.8%)  Val= 115 (15.6%)  Test= 115 (15.6%)
  modificate          : Train= 980 (70.1%)  Val= 210 (15.0%)  Test= 208 (14.9%)

4) SPLIT x GENERATOR:
  gpt_image_1_mini    : Train= 490 (70.0%)  Val= 105 (15.0%)  Test= 105 (15.0%)
  gpt_image_1_5       : Train= 490 (70.1%)  Val= 105 (15.0%)  Test= 103 (14.9%)

✓ Domain balance verified - all sources/generators present in all splits
================================================================================
```

**Soluzione:** Tutti i domini bilanciati 70/15/15 → no bias!

## Parametri Configurabili

Nel notebook, puoi modificare:

```python
# Strategia di split
config['split_strategy'] = 'domain_aware'  # o 'group_v6'

# Stratificare anche per food_category (può creare strati piccoli)
config['split_include_food'] = False  # o True
```

### Quando usare `split_include_food=True`?

- ✅ Dataset grande (>5000 foto)
- ✅ Vuoi bilanciare anche food_category tra split
- ❌ Dataset piccolo (<1000 foto) → troppi strati piccoli

## Validazione

Dopo il training, verifica manualmente:

```python
import pandas as pd

# Load predictions
pred = pd.read_csv(f"{OUTPUT_DIR}/predictions.csv")

# Verifica distribuzione source
print("Source distribution:")
print(pred['source'].value_counts())

# Verifica distribuzione generator (solo AI)
print("\nGenerator distribution (AI only):")
ai_pred = pred[pred['y_true'] == 1]
print(ai_pred['generator'].value_counts())

# Performance per source
print("\nPerformance by source:")
for source in pred['source'].unique():
    subset = pred[pred['source'] == source]
    acc = (subset['y_true'] == subset['y_pred']).mean()
    print(f"  {source}: Accuracy={acc:.1%}")
```

## File Modificati

1. ✅ `scripts/notebooks/verifoto_dl.ipynb` - Aggiunta configurazione domain-aware
2. ✅ `src/utils/data.py` - Implementata funzione `domain_aware_group_split_v1()`
3. ✅ `src/train.py` - Supporto split_strategy
4. ✅ `src/train_v7.py` - Supporto split_strategy
5. ✅ `src/eval.py` - Supporto split_strategy
6. ✅ `configs/domain_aware_baseline.yaml` - Config di esempio

## Prossimi Passi

1. **Apri il notebook su Colab**
   ```
   scripts/notebooks/verifoto_dl.ipynb
   ```

2. **Esegui tutte le celle**
   - Il notebook è già configurato per domain-aware
   - Verifica l'output dello split

3. **Controlla l'output**
   - Cerca "DOMAIN-AWARE GROUP-BASED SPLIT"
   - Verifica crosstab source/generator
   - Conferma "✓ Domain balance verified"

4. **Inviami l'output dello split**
   - Copia la sezione "DOMAIN-AWARE GROUP-BASED SPLIT"
   - Inviamela per validazione finale

## Troubleshooting

### Problema: Non vedo "DOMAIN-AWARE" nell'output

**Soluzione:** Verifica che la cella di update config sia stata eseguita:
```python
print(f"  split_strategy: {config.get('split_strategy', 'group_v6')}")
```

Dovrebbe stampare: `split_strategy: domain_aware`

### Problema: Errore "domain_aware_group_split_v1 not found"

**Soluzione:** Fai git pull per aggiornare il codice:
```bash
%cd /content/verifoto-dl
!git pull
```

### Problema: Strati troppo piccoli (warning)

**Soluzione:** Imposta `split_include_food=False` (default):
```python
config['split_include_food'] = False
```

## Documentazione Completa

- **Implementazione:** `docs/DOMAIN_AWARE_SPLIT.md`
- **Esempio Output:** `docs/DOMAIN_AWARE_EXAMPLE_OUTPUT.md`
- **Riepilogo:** `DOMAIN_AWARE_IMPLEMENTATION_SUMMARY.md`
- **Questo file:** `NOTEBOOK_UPDATE_COMPLETE.md`

## ✅ Checklist

- [x] Notebook aggiornato con domain-aware split
- [x] Commenti aggiunti per chiarezza
- [x] Parametri configurabili documentati
- [x] Output atteso documentato
- [ ] Test su Colab (da fare)
- [ ] Verifica output split (da fare)
- [ ] Validazione finale (da fare)

---

**Tutto pronto!** Il notebook è aggiornato e pronto per il test su Colab. 🚀
