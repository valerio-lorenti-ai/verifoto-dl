# Migrazione al Dataset augmented_v6

Questa guida spiega come migrare dal vecchio dataset (real/fake) al nuovo `augmented_v6` con struttura gerarchica.

## Cosa Cambia

### Vecchio Dataset (exp_3_augmented_v6.1)
```
dataset/
├── images/
│   ├── NON_FRODE/  (o real/)
│   │   └── *.jpg
│   └── FRODE/  (o fake/)
│       └── *.jpg
```

### Nuovo Dataset (augmented_v6)
```
augmented_v6/
├── originali/
│   ├── buono/<food>/*.jpg
│   └── cattivo/<food>/<defect>/*.jpg
└── modificate/
    └── <food>/<defect>/<generator>/*.jpg
```

## Vantaggi del Nuovo Formato

1. **Metadati automatici**: Food category, defect type, generator estratti dal path
2. **Analisi granulare**: Metriche per ogni sottogruppo
3. **Error analysis**: Identificare pattern negli errori
4. **Debugging mirato**: Capire quali categorie/generatori causano problemi

## Passi per la Migrazione

### 1. Aggiorna Config

Modifica `configs/baseline.yaml`:

```yaml
# Prima
dataset_root: "/content/drive/MyDrive/DatasetVerifoto/images/exp_3_augmented_v6.1"

# Dopo
dataset_root: "/content/drive/MyDrive/DatasetVerifoto/augmented_v6"
```

### 2. Aggiorna Notebook Colab

Nel notebook `scripts/Verifoto_Training.ipynb`, prima cella:

```python
# Prima
DATASET_ROOT = "/content/drive/MyDrive/DatasetVerifoto/images/exp_3_augmented_v6.1"

# Dopo
DATASET_ROOT = "/content/drive/MyDrive/DatasetVerifoto/augmented_v6"
```

### 3. Verifica Struttura Dataset

Prima di lanciare il training, verifica che il dataset sia strutturato correttamente:

```python
from pathlib import Path

dataset_root = Path("/content/drive/MyDrive/DatasetVerifoto/augmented_v6")

# Verifica cartelle principali
assert (dataset_root / "originali").exists(), "Manca cartella originali/"
assert (dataset_root / "modificate").exists(), "Manca cartella modificate/"

# Verifica sottocartelle originali
assert (dataset_root / "originali" / "buono").exists(), "Manca originali/buono/"
assert (dataset_root / "originali" / "cattivo").exists(), "Manca originali/cattivo/"

print("✓ Struttura dataset corretta!")
```

### 4. Primo Training

Lancia un training di test:

```bash
python -m src.train \
    --config configs/quick_test.yaml \
    --run_name "test_v6" \
    --checkpoint_dir "/content/drive/MyDrive/verifoto_checkpoints"
```

Questo usa solo 1-2 epoche per verificare che tutto funzioni.

### 5. Verifica Output

Dopo il training, verifica che siano stati generati tutti i file:

```bash
ls outputs/runs/test_v6/

# Dovresti vedere:
# - predictions.csv
# - group_metrics_food.csv
# - group_metrics_defect.csv
# - group_metrics_generator.csv
# - group_metrics_quality.csv
# - top_false_positives.csv
# - top_false_negatives.csv
# - metrics.json
# - notes.md
# - cm.png, roc_curve.png, pr_curve.png, prob_dist.png
```

### 6. Analizza Risultati

Usa lo script di analisi:

```bash
python scripts/analyze_results.py test_v6
```

Questo mostra:
- Summary generale
- Metriche per food category
- Metriche per defect type
- Confronto tra generatori
- Top errori

## Nuovi Output Files

### predictions.csv
Tutte le predizioni con metadati completi:

```csv
path,y_true,y_prob,y_pred,source,quality,food_category,defect_type,generator
/path/to/img.jpg,0,0.12,0,originali,buono,pizza,,,
/path/to/img2.jpg,1,0.89,1,modificate,,,bruciato,gpt_image_1_5
```

### group_metrics_*.csv
Metriche aggregate per ogni gruppo:

```csv
food_category,n_samples,n_pos,n_neg,accuracy,precision,recall,f1,roc_auc,pr_auc,tp,fp,tn,fn
pizza,1000,500,500,0.95,0.94,0.96,0.95,0.98,0.97,480,30,470,20
carne,800,400,400,0.92,0.90,0.94,0.92,0.96,0.95,376,44,356,24
```

### top_false_*.csv
Top errori con metadati per debugging:

```csv
path,y_true,y_prob,y_pred,source,quality,food_category,defect_type,generator
/path/to/fp1.jpg,0,0.95,1,originali,buono,pizza,,,
/path/to/fp2.jpg,0,0.89,1,originali,cattivo,carne,marcio,,
```

## Analisi Avanzate

### Esempio 1: Trovare categorie problematiche

```python
import pandas as pd

# Carica metriche per food
food = pd.read_csv("outputs/runs/<run_name>/group_metrics_food.csv")

# Filtra categorie con basso F1
problematic = food[food['f1'] < 0.80].sort_values('f1')

print("Categorie problematiche:")
for _, row in problematic.iterrows():
    print(f"  {row['food_category']}: F1={row['f1']:.3f}, "
          f"samples={row['n_samples']}, FP={row['fp']}, FN={row['fn']}")
```

### Esempio 2: Confrontare generatori

```python
# Carica metriche per generator
gen = pd.read_csv("outputs/runs/<run_name>/group_metrics_generator.csv")

# Ordina per F1
gen_sorted = gen.sort_values('f1', ascending=False)

print("Performance per generatore:")
print(gen_sorted[['generator', 'n_samples', 'f1', 'precision', 'recall']])
```

### Esempio 3: Analizzare falsi positivi

```python
# Carica falsi positivi
fp = pd.read_csv("outputs/runs/<run_name>/top_false_positives.csv")

# Conta per food category
fp_by_food = fp.groupby('food_category').size().sort_values(ascending=False)
print("Falsi positivi per categoria:")
print(fp_by_food)

# Conta per quality
fp_by_quality = fp.groupby('quality').size()
print("\nFalsi positivi per qualità:")
print(fp_by_quality)
```

### Esempio 4: Identificare pattern negli errori

```python
# Carica predictions
pred = pd.read_csv("outputs/runs/<run_name>/predictions.csv")

# Filtra errori
errors = pred[pred['y_true'] != pred['y_pred']]

# Analizza distribuzione errori
print("Errori per source:")
print(errors.groupby('source').size())

print("\nErrori per food_category:")
print(errors.groupby('food_category').size().sort_values(ascending=False))

print("\nErrori per defect_type:")
print(errors.groupby('defect_type').size().sort_values(ascending=False))
```

## Backward Compatibility

Il codice mantiene backward compatibility con il vecchio dataset:

- Le funzioni `find_class_dirs()`, `list_images_in_dir()`, `stratified_group_split()` sono ancora disponibili
- Puoi usare il vecchio dataset cambiando solo `dataset_root` nel config

Tuttavia, con il vecchio dataset NON avrai:
- Metadati dettagliati
- Group metrics
- Error analysis avanzata

## Troubleshooting

### "Nessuna immagine trovata"

Verifica che:
1. Il path in `dataset_root` sia corretto
2. Le cartelle `originali/` e `modificate/` esistano
3. Ci siano file `.jpg` nelle sottocartelle

### "Struttura non riconosciuta"

Il parser si aspetta questa gerarchia:
- `originali/buono/<food>/`
- `originali/cattivo/<food>/<defect>/`
- `modificate/<food>/<defect>/<generator>/`

Se la tua struttura è diversa, adatta il parser in `src/utils/data.py`.

### "Group metrics vuoti"

Alcuni gruppi potrebbero essere vuoti se:
- Non ci sono campioni per quel gruppo nel test set
- Il valore è `None` o `NaN` nei metadati

Questo è normale e non causa problemi.

## Checklist Migrazione

- [ ] Dataset strutturato correttamente su Drive
- [ ] Config aggiornato con nuovo `dataset_root`
- [ ] Notebook Colab aggiornato
- [ ] Primo training di test completato
- [ ] Verificati tutti i file CSV generati
- [ ] Script di analisi funzionante
- [ ] Risultati committati su GitHub

## Prossimi Passi

Dopo la migrazione:

1. **Analizza i risultati**: Usa `scripts/analyze_results.py`
2. **Identifica problemi**: Guarda group metrics e top errors
3. **Itera**: Migliora il modello basandoti sui pattern trovati
4. **Documenta**: Annota i findings nei notes.md

## Supporto

Per domande o problemi:
1. Controlla `docs/AUGMENTED_V6_DATASET.md` per dettagli sul formato
2. Rivedi esempi di analisi in questa guida
3. Verifica che la struttura del dataset sia corretta
