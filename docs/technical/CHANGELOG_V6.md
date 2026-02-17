# Changelog - augmented_v6 Dataset Support

## Versione 2.0 - Dataset Gerarchico con Metadati

### 🎯 Obiettivo
Supportare il nuovo dataset `augmented_v6` con struttura gerarchica che permette di tracciare metadati dettagliati (food category, defect type, generator) per analisi approfondita degli errori.

### ✨ Nuove Funzionalità

#### 1. Parser Dataset Gerarchico
- **File**: `src/utils/data.py`
- **Funzione**: `parse_augmented_v6_dataset()`
- Scansiona ricorsivamente `augmented_v6/` e estrae metadati dal path
- Crea DataFrame con: path, label, source, quality, food_category, defect_type, generator
- Supporta struttura: `originali/buono|cattivo/<food>/<defect>/` e `modificate/<food>/<defect>/<generator>/`

#### 2. Split Stratificato Avanzato
- **File**: `src/utils/data.py`
- **Funzione**: `stratified_group_split_v6()`
- Stratifica per label E food_category
- Garantisce distribuzione bilanciata in train/val/test

#### 3. Dataset con Metadati
- **File**: `src/utils/data.py`
- **Classe**: `ImageBinaryDataset` (aggiornata)
- `__getitem__` ora restituisce: `(image, label, metadata_dict)`
- Metadati passati attraverso DataLoader per analisi

#### 4. Metriche per Gruppi
- **File**: `src/utils/metrics.py`
- **Funzione**: `compute_group_metrics()`
- Calcola accuracy, precision, recall, F1, AUC per ogni gruppo
- Supporta raggruppamento per: food_category, defect_type, generator, quality

#### 5. Analisi Errori
- **File**: `src/utils/metrics.py`
- **Funzione**: `get_top_errors()`
- Estrae top N falsi positivi/negativi
- Ordina per confidenza per identificare errori più "sicuri"

#### 6. Output CSV Dettagliati
- **File**: `src/train.py`, `src/eval.py`
- Genera 7 nuovi file CSV per ogni run:
  1. `predictions.csv` - Tutte le predizioni con metadati
  2. `group_metrics_food.csv` - Metriche per food category
  3. `group_metrics_defect.csv` - Metriche per defect type
  4. `group_metrics_generator.csv` - Metriche per generator
  5. `group_metrics_quality.csv` - Metriche per quality (originali)
  6. `top_false_positives.csv` - Top 50 FP con metadati
  7. `top_false_negatives.csv` - Top 50 FN con metadati

#### 7. Script di Analisi
- **File**: `scripts/analyze_results.py`
- Analizza risultati di un run
- Mostra: summary, group metrics, top errors, generator comparison
- Usage: `python scripts/analyze_results.py <run_name>`

### 📝 File Modificati

#### Core Code
- `src/utils/data.py` - Aggiunto parser v6, split stratificato, dataset con metadati
- `src/utils/metrics.py` - Aggiunte funzioni per group metrics e error analysis
- `src/train.py` - Integrato nuovo dataset, generazione CSV
- `src/eval.py` - Integrato nuovo dataset, generazione CSV

#### Configuration
- `configs/baseline.yaml` - Aggiornato dataset_root
- `configs/convnext_experiment.yaml` - Aggiornato dataset_root
- `configs/quick_test.yaml` - Aggiornato dataset_root

#### Scripts
- `scripts/Verifoto_Training.ipynb` - Aggiornato DATASET_ROOT
- `scripts/analyze_results.py` - Nuovo script per analisi

#### Documentation
- `docs/AUGMENTED_V6_DATASET.md` - Guida completa al nuovo dataset
- `docs/MIGRATION_V6.md` - Guida migrazione dal vecchio dataset
- `README.md` - Aggiornato output format
- `QUICKSTART.md` - Aggiunti comandi per analisi
- `CHANGELOG_V6.md` - Questo file

#### Dependencies
- `requirements.txt` - Aggiunto pandas>=2.0.0

### 🔄 Backward Compatibility

Il codice mantiene piena compatibilità con il vecchio dataset:
- Funzioni legacy (`find_class_dirs`, `stratified_group_split`) ancora disponibili
- Possibile usare vecchio dataset cambiando solo `dataset_root`
- Nessuna breaking change per utenti esistenti

### 📊 Esempio Output

Prima (vecchio dataset):
```
outputs/runs/<run_name>/
├── metrics.json
├── notes.md
└── *.png (plots)
```

Dopo (augmented_v6):
```
outputs/runs/<run_name>/
├── metrics.json
├── notes.md
├── predictions.csv                    # NEW
├── group_metrics_food.csv             # NEW
├── group_metrics_defect.csv           # NEW
├── group_metrics_generator.csv        # NEW
├── group_metrics_quality.csv          # NEW
├── top_false_positives.csv            # NEW
├── top_false_negatives.csv            # NEW
└── *.png (plots)
```

### 🎓 Use Cases

#### 1. Identificare Categorie Problematiche
```bash
python scripts/analyze_results.py my_run
# Mostra quali food categories hanno F1 basso
```

#### 2. Confrontare Generatori
```python
import pandas as pd
gen = pd.read_csv("outputs/runs/my_run/group_metrics_generator.csv")
print(gen.sort_values('f1'))
# Identifica quale generatore è più difficile da rilevare
```

#### 3. Analizzare Pattern negli Errori
```python
fp = pd.read_csv("outputs/runs/my_run/top_false_positives.csv")
print(fp.groupby('food_category').size())
# Capisce su quali categorie il modello sbaglia di più
```

#### 4. Debug Specifico
```python
pred = pd.read_csv("outputs/runs/my_run/predictions.csv")
errors = pred[pred['y_true'] != pred['y_pred']]
pizza_errors = errors[errors['food_category'] == 'pizza']
# Analizza errori solo su pizza
```

### 🚀 Quick Start

1. **Aggiorna dataset_root** in `configs/baseline.yaml`:
   ```yaml
   dataset_root: "/content/drive/MyDrive/DatasetVerifoto/augmented_v6"
   ```

2. **Run training**:
   ```bash
   python -m src.train --config configs/baseline.yaml --run_name "test_v6"
   ```

3. **Analyze results**:
   ```bash
   python scripts/analyze_results.py test_v6
   ```

### 📚 Documentation

- **Dataset Format**: `docs/AUGMENTED_V6_DATASET.md`
- **Migration Guide**: `docs/MIGRATION_V6.md`
- **Quick Reference**: `QUICKSTART.md`

### ⚠️ Breaking Changes

Nessuna! Il codice è completamente backward compatible.

### 🐛 Known Issues

Nessuno al momento.

### 🔮 Future Improvements

Possibili miglioramenti futuri:
1. Visualizzazioni interattive per group metrics
2. Report HTML automatico con grafici
3. Confronto automatico tra run diversi
4. Export per Weights & Biases o TensorBoard
5. Analisi temporale (performance nel tempo)

### 👥 Contributors

- Adattamento dataset gerarchico
- Implementazione metadata tracking
- Script di analisi avanzata
- Documentazione completa

### 📅 Release Date

2026-02-16

---

Per domande o supporto, consulta:
- `docs/AUGMENTED_V6_DATASET.md` - Formato dataset
- `docs/MIGRATION_V6.md` - Guida migrazione
- `QUICKSTART.md` - Comandi rapidi
