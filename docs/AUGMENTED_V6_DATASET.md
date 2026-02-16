# Augmented V6 Dataset Guide

## Struttura Dataset

Il nuovo dataset `augmented_v6` ha una struttura gerarchica che permette di tracciare metadati dettagliati per ogni immagine.

### Struttura Directory

```
augmented_v6/
├── originali/
│   ├── buono/
│   │   ├── pizza/
│   │   │   ├── img001.jpg
│   │   │   ├── img002.jpg
│   │   │   └── ...
│   │   ├── carne/
│   │   ├── sushi/
│   │   └── altro/
│   └── cattivo/
│       ├── pizza/
│       │   ├── bruciato/
│       │   │   ├── img001.jpg
│       │   │   └── ...
│       │   ├── crudo/
│       │   └── altro/
│       ├── carne/
│       │   ├── marcio/
│       │   ├── ammuffito/
│       │   └── ...
│       └── ...
└── modificate/
    ├── pizza/
    │   ├── bruciato/
    │   │   ├── gpt_image_1_5/
    │   │   │   ├── img001.jpg
    │   │   │   └── ...
    │   │   └── gpt_image_1_mini/
    │   │       └── ...
    │   └── crudo/
    │       └── ...
    ├── carne/
    └── ...
```

## Mapping Label

- **Label 0 (NON_FRODE)**: Tutte le immagini sotto `originali/`
- **Label 1 (FRODE)**: Tutte le immagini sotto `modificate/`

## Metadati Estratti

Per ogni immagine vengono estratti i seguenti metadati dal path:

| Campo | Descrizione | Esempio | Quando presente |
|-------|-------------|---------|-----------------|
| `path` | Path completo dell'immagine | `/path/to/img.jpg` | Sempre |
| `label` | 0=originale, 1=modificata | 0 o 1 | Sempre |
| `source` | Origine dell'immagine | "originali" o "modificate" | Sempre |
| `quality` | Qualità dell'originale | "buono" o "cattivo" | Solo se source=originali |
| `food_category` | Categoria di cibo | "pizza", "carne", "sushi", etc. | Sempre |
| `defect_type` | Tipo di difetto | "bruciato", "crudo", "marcio", etc. | Se presente nel path |
| `generator` | Modello generatore | "gpt_image_1_5", "gpt_image_1_mini" | Solo se source=modificate |

## Esempi di Parsing

### Esempio 1: Originale Buono
```
Path: augmented_v6/originali/buono/pizza/img001.jpg

Metadati:
- label: 0
- source: "originali"
- quality: "buono"
- food_category: "pizza"
- defect_type: None
- generator: None
```

### Esempio 2: Originale Cattivo
```
Path: augmented_v6/originali/cattivo/carne/marcio/img002.jpg

Metadati:
- label: 0
- source: "originali"
- quality: "cattivo"
- food_category: "carne"
- defect_type: "marcio"
- generator: None
```

### Esempio 3: Modificata
```
Path: augmented_v6/modificate/pizza/bruciato/gpt_image_1_5/img003.jpg

Metadati:
- label: 1
- source: "modificate"
- quality: None
- food_category: "pizza"
- defect_type: "bruciato"
- generator: "gpt_image_1_5"
```

## Output Files

Dopo il training/evaluation, vengono generati i seguenti file CSV:

### 1. predictions.csv
Contiene tutte le predizioni con metadati:

| Colonna | Descrizione |
|---------|-------------|
| path | Path dell'immagine |
| y_true | Label vera (0 o 1) |
| y_prob | Probabilità predetta (0.0-1.0) |
| y_pred | Label predetta (0 o 1) |
| source | "originali" o "modificate" |
| quality | "buono" o "cattivo" (solo originali) |
| food_category | Categoria di cibo |
| defect_type | Tipo di difetto |
| generator | Modello generatore (solo modificate) |

### 2. Group Metrics

#### group_metrics_food.csv
Metriche aggregate per categoria di cibo:

| Colonna | Descrizione |
|---------|-------------|
| food_category | Nome categoria |
| n_samples | Numero di campioni |
| n_pos | Numero di positivi (label=1) |
| n_neg | Numero di negativi (label=0) |
| accuracy | Accuratezza |
| precision | Precisione |
| recall | Recall |
| f1 | F1-score |
| roc_auc | ROC-AUC |
| pr_auc | PR-AUC |
| tp, fp, tn, fn | Confusion matrix |

#### group_metrics_defect.csv
Metriche per tipo di difetto (es: bruciato, crudo, marcio)

#### group_metrics_generator.csv
Metriche per generatore (es: gpt_image_1_5, gpt_image_1_mini)

#### group_metrics_quality.csv
Metriche per qualità originali (buono vs cattivo)

### 3. Error Analysis

#### top_false_positives.csv
Top 50 falsi positivi (predetto FRODE ma era ORIGINALE), ordinati per confidenza decrescente.

#### top_false_negatives.csv
Top 50 falsi negativi (predetto ORIGINALE ma era FRODE), ordinati per confidenza crescente.

## Analisi degli Errori

### Esempio: Analizzare errori per categoria

```python
import pandas as pd

# Carica predictions
df = pd.read_csv("outputs/runs/<run_name>/predictions.csv")

# Filtra errori
errors = df[df['y_true'] != df['y_pred']]

# Conta errori per categoria
error_counts = errors.groupby('food_category').size().sort_values(ascending=False)
print(error_counts)

# Analizza falsi positivi per generatore
fp = df[(df['y_pred'] == 1) & (df['y_true'] == 0)]
fp_by_gen = fp.groupby('generator').size()
print(fp_by_gen)
```

### Esempio: Confrontare performance tra generatori

```python
# Carica group metrics
gen_metrics = pd.read_csv("outputs/runs/<run_name>/group_metrics_generator.csv")

# Ordina per F1
gen_metrics_sorted = gen_metrics.sort_values('f1', ascending=False)
print(gen_metrics_sorted[['generator', 'f1', 'precision', 'recall']])
```

### Esempio: Identificare categorie problematiche

```python
# Carica group metrics per food
food_metrics = pd.read_csv("outputs/runs/<run_name>/group_metrics_food.csv")

# Trova categorie con basso F1
low_f1 = food_metrics[food_metrics['f1'] < 0.8].sort_values('f1')
print(low_f1[['food_category', 'n_samples', 'f1', 'precision', 'recall']])
```

## Stratificazione Split

Il dataset viene splittato in train/val/test (70/15/15) con stratificazione su:
1. **Label** (0 vs 1)
2. **Food category** (pizza, carne, sushi, etc.)

Questo garantisce che ogni split abbia una distribuzione simile di:
- Originali vs modificate
- Categorie di cibo

## Vantaggi del Nuovo Formato

1. **Tracciabilità completa**: Ogni errore può essere analizzato in dettaglio
2. **Debugging mirato**: Identificare quali categorie/generatori causano problemi
3. **Analisi granulare**: Metriche per ogni sottogruppo
4. **Miglioramento iterativo**: Capire dove concentrare gli sforzi

## Migrazione dal Vecchio Dataset

Se hai un dataset con struttura `real/fake`, puoi:

1. Mantenere il vecchio codice (ancora supportato per backward compatibility)
2. Migrare al nuovo formato per avere analisi più dettagliate

Il nuovo sistema è progettato per coesistere con il vecchio approccio.

## Note Importanti

- Il training resta **binario** (0 vs 1)
- I metadati sono usati **solo per analisi**, non per training
- Tutti i CSV sono **leggeri** e possono essere versionati su GitHub
- I checkpoint restano su Google Drive
