# Google Colab Workflow Guide

## Overview

Questa guida spiega come usare i notebook Colab per la pipeline di training completa.

## 📚 Notebook Disponibili

### 1. `Verifoto_Training_V2.ipynb` - Training Completo
Notebook principale per training da zero.

**Quando usarlo:**
- Nuovo esperimento
- Training completo con nuovi parametri
- Primo run su un dataset

**Workflow:**
1. Configura esperimento (1 cella)
2. Setup automatico
3. Training
4. Analisi risultati inline
5. Backup e push su GitHub

### 2. `Verifoto_Recovery.ipynb` - Recovery Sessione
Notebook per rigenerare risultati quando la sessione Colab scade.

**Quando usarlo:**
- Sessione Colab scaduta
- Checkpoint salvato ma risultati persi
- Vuoi rigenerare metriche con threshold diverso

**Workflow:**
1. Configura recovery
2. Carica checkpoint da Drive
3. Rigenera metriche
4. Push su GitHub

---

## 🎯 Training Workflow (Verifoto_Training_V2.ipynb)

### Step 1: Configurazione Esperimento

**Cella 1 - UNICA CELLA DA MODIFICARE:**

```python
EXPERIMENT_NAME = "2026-02-17_baseline_test"  # Nome univoco esperimento
DATASET_NAME = "exp_3_augmented_v6.2_noK"     # Nome dataset in Drive
CONFIG_FILE = "baseline.yaml"                  # Config da usare
GITHUB_TOKEN = ""                              # Token GitHub (opzionale)
```

**Regole:**
- `EXPERIMENT_NAME`: deve essere univoco per ogni run
- `DATASET_NAME`: deve corrispondere alla cartella in Drive
- `CONFIG_FILE`: uno dei file in `configs/`
- `GITHUB_TOKEN`: lascia vuoto per inserirlo quando richiesto

### Step 2: Esecuzione

Esegui tutte le celle in ordine (Runtime → Run all).

Il notebook:
1. ✓ Verifica GPU disponibile
2. ✓ Clona repo e installa dipendenze
3. ✓ Monta Google Drive
4. ✓ Verifica dataset esiste
5. ✓ Aggiorna config con path corretto
6. ✓ Esegue training
7. ✓ Genera analisi dettagliata
8. ✓ Mostra grafici inline
9. ✓ Backup su Drive
10. ✓ Push risultati su GitHub

### Step 3: Risultati

Dopo il training, vedrai inline:
- Metriche test (accuracy, precision, recall, F1, AUC)
- Confusion matrix
- Grafici (ROC, PR curve, distribuzioni)
- Performance per categoria food
- Confronto generatori
- Top false positives

### Step 4: Sync con GitHub

Il notebook pusha automaticamente i risultati su GitHub.

**Opzioni:**
1. Token nella cella config → push automatico
2. Token vuoto → richiesto durante esecuzione
3. Skip token → solo backup su Drive

---

## 🔄 Recovery Workflow (Verifoto_Recovery.ipynb)

### Quando Usare

La sessione Colab scade dopo 12 ore di inattività. Se hai:
- ✓ Checkpoint salvato su Drive
- ✗ Risultati persi (metriche, grafici)

Usa il recovery notebook per rigenerare tutto.

### Step 1: Configurazione

```python
ORIGINAL_RUN_NAME = "2026-02-16_noK"           # Nome run originale
RECOVERY_RUN_NAME = "2026-02-16_noK_recovered" # Nome per recovery
DATASET_NAME = "exp_3_augmented_v6.2_noK"      # Dataset usato
CONFIG_FILE = "baseline.yaml"                   # Config usata
THRESHOLD = 0.5                                 # Threshold classificazione
```

### Step 2: Esecuzione

Esegui tutte le celle. Il notebook:
1. Carica checkpoint da Drive
2. Rigenera tutte le metriche
3. Crea tutti i grafici
4. Salva risultati
5. Push su GitHub

**Nota:** Il recovery è molto più veloce del training (pochi minuti vs ore).

---

## 📁 Struttura File

### Su Google Drive

```
/content/drive/MyDrive/
├── DatasetVerifoto/
│   └── images/
│       ├── exp_3_augmented_v6.1_categorized/
│       ├── exp_3_augmented_v6.2_noK/
│       └── ...
├── verifoto_checkpoints/
│   ├── 2026-02-16_noK/
│   │   └── best.pt
│   └── ...
└── verifoto_results/          # Backup risultati
    ├── 2026-02-16_noK/
    └── ...
```

### Su GitHub (dopo push)

```
outputs/runs/
├── 2026-02-16_noK/
│   ├── metrics.json
│   ├── predictions.csv
│   ├── group_metrics_food.csv
│   ├── group_metrics_defect.csv
│   ├── group_metrics_generator.csv
│   ├── group_metrics_quality.csv
│   ├── top_false_positives.csv
│   ├── top_false_negatives.csv
│   ├── cm.png
│   ├── prob_dist.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── notes.md
└── ...
```

---

## 🔑 GitHub Token Setup

### Creare un Personal Access Token (PAT)

1. Vai su GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token (classic)
3. Seleziona scope: `repo` (full control)
4. Copia il token (lo vedrai solo una volta!)

### Usare il Token

**Opzione 1: Nella cella config**
```python
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"
```

**Opzione 2: Quando richiesto**
Lascia vuoto nella config, il notebook lo chiederà durante l'esecuzione.

**Sicurezza:**
- Non committare il token nel notebook
- Usa token con scope minimo necessario
- Rigenera periodicamente

---

## 🎨 Best Practices

### Naming Convention

```
EXPERIMENT_NAME = "YYYY-MM-DD_description"
```

Esempi:
- `2026-02-17_baseline_test`
- `2026-02-17_convnext_augmented`
- `2026-02-17_efficientnet_noK`

### Organizzazione Esperimenti

1. **Baseline**: primo run con config standard
2. **Varianti**: modifica 1 parametro alla volta
3. **Naming**: descrivi cosa cambia rispetto al baseline

### Workflow Consigliato

1. **Locale**: modifica codice, test, commit
2. **Colab**: pull, training, push risultati
3. **Locale**: pull risultati, analisi approfondita

---

## 🐛 Troubleshooting

### "Dataset not found"
- Verifica `DATASET_NAME` corrisponda alla cartella in Drive
- Controlla path completo: `/content/drive/MyDrive/DatasetVerifoto/images/{DATASET_NAME}`

### "Checkpoint not found" (Recovery)
- Verifica `ORIGINAL_RUN_NAME` sia corretto
- Controlla checkpoint esista: `/content/drive/MyDrive/verifoto_checkpoints/{ORIGINAL_RUN_NAME}/best.pt`

### "Out of memory"
- Riduci `batch_size` nel config
- Usa modello più piccolo (es. `efficientnet_b0` invece di `b3`)

### "Session expired"
- Usa `Verifoto_Recovery.ipynb` per rigenerare risultati
- Checkpoint è salvato su Drive, non perdi il training

### "Git push failed"
- Verifica token GitHub sia valido
- Controlla scope token includa `repo`
- Prova a rigenerare il token

---

## 📊 Analisi Locale

Dopo il push su GitHub, sul tuo computer locale:

```bash
# Pull risultati
git pull

# Analisi dettagliata
python scripts/analyze_results.py 2026-02-17_baseline_test

# Confronto tra run
python scripts/compare_runs.py 2026-02-16_noK 2026-02-17_baseline_test
```

---

## 🚀 Quick Start

### Nuovo Training

1. Apri `Verifoto_Training_V2.ipynb` su Colab
2. Modifica solo la prima cella:
   ```python
   EXPERIMENT_NAME = "2026-02-17_my_experiment"
   DATASET_NAME = "exp_3_augmented_v6.2_noK"
   CONFIG_FILE = "baseline.yaml"
   ```
3. Runtime → Run all
4. Aspetta (1-2 ore per training completo)
5. Risultati automaticamente su GitHub

### Recovery Sessione

1. Apri `Verifoto_Recovery.ipynb` su Colab
2. Configura:
   ```python
   ORIGINAL_RUN_NAME = "2026-02-16_noK"
   RECOVERY_RUN_NAME = "2026-02-16_noK_recovered"
   ```
3. Runtime → Run all
4. Risultati rigenerati in pochi minuti

---

## 📝 Note Finali

- **Checkpoint**: sempre salvati su Drive, non si perdono
- **Risultati**: backup su Drive + push GitHub
- **Sessioni**: scadono dopo 12h inattività, usa recovery
- **GPU**: T4 gratuita, sufficiente per i nostri modelli
- **Costi**: tutto gratuito con Colab free tier

Per domande o problemi, consulta la documentazione completa in `docs/`.
