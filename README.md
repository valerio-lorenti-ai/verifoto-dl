# Verifoto Deep Learning

Training pipeline per rilevamento frodi fotografiche nel food delivery. Parte della piattaforma Verifoto-AI SaaS che aiuta i ristoranti a identificare foto manipolate usate per ottenere rimborsi non dovuti.

## Cos'è Verifoto

Verifoto-AI è una web app per ristoranti che lavorano con il food delivery, per difendersi dalle frodi fotografiche. Alcuni clienti manipolano digitalmente le immagini del cibo (aggiungendo muffe false, bruciature artificiali, o difetti generati con AI) per ottenere rimborsi ingiustificati.

**Principio chiave**: Approccio conservativo - ottimizzare per precision piuttosto che recall. Meglio perdere qualche frode che accusare falsamente reclami legittimi.

## Struttura Progetto

```
verifoto-dl/
├── src/                  # Codice training/evaluation
├── configs/              # Configurazioni YAML
├── scripts/              # Script helper + notebook Colab
├── outputs/runs/         # Risultati training (versionati)
├── checkpoints/          # Pesi modello (NON versionati, su Drive)
├── docs/                 # Documentazione tecnica di riferimento
└── .kiro/                # Contesto AI assistant (steering + stato)
```

## Quick Start

### Setup Iniziale

```bash
# Clone
git clone https://github.com/<YOUR_USERNAME>/verifoto-dl.git
cd verifoto-dl

# Test locale (opzionale)
pip install -r requirements.txt
python scripts/quick_test.py
```

### Workflow Tipico

```
1. Modifica codice localmente
2. git commit + push
3. Apri Colab → git pull
4. Esegui training (salva su Drive)
5. Commit risultati da Colab
6. git pull localmente
7. Analizza risultati
```

### Training su Colab

```python
# Setup
!git clone https://github.com/<USER>/verifoto-dl.git && cd verifoto-dl
!pip install -q -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

# Training
!python -m src.train \
    --config configs/baseline.yaml \
    --run_name "2026-02-17_baseline" \
    --checkpoint_dir "/content/drive/MyDrive/verifoto_checkpoints"
```

**Importante**: Aggiorna `dataset_root` in `configs/baseline.yaml` con il tuo path su Drive.

### Analisi Risultati

```bash
# Pull risultati
git pull

# Confronta tutti i run
python scripts/compare_runs.py

# Analizza run specifico
python scripts/analyze_results.py <run_name>

# Visualizza metriche
cat outputs/runs/<run_name>/metrics.json
```

## Comandi Principali

### Training

```bash
python -m src.train \
    --config configs/baseline.yaml \
    --run_name "2026-02-17_exp1" \
    --checkpoint_dir "/content/drive/MyDrive/verifoto_checkpoints"
```

### Evaluation

```bash
python -m src.eval \
    --config configs/baseline.yaml \
    --run_name "2026-02-17_eval" \
    --checkpoint_path "/path/to/checkpoint.pt" \
    --threshold 0.5
```

### Quick Test (debug veloce)

```bash
python -m src.train \
    --config configs/quick_test.yaml \
    --run_name "debug" \
    --checkpoint_dir "checkpoints"
```

## Output di Ogni Run

```
outputs/runs/<run_name>/
├── metrics.json                      # Metriche complete + config
├── notes.md                          # Riepilogo human-readable
├── predictions.csv                   # Tutte le predizioni con metadati
├── group_metrics_food.csv            # Metriche per categoria cibo
├── group_metrics_defect.csv          # Metriche per tipo difetto
├── group_metrics_generator.csv       # Metriche per generatore
├── group_metrics_quality.csv         # Metriche per qualità
├── top_false_positives.csv           # Top 50 errori FP
├── top_false_negatives.csv           # Top 50 errori FN
├── cm.png                            # Confusion matrix
├── roc_curve.png                     # ROC curve
├── pr_curve.png                      # Precision-Recall curve
└── prob_dist.png                     # Distribuzione probabilità
```

## Configurazione

Modifica `configs/baseline.yaml` per cambiare:

```yaml
model_name: "efficientnet_b3"    # o "convnext_tiny", "resnet50"
batch_size: 16
epochs_head: 5
epochs_finetune: 30
lr_head: 0.001
lr_finetune: 0.0001
dataset_root: "/path/to/your/dataset"
```

## Dataset augmented_v6

Il progetto supporta il dataset gerarchico `augmented_v6` con metadati dettagliati:

```
augmented_v6/
├── originali/
│   ├── buono/<food>/<defect>/
│   └── cattivo/<food>/<defect>/
└── modificate/<food>/<defect>/<generator>/
```

Questo permette analisi approfondite degli errori per categoria, tipo difetto, e generatore.

## Features Chiave

- **Riproducibile**: Seed fissi, tracking commit git
- **Efficiente**: Checkpoint su Drive, solo risultati leggeri su GitHub
- **Iterazione veloce**: Pull modifiche codice in Colab, no copia manuale file
- **Output strutturato**: JSON per parsing facile
- **Split group-aware**: Previene data leakage da near-duplicates
- **Tracking metadati**: Analizza errori per categoria, difetto, generatore
- **Approccio conservativo**: Ottimizzato per alta precision (bassi falsi positivi)

## Contesto Produzione

Questo modello sarà deployato nella web app Verifoto-AI dove:
- I ristoranti caricano immagini sospette
- Il modello fornisce confidence score + indicatori tecnici
- Un LLM genera spiegazione human-readable
- Il report è usato per decisioni interne
- **I falsi positivi danneggiano la fiducia** → precision è critica

## Tips

1. **Esperimenti multipli**: Crea `configs/experiment2.yaml` per setup diversi
2. **Tuning threshold**: Usa `eval.py` con diversi valori `--threshold`
3. **Test veloci**: Riduci `epochs_head` e `epochs_finetune` nel config per debug
4. **Organizzazione Drive**: Mantieni checkpoint organizzati per data/nome esperimento

## Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| "Dataset not found" | Aggiorna `dataset_root` nel config YAML |
| "Out of memory" | Riduci `batch_size` nel config |
| "No GPU" | Runtime → Change runtime type → GPU |
| "Module not found" | `!pip install -r requirements.txt` |

## Documentazione Tecnica

Per dettagli tecnici approfonditi, vedi:
- `docs/WORKFLOW.md` - Workflow dettagliato
- `docs/AUGMENTED_V6_DATASET.md` - Formato dataset
- `.kiro/steering/` - Contesto per AI assistant

## Requirements

- Python 3.8+
- PyTorch 2.0+
- GPU CUDA (per training)
- Google Drive (per storage checkpoint)
