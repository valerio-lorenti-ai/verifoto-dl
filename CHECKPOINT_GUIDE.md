# Guida Completa: Checkpoint e Pipeline End-to-End

## A) Dove Vengono Salvati i Checkpoint

### Durante Training (`src/train_v7.py`)

**Percorso completo:**
```
checkpoints/{run_name}/best.pt
```

**Esempio su Colab con Drive:**
```
/content/drive/MyDrive/checkpoints/2026-02-20_convnext_v8/best.pt
```

**Nome file:** `best.pt` (formato PyTorch)

**Quando viene salvato:**
- Solo il BEST checkpoint (non ogni epoch)
- Salvato quando la metrica monitorata migliora
- Metrica di default: `pr_auc` (configurabile in config.yaml con `monitor`)

**Codice di salvataggio** (`src/train_v7.py`, linea 101-108):
```python
def save_checkpoint(model, path: Path, best_metric: float = None, cfg: dict = None):
    payload = {
        "state_dict": model.state_dict(),
        "best_metric": float(best_metric) if best_metric is not None else None,
        "cfg": cfg if cfg is not None else None,
    }
    torch.save(payload, str(path))
```

**Contenuto del checkpoint:**
- `state_dict`: pesi del modello
- `best_metric`: valore della metrica migliore (es: pr_auc=0.95)
- `cfg`: configurazione completa del training

### Checkpoint Migliore da Usare

**File:** `checkpoints/{run_name}/best.pt`

**Metrica di selezione:** 
- Default: `pr_auc` (Area Under Precision-Recall Curve)
- Alternativa: `f1`, `precision_at_recall_90`
- Configurabile in `config.yaml` con parametro `monitor`

**Esempio per V8:**
```yaml
monitor: "pr_auc"  # Metrica usata per scegliere best checkpoint
```

---

## B) Cartella outputs/runs/

### Struttura Completa

```
outputs/runs/{run_name}/
├── predictions.csv              # Predizioni su test set
├── metrics.json                 # Metriche aggregate
├── notes.md                     # Summary leggibile
├── cm.png                       # Confusion matrix
├── prob_dist.png                # Distribuzione probabilità
├── roc_curve.png                # ROC curve
├── pr_curve.png                 # Precision-Recall curve
├── group_metrics_food.csv       # Metriche per categoria cibo
├── group_metrics_defect.csv     # Metriche per tipo difetto
├── group_metrics_generator.csv  # Metriche per generatore
├── group_metrics_quality.csv    # Metriche per qualità
├── top_false_positives.csv      # Top 50 FP
├── top_false_negatives.csv      # Top 50 FN
├── chosen_threshold.json        # Threshold ottimizzato
├── photo_metrics.csv            # Metriche per photo_id
└── photo_hard_fp.csv            # Photo con hard FP (per fine-tuning)
```

**NON contiene pesi del modello** - quelli sono in `checkpoints/`

---

## C) Come Vengono Caricati i Checkpoint

### Durante Evaluation (`src/eval.py`)

**Parametro richiesto:**
```bash
python src/eval.py \
  --config configs/convnext_v8.yaml \
  --run_name test_run \
  --checkpoint_path checkpoints/2026-02-20_convnext_v8/best.pt \
  --threshold 0.55
```

**Codice di caricamento** (`src/eval.py`, linea 115-120):
```python
model = build_model(model_name, pretrained=False, drop_rate=drop_rate).to(device)
ckpt = torch.load(args.checkpoint_path, map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

### Durante Fine-Tuning (`scripts/hard_negative_finetune.py`)

**Parametro richiesto:**
```bash
python scripts/hard_negative_finetune.py \
  --run outputs/runs/2026-02-20_convnext_v8 \
  --config configs/convnext_v8.yaml \
  --checkpoint_dir /content/drive/MyDrive/checkpoints \
  --epochs 5 \
  --lr 1e-5
```

**Logica di caricamento** (`scripts/hard_negative_finetune.py`, linea 175-185):
```python
# Se --checkpoint_dir è fornito (Colab/Drive)
if args.checkpoint_dir:
    checkpoint_path = Path(args.checkpoint_dir) / run_dir.name / "best.pt"
else:
    # Default: cerca in checkpoints/ locale
    checkpoint_path = run_dir.parent.parent / "checkpoints" / run_dir.name / "best.pt"

model = build_model(model_name, pretrained=False, drop_rate=drop_rate).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["state_dict"])
```

**Nuovo run separato:**
- Fine-tuning crea un NUOVO run: `{run_name}_hard_negative`
- Salva nuovi checkpoint in: `checkpoints/{run_name}_hard_negative/best.pt`
- Salva risultati in: `outputs/runs/{run_name}_hard_negative/`

**Layer trainabili:**
- TUTTI i layer sono trainabili (no freeze)
- Learning rate molto basso: `1e-5` (vs `1e-4` del training normale)
- Questo permette aggiustamenti fini senza dimenticare

---

## D) Dati Usati per Fine-Tuning

### Quali Immagini

**Dataset completo:**
- Usa lo STESSO dataset del training originale
- Split identico (stesso seed=42)
- Include: originali + GPT-1 + GPT-1.5 + altri generatori

**Hard Negatives:**
- Definiti come: immagini REALI (label=0) che il modello classifica come FAKE con alta confidence
- Identificati tramite `scripts/analyze_by_photo.py` → genera `photo_hard_fp.csv`
- Criterio: photo_id con FP rate > soglia (es: >50% delle versioni sono FP)

### Sampling Strategy

**Oversampling degli hard negatives:**
```python
# scripts/hard_negative_finetune.py, linea 44-75
def create_hard_negative_sampler(dataset, hard_fp_ids: set, repeat_factor: float = 3.0):
    """
    Crea sampler che oversampla hard negatives di repeat_factor volte
    """
    weights = []
    for idx in range(len(dataset)):
        path = dataset.df.iloc[idx]['path']
        photo_id = extract_photo_id(path)
        label = dataset.df.iloc[idx]['label']
        
        # Hard negative (real image che viene classificata come fake)
        if label == 0 and photo_id in hard_fp_ids:
            weights.append(repeat_factor)  # 3x più probabilità
        else:
            weights.append(1.0)
    
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

**Proporzione effettiva in un batch:**
- Se hai 100 hard negatives su 1000 immagini totali
- Con `repeat_factor=3.0`, gli hard negatives appaiono ~23% delle volte
- Resto: mix normale di originali + fake

**Loss function:**
- Usa la STESSA loss del training originale
- Se config ha `weighted_focal`, usa quella
- Mantiene `pos_weight` per bilanciare classi

---

## E) Meccanismi Anti-Dimenticanza

### Strategie Implementate

1. **Learning Rate Molto Basso**
   - Training: `lr_finetune = 1e-4`
   - Fine-tuning: `lr = 1e-5` (10x più basso)
   - Permette aggiustamenti fini senza stravolgere pesi

2. **Replay di Esempi Vecchi**
   - Il dataset include TUTTE le immagini originali
   - Non solo hard negatives, ma anche:
     - Immagini reali "facili" (correttamente classificate)
     - Tutte le immagini fake (GPT-1, GPT-1.5, altri)
   - Oversampling aumenta solo la frequenza degli hard negatives

3. **Poche Epochs**
   - Default: 5 epochs (vs 35 del training)
   - Riduce rischio di overfitting sugli hard negatives

4. **No Freeze**
   - Tutti i layer sono trainabili
   - Ma con lr basso, i cambiamenti sono graduali

### Metriche Guardrail

**Durante fine-tuning, monitora:**
```python
val_m = validate(model, val_loader, threshold=0.5, device=device)
print(f"val_f1={val_m['f1']:.4f} val_prec={val_m['prec']:.4f} val_rec={val_m['rec']:.4f}")
```

**Dopo fine-tuning, confronta con run originale:**
```bash
python scripts/compare_runs.py \
  --run1 outputs/runs/2026-02-20_convnext_v8 \
  --run2 outputs/runs/2026-02-20_convnext_v8_hard_negative
```

**Segnali di allarme:**
- Precision scende >5% → troppi nuovi FP
- Recall su GPT-1 scende >5% → sta dimenticando
- F1 complessivo scende → fine-tuning non sta aiutando

### Strategie di Correzione

**Se precision scende:**
1. Riduci `repeat_factor` (da 3.0 a 2.0)
2. Aggiungi più hard negatives ORIGINALI al dataset
3. Riduci epochs (da 5 a 3)
4. Aumenta `real_weight` nella loss function

**Se recall scende:**
1. Aumenta `repeat_factor` (da 3.0 a 4.0)
2. Aggiungi più hard positives (fake classificati come real)
3. Riduci learning rate (da 1e-5 a 5e-6)

---

## F) Pipeline End-to-End

### 1. Training Iniziale

**Input:**
```
dataset: /content/drive/MyDrive/DatasetVerifoto/images/exp_3_augmented_v6.1_categorized/
config: configs/convnext_v8.yaml
```

**Comando:**
```bash
python src/train_v7.py \
  --config configs/convnext_v8.yaml \
  --run_name 2026-02-20_convnext_v8 \
  --checkpoint_dir /content/drive/MyDrive/checkpoints
```

**Output:**
```
checkpoints/2026-02-20_convnext_v8/best.pt  # Pesi del modello (metrica: pr_auc)
```

### 2. Evaluation

**Input:**
```
checkpoint: checkpoints/2026-02-20_convnext_v8/best.pt
config: configs/convnext_v8.yaml
threshold: 0.55  # Da config o da ottimizzazione
```

**Comando:**
```bash
python src/eval.py \
  --config configs/convnext_v8.yaml \
  --run_name 2026-02-20_convnext_v8 \
  --checkpoint_path checkpoints/2026-02-20_convnext_v8/best.pt \
  --threshold 0.55
```

**Output:**
```
outputs/runs/2026-02-20_convnext_v8/
├── predictions.csv
├── metrics.json
├── group_metrics_*.csv
├── top_false_positives.csv
├── top_false_negatives.csv
└── visualizations (cm.png, roc_curve.png, etc.)
```

### 3. Photo-Level Analysis

**Input:**
```
run: outputs/runs/2026-02-20_convnext_v8
```

**Comando:**
```bash
python scripts/analyze_by_photo.py \
  --run outputs/runs/2026-02-20_convnext_v8 \
  --min-recall 0.90
```

**Output:**
```
outputs/runs/2026-02-20_convnext_v8/
├── photo_metrics.csv        # Metriche per photo_id
├── photo_hard_fp.csv        # Photo con hard FP (per fine-tuning)
└── chosen_threshold.json    # Threshold ottimizzato
```

### 4. Fine-Tuning con Hard Negatives

**Input:**
```
checkpoint: checkpoints/2026-02-20_convnext_v8/best.pt
hard_fp_list: outputs/runs/2026-02-20_convnext_v8/photo_hard_fp.csv
config: configs/convnext_v8.yaml
```

**Comando:**
```bash
python scripts/hard_negative_finetune.py \
  --run outputs/runs/2026-02-20_convnext_v8 \
  --config configs/convnext_v8.yaml \
  --checkpoint_dir /content/drive/MyDrive/checkpoints \
  --epochs 5 \
  --lr 1e-5 \
  --repeat_factor 3.0
```

**Output:**
```
checkpoints/2026-02-20_convnext_v8_hard_negative/best.pt  # Nuovi pesi
outputs/runs/2026-02-20_convnext_v8_hard_negative/
├── predictions.csv
├── metrics.json
└── ...
```

### 5. Confronto Runs

**Input:**
```
run1: outputs/runs/2026-02-20_convnext_v8
run2: outputs/runs/2026-02-20_convnext_v8_hard_negative
```

**Comando:**
```bash
python scripts/compare_runs.py \
  --run1 outputs/runs/2026-02-20_convnext_v8 \
  --run2 outputs/runs/2026-02-20_convnext_v8_hard_negative
```

**Output:**
```
Confronto metriche, breakdown per categoria, miglioramenti/peggioramenti
```

### 6. Deployment/Inferenza

**Checkpoint finale:**
```
checkpoints/2026-02-20_convnext_v8_hard_negative/best.pt
```

**Threshold finale:**
```
0.55  # Da chosen_threshold.json del run finale
```

**Codice di inferenza:**
```python
import torch
from src.utils.model import build_model

# Load model
model = build_model("convnext_tiny", pretrained=False, drop_rate=0.3)
ckpt = torch.load("checkpoints/2026-02-20_convnext_v8_hard_negative/best.pt")
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Predict
with torch.no_grad():
    logits = model(image_tensor)
    prob = torch.sigmoid(logits).item()
    prediction = "FAKE" if prob >= 0.55 else "REAL"
```

---

## G) Domande Frequenti

### Devo ricalcolare threshold dopo fine-tuning?

**Sì, ma dipende:**

1. **Approccio conservativo (raccomandato):**
   - Usa lo STESSO threshold del run originale (0.55)
   - Valuta se le metriche sono ancora accettabili
   - Se precision/recall cambiano troppo, ri-ottimizza

2. **Approccio ottimale:**
   - Ri-esegui `analyze_by_photo.py` sul nuovo run
   - Trova nuovo threshold ottimale
   - Confronta con threshold originale

**Codice attuale:**
```python
# scripts/hard_negative_finetune.py, linea 245-250
threshold_file = run_dir / "chosen_threshold.json"
if threshold_file.exists():
    with open(threshold_file, 'r') as f:
        threshold_data = json.load(f)
        test_threshold = threshold_data.get('recommendation', 0.5)
else:
    test_threshold = 0.5
```

### Posso fare fine-tuning su fine-tuning?

**Sì, ma con cautela:**

```bash
# Fine-tuning iterativo
python scripts/hard_negative_finetune.py \
  --run outputs/runs/2026-02-20_convnext_v8_hard_negative \
  --config configs/convnext_v8.yaml \
  --epochs 3 \
  --lr 5e-6  # Ancora più basso
```

**Rischi:**
- Overfitting progressivo
- Dimenticanza cumulativa
- Instabilità

**Raccomandazione:**
- Massimo 2 iterazioni
- Sempre confronta con baseline originale
- Se peggiora, torna al checkpoint precedente

### Come scelgo repeat_factor?

**Linee guida:**

- `repeat_factor=1.0`: nessun oversampling (baseline)
- `repeat_factor=2.0`: hard negatives appaiono 2x più spesso (conservativo)
- `repeat_factor=3.0`: hard negatives appaiono 3x più spesso (default, bilanciato)
- `repeat_factor=5.0`: hard negatives appaiono 5x più spesso (aggressivo)

**Scegli in base a:**
- Quanti hard negatives hai (pochi → repeat_factor alto)
- Quanto vuoi ridurre FP (molto → repeat_factor alto)
- Rischio di dimenticanza (alto → repeat_factor basso)

---

## H) Checklist Pre-Deployment

Prima di usare un checkpoint in produzione:

- [ ] Checkpoint esiste e si carica correttamente
- [ ] Threshold è stato ottimizzato (non usare 0.5 di default)
- [ ] Metriche su test set sono accettabili:
  - [ ] Precision ≥ 75%
  - [ ] Recall ≥ 85%
  - [ ] F1 ≥ 80%
- [ ] Breakdown per categoria non ha outlier:
  - [ ] Nessuna categoria con recall < 70%
  - [ ] Nessuna categoria con precision < 60%
- [ ] Confronto con baseline mostra miglioramento
- [ ] Test su dataset esterno (se disponibile)
- [ ] Documentazione completa (config, threshold, metriche)

---

## I) File di Riferimento

**Training:**
- `src/train_v7.py` - Training principale
- `src/utils/model.py` - Architetture modelli
- `src/utils/losses.py` - Loss functions
- `src/utils/data.py` - Dataset e split

**Evaluation:**
- `src/eval.py` - Evaluation su test set
- `scripts/analyze_by_photo.py` - Photo-level analysis
- `scripts/compare_runs.py` - Confronto tra runs

**Fine-Tuning:**
- `scripts/hard_negative_finetune.py` - Fine-tuning con hard negatives

**Config:**
- `configs/convnext_v8.yaml` - Configurazione V8 (raccomandato)
