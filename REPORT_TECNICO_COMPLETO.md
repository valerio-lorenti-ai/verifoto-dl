# REPORT TECNICO COMPLETO - Verifoto Deep Learning

**Data**: 24 Febbraio 2026  
**Progetto**: Detection manipolazioni fotografiche per food delivery  
**Modello**: ConvNeXt Tiny (V8)

---

## 1. STRUTTURA REPOSITORY

```
verifoto-dl/
├── src/                          # Codice training/evaluation
│   ├── train_v7.py              # Script training principale
│   ├── eval.py                  # Script evaluation
│   └── utils/
│       ├── data.py              # Dataset loader + split strategies
│       ├── model.py             # Model builder (timm)
│       ├── augmentations.py     # Augmentation avanzate
│       ├── losses.py            # Loss functions (BCE, Focal, Weighted Focal)
│       ├── metrics.py           # Metriche + threshold optimization
│       └── visualization.py     # Plot (ROC, PR, CM, distributions)
├── configs/                      # Configurazioni YAML
│   ├── baseline.yaml
│   ├── convnext_v8.yaml         # Config attuale (ottimizzata)
│   └── ...
├── scripts/                      # Script analisi + notebook Colab
│   ├── analyze_results.py
│   ├── compare_runs.py
│   └── notebooks/verifoto_dl.ipynb
├── outputs/runs/                 # Risultati training (versionati su git)
├── checkpoints/                  # Pesi modello (NON versionati, su Drive)
└── docs/                         # Documentazione tecnica
```

### Workflow tipico
1. Sviluppo locale → commit + push
2. Colab → git pull + training (salva checkpoint su Drive)
3. Commit risultati da Colab
4. Pull locale → analisi risultati

---

## 2. DATASET LOADER

### Struttura dataset: `augmented_v6.1_categorized`

```
augmented_v6/
├── originali/                    # Label = 0 (NON_FRODE)
│   ├── buono/<food_category>/*.jpg
│   └── cattivo/<food_category>/<defect_type>/*.jpg
└── modificate/                   # Label = 1 (FRODE)
    └── <food_category>/<defect_type>/<generator>/*.jpg
```

### Funzione: `parse_augmented_v6_dataset(root)`
- Scansiona ricorsivamente tutte le immagini (.jpg, .jpeg, .png, .webp, .bmp)
- Estrae metadati da path:
  - `label`: 0 (originali) o 1 (modificate)
  - `source`: "originali" o "modificate"
  - `quality`: "buono" o "cattivo" (solo originali)
  - `food_category`: categoria cibo (es: "pasta", "pizza", "riso_paella")
  - `defect_type`: tipo difetto (es: "bruciato", "crudo", "muffa")
  - `generator`: generatore AI (es: "gpt", "dalle", "midjourney") - solo modificate
- Ritorna DataFrame con tutte le immagini + metadati

### Photo ID extraction
- Ogni foto ha ID univoco: **primi 4 caratteri del filename**
- Esempio: `1976_q95.jpg` → photo_id = `1976`
- Tutte le versioni della stessa foto (originale, modificata, augmented) condividono lo stesso photo_id
- Usato per prevenire data leakage nello split

---

## 3. TRASFORMAZIONI / AUGMENTATIONS

### Training transforms (normale)
```python
transforms.RandomResizedCrop(224, scale=(0.80, 1.0), ratio=(0.90, 1.10))
transforms.RandomHorizontalFlip(p=0.3)
transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02)
transforms.RandomApply([GaussianBlur(kernel_size=3)], p=0.15)
RandomJPEGCompression(quality_min=55, quality_max=95, p=0.55)
RandomGaussianNoise(sigma_max=0.02, p=0.35)
transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
```

### Training transforms (strong) - per immagini reali
```python
transforms.RandomResizedCrop(224, scale=(0.70, 1.0), ratio=(0.85, 1.15))
transforms.RandomHorizontalFlip(p=0.4)
transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03)
transforms.RandomApply([GaussianBlur(kernel_size=3)], p=0.25)
RandomJPEGCompression(quality_min=50, quality_max=95, p=0.65)
RandomGaussianNoise(sigma_max=0.03, p=0.45)
```

### Evaluation transforms
```python
transforms.Resize(int(224 * 1.15))  # 257
transforms.CenterCrop(224)
transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
```

### Augmentation differenziata
- **Config**: `real_augmentation_multiplier: 1.5` + `use_differential_aug: True`
- **Logica**: Immagini reali (label=0) ricevono augmentation più aggressiva
- **Obiettivo**: Ridurre overfitting su dettagli fini delle immagini reali → meno FP

### Custom augmentations (in `augmentations.py`)
- `RandomJPEGCompression`: Simula compressione WhatsApp/social media
- `RandomResizeDownUp`: Simula screenshot + resize
- `RandomSharpening`: Simula filtri Instagram
- `RandomGaussianNoise`: Simula sensor noise
- `RandomScreenshotArtifacts`: Simula bordi, crop, UI margins
- `RandomBlur`: Simula motion blur, out-of-focus

---

## 4. ARCHITETTURA MODELLO

### Model builder: `build_model()`
```python
model = timm.create_model(
    model_name="convnext_tiny",
    pretrained=True,              # ImageNet-1K weights
    num_classes=1,                # Binary classification (logit singolo)
    drop_rate=0.3                 # Dropout rate
)
```

### ConvNeXt Tiny specs
- **Parametri**: ~28M
- **Pretrained**: ImageNet-1K (timm)
- **Input**: 224x224 RGB
- **Output**: 1 logit (sigmoid → probabilità)
- **Dropout**: 0.3 (nella head)

### Frozen layers strategy
**Phase 1: Head-only** (5 epochs)
- Backbone: FROZEN
- Head (classifier): TRAINABLE
- Learning rate: 0.0005

**Phase 2: Fine-tuning** (35 epochs)
- Backbone: TRAINABLE
- Head: TRAINABLE
- Learning rate: 0.00005 (10x più basso)

### Funzione: `set_backbone_trainable(model, trainable)`
- Congela/scongela tutti i parametri
- Garantisce che la head sia sempre trainable

---

## 5. TRAINING LOOP COMPLETO

### Script: `src/train_v7.py`

### Setup iniziale
```python
set_seed(42)                      # Riproducibilità
device = "cuda" if available else "cpu"
```

### Data loading
```python
df = parse_augmented_v6_dataset(dataset_root)

# Split strategy: domain_aware (default) o group_v6
train_df, val_df, test_df = domain_aware_group_split_v1(
    df, 0.70, 0.15, 0.15, seed=42, include_food=False
)

# Salva split per reproducibility
train_df.to_csv(output_dir / "split/train_split.csv")
val_df.to_csv(output_dir / "split/val_split.csv")
test_df.to_csv(output_dir / "split/test_split.csv")
```

### DataLoader
```python
train_loader = DataLoader(
    train_ds, 
    batch_size=12,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_with_metadata
)
```

### Training phases

**Phase 1: Head-only (5 epochs)**
```python
set_backbone_trainable(model, trainable=False)
optimizer = AdamW(params, lr=0.0005, weight_decay=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

for epoch in range(1, 6):
    train_loss = train_one_epoch(model, train_loader, optimizer, 
                                  criterion, scheduler, max_grad_norm=1.0)
    val_metrics = validate(model, val_loader, threshold=0.5)
    
    if val_metrics[monitor] > best_metric:
        save_checkpoint(model, best_ckpt_path)
    
    if early_stopping.step(val_metrics[monitor]):
        break
```

**Phase 2: Fine-tuning (35 epochs)**
```python
set_backbone_trainable(model, trainable=True)
optimizer = AdamW(model.parameters(), lr=0.00005, weight_decay=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

for epoch in range(1, 36):
    # Same as Phase 1
```

### Training step
```python
def train_one_epoch(model, loader, optimizer, criterion, scheduler, max_grad_norm):
    model.train()
    for x, y, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
```

### Validation step
```python
@torch.no_grad()
def validate(model, loader, threshold=0.5):
    probs, y_true, _ = predict_proba(model, loader, device)
    return compute_metrics_from_probs(probs, y_true, threshold)
```

### Early stopping
- **Patience**: 8 epochs
- **Monitor**: `pr_auc` (Precision-Recall AUC)
- **Mode**: maximize
- **Min delta**: 1e-4

---

## 6. LOSS FUNCTION

### Config attuale: `weighted_focal`
```python
loss_type: "weighted_focal"
focal_alpha: 0.25              # Peso classe positiva
focal_gamma: 2.0               # Focusing parameter
real_weight: 2.0               # Penalità 2x per errori su immagini reali
```

### Formula Weighted Focal Loss
```
FL(p_t) = alpha_t * (1 - p_t)^gamma * BCE(p_t) * real_weight_mask

dove:
- p_t = prob se y=1, (1-prob) se y=0
- alpha_t = focal_alpha se y=1, (1-focal_alpha) se y=0
- real_weight_mask = real_weight se y=0 (real), 1.0 se y=1 (generated)
```

### Effetti
1. **Focal Loss**: Riduce peso su easy examples (immagini generate ovvie)
2. **Class weights**: Bilancia classi sbilanciate (pos_weight calcolato automaticamente)
3. **Real weight**: Penalizza 2x errori su immagini reali → riduce FP

### Altre loss disponibili
- `bce`: BCE standard con pos_weight
- `focal`: Focal Loss senza real_weight
- `cost_sensitive`: Loss con costi asimmetrici FP/FN

---

## 7. OPTIMIZER + LR + SCHEDULER

### Optimizer: AdamW
```python
# Phase 1: Head-only
optimizer = torch.optim.AdamW(
    params=[p for p in model.parameters() if p.requires_grad],
    lr=0.0005,
    weight_decay=0.001
)

# Phase 2: Fine-tuning
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.00005,                # 10x più basso
    weight_decay=0.001
)
```

### Scheduler: CosineAnnealingLR
```python
total_steps = epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=total_steps
)
```

### Learning rate schedule
- **Head-only**: 0.0005 → 0 (cosine decay)
- **Fine-tuning**: 0.00005 → 0 (cosine decay)
- **Step**: Ogni batch (non ogni epoch)

### Gradient clipping
```python
max_grad_norm = 1.0
nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

---

## 8. BATCH SIZE

### Config attuale
```yaml
batch_size: 12
```

### Motivazione
- ConvNeXt Tiny è più pesante di EfficientNet
- Colab GPU (T4/V100) ha memoria limitata
- Batch size 12 è un buon compromesso tra:
  - Stabilità training (batch non troppo piccolo)
  - Memory usage (non OOM)
  - Velocità (con num_workers=2)

### DataLoader settings
```python
num_workers: 2                 # Colab ha 2 CPU
pin_memory: True               # Faster GPU transfer
shuffle: True (train only)
```

---

## 9. STRATEGY DI SPLIT (train/val/test)

### Config attuale: `domain_aware_group_split_v1`
```yaml
split_strategy: "domain_aware"
split_include_food: false
train_ratio: 0.70
val_ratio: 0.15
test_ratio: 0.15
seed: 42
```

### Funzionamento
1. **Estrae photo_id** per ogni immagine (primi 4 caratteri filename)
2. **Raggruppa** tutte le versioni della stessa foto
3. **Stratifica** per `label|source|generator` (opzionalmente `|food_category`)
4. **Split** a livello di foto (non immagini)
5. **Verifica** no overlap tra train/val/test

### Stratification key
```python
strat_key = f"{label}|{source}|{generator}"
# Esempi:
# "0|originali|none"           → immagini reali buone
# "0|originali|none"           → immagini reali cattive
# "1|modificate|gpt"           → immagini generate da GPT
# "1|modificate|dalle"         → immagini generate da DALL-E
```

### Garanzie
✅ **No data leakage**: Versioni della stessa foto NON finiscono in train E test  
✅ **Domain balance**: Tutti i source/generator presenti in tutti gli split  
✅ **Label balance**: Proporzioni label simili tra split  
✅ **Reproducibility**: Seed fisso + split salvato su disco

### Alternative
- `group_v6`: Split senza stratificazione per dominio (solo per label+food)
- `stratified_group_split_v6`: DEPRECATED (ha data leakage)

---

## 10. METRICHE CALCOLATE

### Metriche principali (threshold-dependent)
```python
accuracy                       # (TP + TN) / (TP + TN + FP + FN)
precision                      # TP / (TP + FP)
recall                         # TP / (TP + FN)
f1                            # 2 * (prec * rec) / (prec + rec)
```

### Metriche threshold-independent
```python
roc_auc                       # Area under ROC curve
pr_auc                        # Area under Precision-Recall curve (MONITOR)
```

### Confusion matrix
```
[[TN  FP]
 [FN  TP]]
```

### Group metrics (per categoria)
Calcolate per:
- `food_category`: pasta, pizza, riso_paella, etc.
- `defect_type`: bruciato, crudo, muffa, etc.
- `generator`: gpt, dalle, midjourney, etc.
- `quality`: buono, cattivo

Per ogni gruppo:
- n_samples, n_pos, n_neg
- accuracy, precision, recall, f1
- roc_auc, pr_auc
- tp, fp, tn, fn

### Top errors
- **Top 50 False Positives**: Immagini reali predette come frodi (ordinate per confidenza)
- **Top 50 False Negatives**: Immagini generate predette come reali (ordinate per confidenza)

---

## 11. EVALUATION SU DATASET ESTERNO

### Script: `src/eval.py`

### Modalità 1: Internal test (default)
```bash
python -m src.eval \
    --config configs/convnext_v8.yaml \
    --run_name "eval_internal" \
    --checkpoint_path "checkpoints/best.pt" \
    --threshold 0.55
```
- Usa lo stesso dataset del training
- Applica lo stesso split (seed=42)
- Usa solo il test set (15%)

### Modalità 2: External test
```bash
python -m src.eval \
    --config configs/convnext_v8.yaml \
    --run_name "eval_external" \
    --checkpoint_path "checkpoints/best.pt" \
    --threshold 0.55 \
    --external_test_dataset "/path/to/external/dataset"
```
- Usa dataset COMPLETAMENTE SEPARATO
- Nessuno split (usa tutto il dataset come test)
- Valuta generalizzazione su dati mai visti

### Output evaluation
```
outputs/runs/<run_name>/
├── metrics.json                  # Metriche complete + config
├── notes.md                      # Riepilogo human-readable
├── predictions.csv               # Tutte le predizioni + metadati
├── group_metrics_*.csv           # Metriche per categoria
├── top_false_positives.csv       # Top 50 FP
├── top_false_negatives.csv       # Top 50 FN
├── cm.png                        # Confusion matrix
├── roc_curve.png                 # ROC curve
├── pr_curve.png                  # Precision-Recall curve
└── prob_dist.png                 # Distribuzione probabilità
```

### Threshold optimization (su validation set)
```python
# Durante training, trova threshold ottimale su validation
threshold_strategy: "f1"          # Ottimizza F1
# Oppure:
threshold_strategy: "cost_sensitive"  # Minimizza costo FP/FN
threshold_strategy: "max_fp_rate"     # Vincolo su FP rate massimo

# Threshold trovato viene salvato e usato per test
```

### Metriche salvate
- Test mode (internal/external)
- Threshold usato
- Metriche complete (acc, prec, rec, f1, roc_auc, pr_auc)
- Confusion matrix
- Config completo

---

## RIEPILOGO CONFIGURAZIONE ATTUALE (V8)

```yaml
# Dataset
dataset_root: "/content/drive/MyDrive/DatasetVerifoto/images/exp_3_augmented_v6.1_categorized"

# Model
model_name: "convnext_tiny"
drop_rate: 0.3

# Image
img_size: 224
batch_size: 12

# Training
seed: 42
epochs_head: 5
epochs_finetune: 35
lr_head: 0.0005
lr_finetune: 0.00005
weight_decay: 0.001
max_grad_norm: 1.0

# Loss
loss_type: "weighted_focal"
focal_alpha: 0.25
focal_gamma: 2.0
real_weight: 2.0

# Early stopping
patience: 8
monitor: "pr_auc"

# Augmentation
augmentation_strength: "strong"
real_augmentation_multiplier: 1.5

# Split
split_strategy: "domain_aware"
split_include_food: false

# Threshold
threshold: 0.55                   # Ottimizzato tramite photo-level analysis
threshold_strategy: "f1"
```

---

## NOTE FINALI

### Punti di forza
✅ Split domain-aware previene data leakage  
✅ Augmentation differenziata riduce overfitting su real  
✅ Weighted Focal Loss penalizza FP su real  
✅ Threshold ottimizzato su validation (0.55)  
✅ Metriche dettagliate per categoria  
✅ Riproducibilità garantita (seed + split salvato)

### Aree di miglioramento potenziali
- Calibrazione probabilità (Temperature Scaling, Platt Scaling)
- Hard negative mining (fine-tuning su errori)
- Ensemble di modelli
- Test-time augmentation (TTA)
- Threshold dinamico per categoria

---

**Fine Report Tecnico**
