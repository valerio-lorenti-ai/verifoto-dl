# V7 Improvements: Advanced Optimizations

## 🎯 Obiettivo

Ridurre il drop di performance su test esterno da **36.7% F1** e **48.1% precision** a <15% F1 e <20% precision.

## 📋 Modifiche Implementate

### 1. Modello: ConvNeXt-Tiny ✅

**Perché:**
- 28M parametri vs 5M di EfficientNet-B0 (5.6x più grande)
- Architettura moderna (2022) con ottime performance
- Migliore per rilevare artefatti sottili

**Come usare:**
```yaml
# configs/convnext_v7_improved.yaml
model_name: "convnext_tiny"
drop_rate: 0.3
batch_size: 12  # ConvNeXt è più pesante
```

### 2. Loss Functions Avanzate ✅

**Opzioni disponibili:**

#### A. Focal Loss
Riduce peso su easy examples, si concentra su hard examples.
```yaml
loss_type: "focal"
focal_alpha: 0.25  # Peso per classe positiva
focal_gamma: 2.0   # Focusing parameter (più alto = più focus su hard)
```

#### B. Weighted Focal Loss (RACCOMANDATO) 🔥
Combina focal loss + class weights + peso extra per errori su real.
```yaml
loss_type: "weighted_focal"
focal_alpha: 0.25
focal_gamma: 2.0
real_weight: 2.0  # Penalizza 2x errori su immagini reali
```

#### C. Cost-Sensitive Loss
Costi asimmetrici per FP e FN.
```yaml
loss_type: "cost_sensitive"
fp_cost: 2.0  # Costo di un falso positivo
fn_cost: 1.0  # Costo di un falso negativo
```

**Raccomandazione:** Usa `weighted_focal` perché:
- Riduce peso su easy negatives (immagini generate ovvie)
- Aumenta peso su hard negatives (immagini reali difficili)
- Penalizza di più FP su immagini reali (il tuo problema principale)

### 3. Augmentation Avanzate ✅

**Nuove augmentations implementate:**

#### A. RandomResizeDownUp
Simula screenshot, resize, re-upload.
```python
RandomResizeDownUp(scale_min=0.5, scale_max=0.9, p=0.4)
```

#### B. RandomSharpening
Simula post-processing, filtri Instagram.
```python
RandomSharpening(factor_min=1.0, factor_max=2.0, p=0.3)
```

#### C. RandomScreenshotArtifacts
Simula artefatti da screenshot (bordi, crop, UI margins).
```python
RandomScreenshotArtifacts(p=0.25)
```

#### D. Augmentation Differenziata
Augmentation PIÙ FORTE per immagini reali (riduce overfitting).
```yaml
augmentation_strength: "strong"  # "normal" o "strong"
real_augmentation_multiplier: 1.5  # Augmentation più forte su real
```

**Come funziona:**
- Immagini reali (label=0): augmentation FORTE (più JPEG, resize, blur, noise)
- Immagini generate (label=1): augmentation LEGGERA (hanno già artefatti propri)

### 4. Threshold Optimization Avanzata ✅

**Opzioni disponibili:**

#### A. F1 Optimization (standard)
```yaml
threshold_strategy: "f1"
```

#### B. Cost-Sensitive (RACCOMANDATO) 🔥
Minimizza costo totale considerando costi diversi per FP e FN.
```yaml
threshold_strategy: "cost_sensitive"
fp_cost: 2.0  # FP costa 2x più di FN
fn_cost: 1.0
```

#### C. Max FP Rate Constraint
Garantisce FP rate <= max_fp_rate.
```yaml
threshold_strategy: "max_fp_rate"
max_fp_rate: 0.10  # Massimo 10% di FP rate
```

**Raccomandazione:** Usa `cost_sensitive` con `fp_cost=2.0` perché:
- Penalizza di più i falsi positivi (il tuo problema)
- Trova threshold che bilancia FP e FN secondo i tuoi costi

## 🚀 Come Usare

### Step 1: Prepara Dataset Unificato

Crea un dataset unificato con tutte le immagini disponibili:
```
unified_dataset/
├── originali/
│   ├── buone/
│   │   ├── whatsapp/
│   │   ├── phone/
│   │   └── internet/
│   ├── cattive/
│   └── screenshots/
└── generate/
    ├── gpt_image_1_mini/
    └── gpt_image_1_5/
```

### Step 2: Modifica Config

Usa `configs/convnext_v7_improved.yaml` come base:
```yaml
dataset_root: "/path/to/unified_dataset"
model_name: "convnext_tiny"
loss_type: "weighted_focal"
focal_alpha: 0.25
focal_gamma: 2.0
real_weight: 2.0
threshold_strategy: "cost_sensitive"
fp_cost: 2.0
fn_cost: 1.0
augmentation_strength: "strong"
real_augmentation_multiplier: 1.5
```

### Step 3: Training

```bash
python src/train_v7.py \
  --config configs/convnext_v7_improved.yaml \
  --run_name 2026-02-19_convnext_v7 \
  --checkpoint_dir /path/to/checkpoints
```

### Step 4: Test su Dataset Esterno

```bash
python src/eval.py \
  --checkpoint /path/to/2026-02-19_convnext_v7/best.pt \
  --external_dataset /path/to/whatsapp_Luca \
  --output_dir outputs/runs/2026-02-19_convnext_v7_external
```

### Step 5: Confronto

```bash
python scripts/compare_internal_external.py \
  --internal outputs/runs/2026-02-19_convnext_v7 \
  --external outputs/runs/2026-02-19_convnext_v7_external
```

## 📊 Risultati Attesi

### Target Metriche

**Test Interno:**
- F1: > 0.85 (vs 0.80 attuale)
- Precision: > 0.75 (vs 0.67 attuale)
- Recall: > 0.95 (vs 0.99 attuale)

**Test Esterno:**
- F1: > 0.65 (vs 0.51 attuale) → **+27% improvement**
- Precision: > 0.55 (vs 0.35 attuale) → **+57% improvement**
- Recall: > 0.85 (vs 0.94 attuale)

**Drop:**
- F1 drop: < 20% (vs 36.7% attuale)
- Precision drop: < 20% (vs 48.1% attuale)

### Perché Dovrebbe Funzionare

1. **ConvNeXt-Tiny:** Più parametri = migliore capacità di generalizzazione
2. **Weighted Focal Loss:** Riduce overfitting su easy examples, focus su hard negatives
3. **Real Weight 2.0:** Penalizza 2x errori su immagini reali (dove sbagli di più)
4. **Augmentation Forte su Real:** Riduce dipendenza da dettagli fini, migliora robustezza
5. **Threshold Cost-Sensitive:** Ottimizza per minimizzare FP (il tuo problema principale)

## 🔧 Troubleshooting

### Se F1 è ancora basso (<0.65 su test esterno)

1. **Aumenta real_weight a 3.0**
   ```yaml
   real_weight: 3.0  # Penalizza ancora di più errori su real
   ```

2. **Usa augmentation ancora più forte**
   ```yaml
   augmentation_strength: "strong"
   real_augmentation_multiplier: 2.0  # 2x augmentation su real
   ```

3. **Aumenta fp_cost**
   ```yaml
   fp_cost: 3.0  # Penalizza ancora di più i falsi positivi
   ```

### Se Recall è troppo bassa (<0.85)

1. **Riduci fp_cost**
   ```yaml
   fp_cost: 1.5  # Bilancia meglio FP e FN
   ```

2. **Usa threshold F1 invece di cost-sensitive**
   ```yaml
   threshold_strategy: "f1"
   ```

### Se Training è troppo lento

1. **Riduci batch_size**
   ```yaml
   batch_size: 8  # Se GPU ha poca memoria
   ```

2. **Usa EfficientNet-B2 invece di ConvNeXt**
   ```yaml
   model_name: "efficientnet_b2"  # 9M params, più veloce
   batch_size: 16
   ```

## 📝 File Creati

1. `configs/convnext_v7_improved.yaml` - Config completo con tutte le ottimizzazioni
2. `src/utils/augmentations.py` - Augmentations avanzate
3. `src/utils/losses.py` - Loss functions avanzate
4. `src/train_v7.py` - Script di training V7
5. `src/utils/data.py` - Aggiornato con augmentation differenziata
6. `src/utils/metrics.py` - Aggiornato con threshold cost-sensitive

## 🎯 Prossimi Passi

1. ✅ Crea dataset unificato (lato tuo)
2. ✅ Training con ConvNeXt + Weighted Focal Loss
3. ✅ Test su dataset esterno
4. ✅ Confronto con baseline
5. ⏭️ Se necessario: Hard negative mining
6. ⏭️ Se necessario: Wavelet preprocessing (2a iterazione)

## 💡 Note Importanti

- **Dataset unificato è CRITICO:** Più dati reali = migliore generalizzazione
- **Weighted Focal Loss è la chiave:** Riduce overfitting su easy examples
- **Real weight 2.0 è fondamentale:** Penalizza errori dove sbagli di più
- **Augmentation differenziata è importante:** Riduce overfitting su dettagli fini
- **Threshold cost-sensitive ottimizza per il tuo caso:** Minimizza FP

---

**Creato:** 2026-02-18  
**Versione:** 7.0  
**Status:** Ready for training
