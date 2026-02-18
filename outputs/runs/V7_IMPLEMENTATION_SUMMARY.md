# V7 Implementation Summary

## 🎯 Obiettivo

Ridurre il drop di performance su test esterno da **-36.7% F1** e **-48.1% precision** a <-15% F1 e <-20% precision.

## ✅ Implementazioni Completate

### 1. Modello: ConvNeXt-Tiny
- **File:** `configs/convnext_v7_improved.yaml`
- **Parametri:** 28M (vs 5M di EfficientNet-B0)
- **Benefici:** Migliore capacità di generalizzazione, ottimo per artefatti sottili

### 2. Loss Functions Avanzate
- **File:** `src/utils/losses.py`
- **Implementate:**
  - `FocalLoss`: Riduce peso su easy examples
  - `WeightedFocalLoss`: Focal + class weights + peso extra per real images
  - `CostSensitiveLoss`: Costi asimmetrici per FP e FN
- **Raccomandato:** `WeightedFocalLoss` con `real_weight=2.0`

### 3. Augmentations Avanzate
- **File:** `src/utils/augmentations.py`
- **Nuove augmentations:**
  - `RandomResizeDownUp`: Simula screenshot/resize
  - `RandomSharpening`: Simula post-processing
  - `RandomScreenshotArtifacts`: Simula bordi/crop/UI margins
  - `StrongAugmentationForReal`: Pipeline aggressiva per immagini reali
- **Benefici:** Riduce overfitting su dettagli fini delle immagini reali

### 4. Augmentation Differenziata
- **File:** `src/utils/data.py` (modificato)
- **Funzionalità:** Augmentation PIÙ FORTE per immagini reali (label=0)
- **Config:**
  ```yaml
  augmentation_strength: "strong"
  real_augmentation_multiplier: 1.5
  ```

### 5. Threshold Optimization Avanzata
- **File:** `src/utils/metrics.py` (modificato)
- **Nuove funzioni:**
  - `find_cost_sensitive_threshold()`: Minimizza costo totale
  - `find_threshold_with_max_fp_rate()`: Garantisce FP rate <= max
- **Config:**
  ```yaml
  threshold_strategy: "cost_sensitive"
  fp_cost: 2.0
  fn_cost: 1.0
  ```

### 6. Training Script V7
- **File:** `src/train_v7.py`
- **Features:**
  - Supporto per tutte le loss functions
  - Augmentation differenziata
  - Threshold optimization avanzata
  - Logging completo

## 📁 File Creati/Modificati

### Nuovi File
1. `configs/convnext_v7_improved.yaml` - Config completo
2. `src/utils/augmentations.py` - Augmentations avanzate
3. `src/utils/losses.py` - Loss functions avanzate
4. `src/train_v7.py` - Training script V7
5. `docs/V7_IMPROVEMENTS.md` - Documentazione completa
6. `.kiro/notes/v7_colab_quickstart.md` - Quick start per Colab

### File Modificati
1. `src/utils/data.py` - Aggiunto supporto per augmentation differenziata
2. `src/utils/metrics.py` - Aggiunte funzioni per threshold cost-sensitive

## 🚀 Come Usare

### Step 1: Prepara Dataset Unificato (Lato Tuo)
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

**Target:** 650-1100 immagini totali (vs 693 attuali)

### Step 2: Training
```bash
python src/train_v7.py \
  --config configs/convnext_v7_improved.yaml \
  --run_name 2026-02-19_convnext_v7 \
  --checkpoint_dir /path/to/checkpoints
```

### Step 3: Test Esterno
```bash
python src/eval.py \
  --checkpoint /path/to/2026-02-19_convnext_v7/best.pt \
  --external_dataset /path/to/whatsapp_Luca \
  --output_dir outputs/runs/2026-02-19_convnext_v7_external
```

### Step 4: Confronto
```bash
python scripts/compare_internal_external.py \
  --internal outputs/runs/2026-02-19_convnext_v7 \
  --external outputs/runs/2026-02-19_convnext_v7_external
```

## 📊 Risultati Attesi

### Baseline (2026-02-18_noK3_noLeakage)
- Test Interno: F1=0.803, Prec=0.672, Rec=0.997
- Test Esterno: F1=0.508, Prec=0.349, Rec=0.938
- Drop: F1=-36.7%, Prec=-48.1%

### Target V7
- Test Interno: F1>0.85, Prec>0.75, Rec>0.95
- Test Esterno: F1>0.65, Prec>0.55, Rec>0.85
- Drop: F1<-20%, Prec<-20%

### Improvement
- F1: +27% su test esterno
- Precision: +57% su test esterno
- Drop ridotto del 45%

## 🔑 Key Features

### 1. Weighted Focal Loss (CRITICO)
```python
# Riduce peso su easy examples
# Aumenta peso su hard examples
# Penalizza 2x errori su immagini reali
loss = WeightedFocalLoss(
    alpha=0.25,
    gamma=2.0,
    real_weight=2.0  # ← CHIAVE per ridurre FP su real
)
```

### 2. Augmentation Differenziata (IMPORTANTE)
```python
# Immagini reali: augmentation FORTE
real_transform = StrongAugmentationForReal()

# Immagini generate: augmentation LEGGERA
generated_transform = NormalAugmentation()

# Riduce overfitting su dettagli fini delle immagini reali
```

### 3. Threshold Cost-Sensitive (UTILE)
```python
# Minimizza costo totale invece di massimizzare F1
# FP costa 2x più di FN
threshold = find_cost_sensitive_threshold(
    probs, labels,
    fp_cost=2.0,  # ← Penalizza FP
    fn_cost=1.0
)
```

## 🎯 Perché Dovrebbe Funzionare

1. **ConvNeXt-Tiny (28M params):**
   - 5.6x più parametri = migliore capacità di generalizzazione
   - Architettura moderna ottimizzata per vision tasks

2. **Weighted Focal Loss:**
   - Riduce overfitting su easy negatives (immagini generate ovvie)
   - Focus su hard negatives (immagini reali difficili)
   - Penalizza 2x errori su real (dove sbagli di più)

3. **Augmentation Forte su Real:**
   - Riduce dipendenza da dettagli fini (JPEG artifacts, noise, etc.)
   - Migliora robustezza a variazioni (resize, compression, etc.)
   - Simula condizioni reali (WhatsApp, screenshot, etc.)

4. **Threshold Cost-Sensitive:**
   - Ottimizza per minimizzare FP (il tuo problema principale)
   - Bilancia FP e FN secondo i tuoi costi
   - Più precision-oriented che F1-oriented

5. **Dataset Unificato:**
   - Più dati = migliore generalizzazione
   - Più variabilità = modello più robusto
   - Bilanciamento riduce bias

## 🔧 Tuning Suggerito

### Se F1 è ancora basso (<0.65)
1. Aumenta `real_weight` a 3.0
2. Aumenta `fp_cost` a 3.0
3. Usa `real_augmentation_multiplier: 2.0`

### Se Recall è troppo bassa (<0.85)
1. Riduci `fp_cost` a 1.5
2. Usa `threshold_strategy: "f1"`

### Se Training è troppo lento
1. Riduci `batch_size` a 8
2. Usa `efficientnet_b2` invece di `convnext_tiny`

## 📝 Checklist

### Pre-Training
- [ ] Dataset unificato creato (650-1100 immagini)
- [ ] Config modificato con path corretti
- [ ] GPU disponibile (T4/V100/A100)
- [ ] Dependencies installate
- [ ] Git commit fatto

### Post-Training
- [ ] Test interno completato (F1 > 0.85)
- [ ] Test esterno completato (F1 > 0.65)
- [ ] Confronto fatto (drop < 20%)
- [ ] Risultati salvati su Drive
- [ ] Analisi errori fatta

### Se Target Non Raggiunto
- [ ] Hard negative mining con FP del test esterno
- [ ] Fine-tuning con hard negatives
- [ ] Ensemble di 3 modelli
- [ ] Wavelet preprocessing (2a iterazione)

## 💡 Note Importanti

1. **Dataset unificato è FONDAMENTALE:** Senza più dati reali, il miglioramento sarà limitato
2. **Weighted Focal Loss è la chiave:** Riduce overfitting su easy examples
3. **Real weight 2.0 è critico:** Penalizza errori dove sbagli di più
4. **Augmentation differenziata è importante:** Riduce overfitting su dettagli fini
5. **Threshold cost-sensitive ottimizza per il tuo caso:** Minimizza FP

## 🎉 Conclusione

Tutte le ottimizzazioni sono implementate e pronte per l'uso. Il prossimo passo è:

1. **Lato tuo:** Crea dataset unificato con tutte le immagini disponibili
2. **Lato mio:** Training con V7 e test su dataset esterno
3. **Insieme:** Analisi risultati e iterazione se necessario

**Obiettivo:** Ridurre drop da 36.7% a <15% in F1 e da 48.1% a <20% in precision.

---

**Data:** 2026-02-18  
**Versione:** 7.0  
**Status:** ✅ Ready for training  
**Next:** Dataset unificato + training
