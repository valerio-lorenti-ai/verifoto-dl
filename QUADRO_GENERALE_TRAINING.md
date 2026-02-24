# Quadro Generale del Training - Spiegazione Semplice

## 🎯 Cosa Stiamo Facendo

Stiamo allenando un modello di deep learning per **distinguere foto vere da foto generate dall'AI**. Il problema principale è che il modello funziona bene sui dati di test interni, ma peggiora molto su foto nuove mai viste (test esterno).

---

## 📊 Il Dataset

Abbiamo circa **1200 foto uniche**, ognuna con **8 versioni diverse** (crop, resize, JPEG compression, ecc.) per un totale di ~10.000 immagini.

### Composizione:
- **Foto VERE (label=0)**: Foto scattate con telefono, scaricate da internet, screenshot
  - Provenienze: `originali`, `kaggle_vale`, `modificate`
  
- **Foto GENERATE (label=1)**: Immagini create da AI
  - Generatori: `gpt_image_1_mini`, `gpt_image_1_5`

---

## 🔀 Come Dividiamo i Dati: Domain-Aware Split

### Il Problema Vecchio (group_based_split_v6)

Prima dividevamo le foto in modo casuale, ma questo creava problemi:

**Esempio concreto:**
- 90% delle foto Kaggle finivano nel training
- 10% delle foto Kaggle finivano nel test
- Il modello imparava a riconoscere "questa è Kaggle" invece di "questa è AI"

**Risultato:** Quando vedeva foto nuove da WhatsApp (mai viste), sbagliava tutto!

### La Soluzione: Domain-Aware Split ✅

Ora dividiamo le foto in modo **intelligente**:

```
Train:  70% delle foto
Val:    15% delle foto  
Test:   15% delle foto
```

**MA** garantendo che:
- Ogni provenienza (originali, kaggle, modificate) sia presente in TUTTI gli split con le stesse proporzioni
- Ogni generatore (GPT-mini, GPT-1.5) sia presente in TUTTI gli split con le stesse proporzioni

**Esempio pratico:**
```
Foto Kaggle:
  - 70% nel train
  - 15% nel val
  - 15% nel test

Foto GPT-mini:
  - 70% nel train
  - 15% nel val
  - 15% nel test
```

### Perché è Importante?

Il modello **non può più barare** imparando "segnali di dominio" (tipo "le foto Kaggle hanno sempre sfondo bianco"). Deve imparare i veri artefatti dell'AI.

### Prevenzione Data Leakage

Usiamo **group-based split** per photo_id:
- Se la foto `63ad` finisce nel test, TUTTE le sue 8 versioni vanno nel test
- Nessuna versione della stessa foto può finire nel train
- Questo previene che il modello "memorizzi" foto specifiche

---

## 🧠 Il Modello: ConvNeXt-Tiny

### Cosa Usiamo

**ConvNeXt-Tiny** - Un modello moderno (2022) con 28 milioni di parametri.

### Perché ConvNeXt?

- **5.6x più grande** di EfficientNet-B0 (il vecchio modello)
- Migliore per rilevare **artefatti sottili** dell'AI (texture strane, pattern ripetuti, bordi innaturali)
- Architettura moderna che generalizza meglio

### Come lo Alleniamo

**Fase 1: Head Training (5 epochs)**
- Congela il backbone (la parte pre-allenata)
- Allena solo la testa (ultimo layer)
- Learning rate alto: 0.0005

**Fase 2: Fine-Tuning (35 epochs)**
- Scongela tutto il modello
- Allena tutto insieme
- Learning rate basso: 0.00005
- Early stopping se non migliora per 8 epochs

---

## 🎯 Loss Function: Weighted Focal Loss

### Il Problema

Non tutti gli errori sono uguali:
- **Easy examples**: Foto generate ovvie (tipo con 6 dita) → il modello le riconosce subito
- **Hard examples**: Foto generate realistiche o foto vere difficili → il modello fatica

### La Soluzione: Focal Loss

Invece di dare lo stesso peso a tutti gli esempi, usiamo **Focal Loss**:

```
Esempio facile (modello sicuro al 99%) → peso BASSO (quasi ignorato)
Esempio difficile (modello incerto al 55%) → peso ALTO (focus su questo!)
```

**Parametri:**
- `focal_gamma = 2.0`: Quanto ridurre il peso degli esempi facili
- `focal_alpha = 0.25`: Peso per la classe positiva (AI generate)

### Weighted: Penalità Extra per Errori su Foto Vere

Aggiungiamo un peso extra (`real_weight = 2.0`) per errori su foto VERE:

**Perché?** Perché il problema principale è che il modello sbaglia troppo sulle foto vere (falsi positivi = dice "è AI" quando è vera).

```
Errore su foto generata: peso 1x
Errore su foto vera:     peso 2x ← penalità doppia!
```

---

## 🔄 Augmentation: Simulare il Mondo Reale

### Cosa Sono le Augmentation

Trasformazioni applicate alle immagini durante il training per renderle più varie:
- JPEG compression (simula WhatsApp, social media)
- Resize up/down (simula screenshot, re-upload)
- Blur, noise, sharpening
- Crop, flip, rotate

### Augmentation Differenziata

**Idea chiave:** Foto vere e foto generate hanno bisogno di augmentation DIVERSE!

**Foto VERE (label=0):**
- Augmentation FORTE (1.5x più aggressiva)
- Più JPEG, resize, blur, noise
- **Perché?** Per evitare che il modello memorizzi dettagli fini specifici delle foto di training

**Foto GENERATE (label=1):**
- Augmentation LEGGERA
- **Perché?** Hanno già i loro artefatti naturali (texture AI, pattern strani)

### Augmentation Speciali

1. **RandomResizeDownUp**: Simula screenshot → resize → re-upload
2. **RandomSharpening**: Simula filtri Instagram, post-processing
3. **RandomScreenshotArtifacts**: Simula bordi, crop, margini UI

---

## 🎚️ Threshold Optimization

### Il Problema

Il modello non dice "è AI" o "è vera", dice "probabilità che sia AI: 0.73".

Dobbiamo scegliere una **soglia (threshold)** per decidere:
- Se prob >= threshold → "è AI"
- Se prob < threshold → "è vera"

### Come Troviamo il Threshold Ottimale

**Attualmente usiamo: 0.55**

Trovato tramite **photo-level analysis** (analisi a livello di foto, non singole immagini):

```
Threshold 0.40: Recall 90.2%, Precision 74.2% → rileva più frodi, più falsi allarmi
Threshold 0.55: Recall 88.2%, Precision 77.6% → BILANCIATO ✅
Threshold 0.60: Recall 86.3%, Precision 77.2% → meno falsi allarmi, perde frodi
```

**Scelta:** 0.55 perché offre il miglior bilanciamento tra precision e recall.

---

## 🔥 Hard Negative Mining

### Cos'è un Hard Negative?

Un **hard negative** è una foto VERA che il modello sbaglia, classificandola come AI (falso positivo).

**Esempio:**
- Foto vera di una pizza
- Modello dice: "probabilità AI = 0.85" ← ERRORE!
- Questa è un hard negative

### Perché Facciamo Hard Negative Mining?

Dopo il primo training, identifichiamo le foto più difficili (quelle che il modello sbaglia) e le usiamo per **fine-tuning mirato**.

### Come Funziona

**Step 1: Training Iniziale**
- Allena il modello normalmente
- Valuta sul test set
- Identifica hard negatives (falsi positivi)

**Step 2: Identificazione Hard Negatives**
- Script `analyze_by_photo.py` analizza i risultati
- Trova foto vere classificate come AI
- Salva lista in `photo_hard_fp.csv`

**Step 3: Hard Negative Fine-Tuning**
- Ri-allena il modello
- Aumenta il peso degli hard negatives nel training set (3x)
- Il modello impara a non sbagliare più su queste foto difficili

### IMPORTANTE: Nessun Data Leakage!

Gli hard negatives identificati dal **test set** NON vengono usati per training!

**Workflow corretto:**
1. Identifica hard negatives dal test set (solo per analisi)
2. Fine-tuning usa solo hard negatives dal **train/val set**
3. Test set rimane completamente isolato
4. Usa lo STESSO split (domain-aware) per evitare leakage

---

## 📈 Risultati Attuali

### Performance Interna (Test Set)

Con threshold 0.55:
- **F1 Score: 82.6%**
- **Precision: 77.6%** (su 100 foto che dice "è AI", 77 sono davvero AI)
- **Recall: 88.2%** (su 100 foto AI, ne rileva 88)

### Obiettivi

**Test Interno:**
- F1 > 85%
- Precision > 75%
- Recall > 90%

**Test Esterno (foto nuove mai viste):**
- F1 > 65% (attualmente ~51%)
- Precision > 55% (attualmente ~35%)
- Drop < 20% rispetto al test interno

---

## 🔧 Configurazione Attuale (convnext_v8.yaml)

```yaml
# Modello
model_name: "convnext_tiny"
drop_rate: 0.3

# Training
epochs_head: 5
epochs_finetune: 35
lr_head: 0.0005
lr_finetune: 0.00005

# Loss
loss_type: "weighted_focal"
focal_alpha: 0.25
focal_gamma: 2.0
real_weight: 2.0  # Penalità 2x per errori su foto vere

# Split
split_strategy: "domain_aware"  # Previene data leakage + bilancia domini
split_include_food: false

# Threshold
threshold: 0.55  # Ottimale per F1

# Augmentation
augmentation_strength: "strong"
real_augmentation_multiplier: 1.5  # Augmentation più forte su foto vere
```

---

## 🎯 Workflow Completo

### 1. Training Iniziale
```bash
python src/train_v7.py \
  --config configs/convnext_v8.yaml \
  --run_name 2026-02-20_convnext_v8_domaiAware
```

**Output:**
- Modello allenato: `checkpoints/2026-02-20_convnext_v8_domaiAware/best.pt`
- Predizioni test: `outputs/runs/.../predictions.csv`
- Metriche: `outputs/runs/.../metrics.json`

### 2. Analisi Photo-Level
```bash
python scripts/analyze_by_photo.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware \
  --min-recall 0.90
```

**Output:**
- Metriche realistiche: `photo_level_metrics.json`
- Hard negatives: `photo_hard_fp.csv`, `photo_hard_fn.csv`
- Threshold ottimale: `chosen_threshold.json`

### 3. Hard Negative Fine-Tuning
```bash
python scripts/hard_negative_finetune.py \
  --config configs/convnext_v8.yaml \
  --run_dir outputs/runs/2026-02-20_convnext_v8_domaiAware \
  --repeat_factor 3.0
```

**Output:**
- Modello migliorato: `checkpoints/.../hard_negative/best.pt`
- Nuove metriche: `outputs/runs/.../hard_negative/metrics.json`

### 4. Valutazione Esterna
```bash
python src/eval.py \
  --checkpoint checkpoints/.../best.pt \
  --external_dataset /path/to/whatsapp_Luca \
  --output_dir outputs/runs/.../external
```

**Output:**
- Performance su dati esterni: `external/metrics.json`
- Confronto interno vs esterno

---

## 💡 Perché Questo Approccio Dovrebbe Funzionare

### 1. Domain-Aware Split
✅ Previene che il modello impari "segnali di dominio" invece di artefatti AI

### 2. ConvNeXt-Tiny
✅ Più parametri = migliore capacità di generalizzazione su dati nuovi

### 3. Weighted Focal Loss
✅ Focus su esempi difficili, penalità extra per errori su foto vere

### 4. Augmentation Differenziata
✅ Riduce overfitting su dettagli fini, migliora robustezza

### 5. Hard Negative Mining
✅ Migliora performance sugli errori più comuni

### 6. Threshold Ottimizzato
✅ Bilanciamento ottimale tra precision e recall

---

## 🚨 Problemi Risolti

### Data Leakage con Hard Negatives ✅ RISOLTO

**Problema:** Hard negative fine-tuning usava split diverso dal training originale
- Training: `domain_aware_group_split_v1`
- Hard negative: `group_based_split_v6` ← DIVERSO!
- Risultato: Foto dal test finivano nel train → leakage!

**Soluzione:** Hard negative fine-tuning ora usa STESSO split del training originale
- Legge `split_strategy` dal config
- Usa `domain_aware_group_split_v1` se specificato
- Test set rimane isolato

**Verifica:**
```bash
python scripts/verify_hard_negative_leakage.py
```

---

## 📝 Prossimi Passi

1. ✅ Training con domain-aware split
2. ✅ Analisi photo-level per trovare threshold ottimale
3. ⏳ Hard negative fine-tuning (con fix leakage)
4. ⏳ Valutazione su dataset esterno
5. ⏳ Confronto performance interno vs esterno
6. ⏳ Iterazione se necessario (più augmentation, più dati, ecc.)

---

**Creato:** 2026-02-24  
**Versione:** 8.1  
**Status:** In produzione
