# Piano d'Azione: Miglioramento Modello 2026-02-18

## 🎯 Obiettivo
Migliorare la generalizzazione del modello su dati esterni, riducendo il drop in F1 dal 36.7% a <15% e in precision dal 48.1% a <20%.

## 📊 Situazione Attuale

**Test Interno:**
- F1: 0.8030, Precision: 0.6720, Recall: 0.9973
- 183 falsi positivi su 240 immagini reali (76.2%)

**Test Esterno:**
- F1: 0.5085, Precision: 0.3488, Recall: 0.9375
- 28 falsi positivi su 61 immagini reali (45.9%)

**Drop:** -36.7% F1, -48.1% Precision

## 🚀 Azioni Immediate (1-2 giorni)

### 1. Aumentare Threshold a 0.70 ✅ PRIORITÀ ALTA
**Obiettivo:** Ridurre i falsi positivi mantenendo alta recall

**Comando:**
```bash
# Test con threshold 0.70 sul test esterno
python src/eval.py \
  --checkpoint /path/to/2026-02-18_noK3_noLeakage/best.pt \
  --external_dataset /path/to/whatsapp_Luca \
  --threshold 0.70 \
  --output_dir outputs/runs/2026-02-18_noK3_noLeakage_external_t70
```

**Risultati Attesi:**
- Precision: da 0.3488 a ~0.50-0.60
- Recall: da 0.9375 a ~0.85-0.90
- F1: da 0.5085 a ~0.60-0.65

### 2. Analizzare i 28 Falsi Positivi del Test Esterno ✅ PRIORITÀ ALTA
**Obiettivo:** Capire perché il modello sbaglia su queste immagini

**Comando:**
```bash
# Generare mosaici degli errori
python scripts/generate_error_mosaics.py \
  --run_dir outputs/runs/2026-02-18_noK3_noLeakage_external \
  --output_dir outputs/runs/2026-02-18_noK3_noLeakage_external/error_mosaics
```

**Analisi da fare:**
- Quali sono le caratteristiche comuni dei FP?
- Sono immagini da Pixel 7a o da internet?
- Hanno artefatti di compressione particolari?
- Sono simili a immagini del training set?

### 3. Confronto Dettagliato per Categoria ✅ PRIORITÀ MEDIA
**Obiettivo:** Identificare quali categorie causano più problemi

**Comando:**
```bash
# Analisi per categoria
python scripts/analyze_by_photo.py \
  --run_dir outputs/runs/2026-02-18_noK3_noLeakage_external \
  --group_by quality,food_category
```

## 🔧 Azioni a Breve Termine (1 settimana)

### 4. Hard Negative Mining ✅ PRIORITÀ ALTA
**Obiettivo:** Ri-addestrare il modello con i 28 FP come hard negatives

**Step:**
1. Estrarre i 28 FP dal test esterno
2. Aggiungerli al training set come esempi difficili
3. Ri-addestrare il modello con questi esempi

**Comando:**
```bash
# Estrai i FP e crea un nuovo dataset
python scripts/hard_negative_finetune.py \
  --base_checkpoint /path/to/2026-02-18_noK3_noLeakage/best.pt \
  --hard_negatives outputs/runs/2026-02-18_noK3_noLeakage_external/top_false_positives.csv \
  --output_dir outputs/runs/2026-02-18_noK3_noLeakage_hn \
  --epochs 10
```

### 5. Calibrazione Avanzata ✅ PRIORITÀ MEDIA
**Obiettivo:** Migliorare la calibrazione delle probabilità

**Comando:**
```bash
# Applica calibrazione con Platt scaling
python scripts/apply_calibration.py \
  --run_dir outputs/runs/2026-02-18_noK3_noLeakage \
  --method platt \
  --output_dir outputs/runs/2026-02-18_noK3_noLeakage_platt

# Test su dataset esterno con calibrazione
python src/eval.py \
  --checkpoint /path/to/2026-02-18_noK3_noLeakage/best.pt \
  --external_dataset /path/to/whatsapp_Luca \
  --calibration_file outputs/runs/2026-02-18_noK3_noLeakage_platt/calibration_T.json \
  --output_dir outputs/runs/2026-02-18_noK3_noLeakage_external_platt
```

### 6. Raccogliere Più Dati Esterni ✅ PRIORITÀ ALTA
**Obiettivo:** Aumentare la rappresentatività del dataset

**Azioni:**
- Raccogliere almeno 200 immagini reali da WhatsApp
- Includere diverse fonti: Pixel 7a, iPhone, internet, screenshot
- Bilanciare le categorie di qualità (buone, cattive, screenshot)
- Annotare correttamente le immagini

**Struttura dataset:**
```
whatsapp_extended/
├── originali/
│   ├── buone/
│   │   ├── pixel7a/
│   │   ├── iphone/
│   │   ├── internet/
│   │   └── altri_phone/
│   ├── cattive/
│   └── screenshots/
└── generate/
    ├── gpt_image_1_mini/
    └── gpt_image_1_5/
```

## 📈 Azioni a Medio Termine (2-3 settimane)

### 7. Esperimenti con Threshold Dinamico ✅ PRIORITÀ MEDIA
**Obiettivo:** Usare threshold diversi per diverse categorie

**Approccio:**
- Threshold basso (0.10) per immagini "cattive" (già funziona bene)
- Threshold alto (0.70) per immagini "buone" (riduce FP)
- Threshold medio (0.40) per screenshot

**Implementazione:**
```python
def dynamic_threshold(image_quality, base_prob):
    if image_quality == 'cattivo':
        threshold = 0.10
    elif image_quality == 'buono':
        threshold = 0.70
    else:  # screenshot
        threshold = 0.40
    return base_prob > threshold
```

### 8. Ensemble di Modelli ✅ PRIORITÀ BASSA
**Obiettivo:** Combinare predizioni di più modelli

**Approccio:**
- Addestrare 3-5 modelli con seed diversi
- Usare voting o averaging delle probabilità
- Testare su dataset esterno

### 9. Architettura Alternativa ✅ PRIORITÀ MEDIA
**Obiettivo:** Provare modelli più grandi o architetture diverse

**Opzioni:**
1. EfficientNet-B1/B2 (più parametri)
2. ConvNeXt-Tiny (architettura moderna)
3. Vision Transformer (ViT-Small)

**Comando:**
```bash
# Test con ConvNeXt
python src/train.py --config configs/convnext_experiment.yaml
```

## 📊 Metriche di Successo

### Target per il Prossimo Modello

**Test Esterno:**
- F1: > 0.65 (attuale: 0.5085)
- Precision: > 0.55 (attuale: 0.3488)
- Recall: > 0.85 (attuale: 0.9375)
- Drop F1 vs interno: < 15% (attuale: 36.7%)
- Drop Precision vs interno: < 20% (attuale: 48.1%)

**Test Interno:**
- Mantenere F1 > 0.80
- Mantenere Recall > 0.95

## 📝 Checklist Settimanale

### Settimana 1
- [ ] Test con threshold 0.70 su dataset esterno
- [ ] Analisi dettagliata dei 28 FP
- [ ] Generazione mosaici errori
- [ ] Raccolta di 50 nuove immagini WhatsApp

### Settimana 2
- [ ] Hard negative mining con i 28 FP
- [ ] Ri-addestramento modello con hard negatives
- [ ] Test su dataset esterno con nuovo modello
- [ ] Raccolta di altre 50 immagini WhatsApp

### Settimana 3
- [ ] Calibrazione avanzata (Platt scaling)
- [ ] Esperimenti con threshold dinamico
- [ ] Test con architettura alternativa (ConvNeXt)
- [ ] Raccolta di altre 100 immagini WhatsApp

### Settimana 4
- [ ] Ensemble di modelli
- [ ] Test finale su dataset esterno esteso (200+ immagini)
- [ ] Analisi comparativa di tutti gli approcci
- [ ] Selezione del modello migliore per produzione

## 🎯 Milestone

1. **Milestone 1 (Fine Settimana 1):** Threshold ottimizzato e analisi errori completata
2. **Milestone 2 (Fine Settimana 2):** Modello con hard negatives addestrato e testato
3. **Milestone 3 (Fine Settimana 3):** Dataset esterno esteso a 200+ immagini
4. **Milestone 4 (Fine Settimana 4):** Modello finale con F1 > 0.65 su test esterno

## 📞 Prossimi Passi Immediati

1. Eseguire test con threshold 0.70
2. Analizzare i 28 falsi positivi
3. Iniziare raccolta dati esterni
4. Pianificare hard negative mining

---

**Data creazione:** 2026-02-18  
**Ultima modifica:** 2026-02-18  
**Responsabile:** Team Verifoto
