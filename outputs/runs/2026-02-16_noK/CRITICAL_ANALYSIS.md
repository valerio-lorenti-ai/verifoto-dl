# Analisi Critica: Training senza kaggle_vale_con_id

**Data**: 2026-02-16  
**Dataset**: augmented_v6 (escluso kaggle_vale_con_id)  
**Modello**: EfficientNet-B0  
**Threshold**: 0.5

---

## 🎯 Metriche Principali

```
Accuracy:   91.05%
Precision:  86.06%  ← CRITICO per produzione
Recall:     98.99%
F1:         92.07%
PR-AUC:     96.39%
ROC-AUC:    97.60%
```

### Confusion Matrix
```
                Predicted
                NON_FRODE  FRODE
True NON_FRODE    297       64    (FP: 64)
True FRODE          4      395    (FN: 4)
```

**False Positive Rate: 17.7%** (64/361)  
**False Negative Rate: 1.0%** (4/399)

---

## ⚠️ PROBLEMA CRITICO: kaggle_vale_con_id ancora presente

**ATTENZIONE**: Il dataset contiene ancora 145 immagini di kaggle_vale_con_id!

```
kaggle_vale_con_id: 145 samples
  - F1: 0.000 (nessuna frode da rilevare)
  - FP: 2 (errori su immagini "facili")
  - Tutte originali/buono
```

### Impatto Reale

Se escludiamo kaggle_vale_con_id dal calcolo:

```
Samples:     615 (invece di 760)
TN:          154 (invece di 297)
FP:          62 (invece di 64)
FPR:         28.7% (invece di 17.7%)
```

**CONCLUSIONE**: Il FPR reale è 28.7%, non 17.7%. Quasi 1 su 3 reclami legittimi viene flaggato come frode.

---

## 📊 Performance per Categoria Cibo

### Categorie Eccellenti (F1 > 0.95)
- **insalata**: F1=1.000 (8 samples) - perfetto
- **pizza**: F1=0.952 (48 samples) - ottimo
- **burger_panino**: F1=0.948 (78 samples) - ottimo

### Categorie Buone (F1 0.90-0.95)
- **altro**: F1=0.946 (101 samples)
- **fritti**: F1=0.938 (20 samples)
- **riso_paella**: F1=0.927 (54 samples)
- **carne**: F1=0.926 (110 samples)
- **dessert**: F1=0.923 (15 samples)
- **sushi**: F1=0.917 (18 samples)
- **frutta_verdura**: F1=0.904 (68 samples)

### Categorie Problematiche
- **pasta**: F1=0.883 (67 samples) - 13 FP, 0 FN
- **patatine**: F1=0.824 (16 samples) - 3 FP, 0 FN
- **zuppa**: F1=0.400 (12 samples) - 3 FP, 0 FN ⚠️

---

## 🔍 Analisi False Positives (64 totali)

### Pattern Identificati

**1. Compressione/Qualità Bassa** (pattern dominante)
- `q50`, `q70`, `q95` - immagini con compressione JPEG
- `tiny_thumb` - thumbnail molto piccole
- `lowres_phone` - foto a bassa risoluzione
- `whatsapp` - compressione WhatsApp

**2. Manipolazioni Tecniche Legittime**
- `rotate_recomp` - rotazione + ricompressione
- `double` - doppia compressione
- `highres_crop` - crop da alta risoluzione
- `noise_light` - rumore leggero

**3. Categorie Più Colpite**
- **pasta**: 13 FP (19% dei campioni pasta)
- **carne**: 11 FP (10% dei campioni carne)
- **frutta_verdura**: 7 FP (10% dei campioni)
- **burger_panino**: 6 FP (8% dei campioni)

### Esempi Critici (confidence 1.0)

Tutti questi sono originali/buono ma predetti come frode con 100% confidence:
- `riso_paella/bb31_q70.jpg`
- `altro/8298_platform.jpg`
- `patatine/e02f_q95.jpg`
- `pasta/414d_rotate_recomp.jpg`

**PROBLEMA**: Il modello confonde artefatti di compressione/processing con manipolazioni AI.

---

## 🎯 Analisi False Negatives (4 totali)

### Frodi Mancate

1. **sushi/insetti** (prob=0.030) - confidence bassissima
   - `2c11_lowres_phone.jpg`
   - Bassa risoluzione + categoria rara (insetti)

2. **altro/marcio** (prob=0.207)
   - `9ab0_heavy.jpg`
   - Manipolazione "heavy" non rilevata

3. **riso_paella/crudo** (prob=0.412)
   - `0f3a_heavy.jpg`
   - Manipolazione "heavy" non rilevata

4. **altro/crudo** (prob=0.457)
   - `46c1_noise_light.jpg`
   - Rumore leggero confonde il modello

**PATTERN**: Manipolazioni "heavy" e immagini a bassa risoluzione sono più difficili da rilevare.

---

## 🔬 Performance per Tipo Difetto

### Eccellenti
- **bruciato**: F1=1.000 (147 samples) - perfetto
- **ammuffito**: F1=1.000 (24 samples) - perfetto
- **marcio**: F1=0.995 (109 samples) - quasi perfetto (1 FN)
- **crudo**: F1=0.993 (151 samples) - quasi perfetto (2 FN)

### Buone
- **insetti**: F1=0.952 (11 samples) - 1 FN

**CONCLUSIONE**: Il modello rileva molto bene tutti i tipi di difetto. I 4 FN sono casi edge (bassa risoluzione, manipolazioni heavy).

---

## 🤖 Confronto Generatori

```
gpt_image_1_mini:  F1=1.000 (198 samples) - perfetto
gpt_image_1_5:     F1=0.990 (201 samples) - 4 FN
```

**OSSERVAZIONE**: Tutti i 4 false negatives sono da gpt_image_1_5. Questo generatore produce manipolazioni leggermente più difficili da rilevare.

---

## 💼 Implicazioni per Produzione

### 1. False Positive Rate Troppo Alto

**FPR attuale: 17.7%** (ma 28.7% senza kaggle_vale_con_id)

In produzione:
- ~1 su 5-6 reclami legittimi flaggato come frode (scenario ottimistico)
- ~1 su 3 reclami legittimi flaggato come frode (scenario realistico)

**IMPATTO BUSINESS**:
- Ristoranti perdono fiducia nel sistema
- Clienti legittimi si sentono accusati ingiustamente
- Rischio di abbandono del servizio

### 2. Cause dei False Positives

Il modello confonde:
- **Compressione JPEG** (q50, q70) con manipolazioni AI
- **Thumbnail/lowres** con artefatti sospetti
- **Doppia compressione** con editing
- **Rumore/artefatti** con modifiche

**PROBLEMA FONDAMENTALE**: Il modello rileva artefatti tecnici, non manipolazioni semantiche.

### 3. Raccomandazioni Immediate

#### A. Aumentare Threshold (PRIORITÀ ALTA)

Testare threshold più alti per ridurre FPR:
- **0.6**: Stima FPR ~12-15%
- **0.7**: Stima FPR ~8-10% ← TARGET
- **0.8**: Stima FPR ~5-7%

Trade-off: Recall scenderà, ma è accettabile (meglio perdere qualche frode che accusare innocenti).

#### B. Escludere kaggle_vale_con_id (PRIORITÀ ALTA)

Il dataset contiene ancora questa categoria. Deve essere rimossa PRIMA del training:
- Modifica il path del dataset su Drive
- Oppure filtra nel parser (già implementato ma non usato)

#### C. Data Augmentation Mirata (PRIORITÀ MEDIA)

Aggiungere al training:
- Più variazioni di compressione JPEG (q30-q95)
- Più thumbnail/lowres legittime
- Più doppia compressione
- Più rumore/artefatti

Obiettivo: Insegnare al modello che questi artefatti sono NORMALI, non sospetti.

#### D. Feature Engineering (PRIORITÀ BASSA)

Considerare features aggiuntive:
- Analisi frequenze (DCT)
- Noise pattern analysis
- Compression artifact detection
- Metadata EXIF (se disponibile)

---

## 📈 Prossimi Passi

### Immediati (questa settimana)

1. **Ri-allenare senza kaggle_vale_con_id**
   - Verificare che il dataset Drive non contenga questa categoria
   - Oppure usare il filtro nel parser

2. **Testare threshold 0.7**
   - Rieseguire eval con `--threshold 0.7`
   - Analizzare nuovo FPR e recall

3. **Analizzare precision/recall curve**
   - Trovare threshold ottimale per FPR < 10%
   - Documentare trade-off precision/recall

### Breve termine (prossime 2 settimane)

4. **Migliorare data augmentation**
   - Aggiungere più compressione JPEG al training
   - Aggiungere più thumbnail/lowres
   - Testare se FPR migliora

5. **Analisi qualitativa FP**
   - Visualizzare i top 20 FP
   - Capire pattern visivi comuni
   - Decidere se servono features aggiuntive

### Medio termine (prossimo mese)

6. **Ensemble o modello più grande**
   - Testare EfficientNet-B1 o B2
   - Considerare ensemble di modelli
   - Valutare se migliora FPR

7. **Test su dati reali**
   - Se disponibili, testare su casi reali di frode
   - Validare che il modello generalizza

---

## ✅ Punti di Forza

1. **Recall eccellente (98.99%)**: Quasi tutte le frodi vengono rilevate
2. **Performance per difetto ottima**: Tutti i tipi di difetto rilevati bene
3. **Generalizzazione buona**: Funziona su tutte le categorie cibo
4. **Pochi false negatives (4)**: Casi edge, non pattern sistematici

---

## ❌ Punti Critici

1. **FPR troppo alto (28.7% reale)**: Inaccettabile per produzione
2. **Confusione artefatti tecnici**: Compressione/lowres scambiati per frode
3. **kaggle_vale_con_id ancora presente**: Gonfia le metriche
4. **Threshold 0.5 troppo basso**: Serve 0.7-0.8 per produzione

---

## 🎯 Obiettivo Produzione

**Target**: FPR < 10% con Recall > 90%

**Strategia**:
1. Escludere kaggle_vale_con_id
2. Threshold 0.7-0.8
3. Migliorare data augmentation
4. Validare su dati reali

**Timeline**: 2-3 settimane per modello production-ready
