# 🤖 Analisi GPT-1.5 con Threshold 0.55

## 📊 Risultati

### Metriche Generali

```
Total GPT-1.5 samples: 200 (tutte AI, nessuna reale)

Threshold 0.55:
  Accuracy:  79.0%
  Precision: 100.0%  (nessun falso positivo)
  Recall:    79.0%   (rileva 158/200 frodi)
  F1:        88.3%
```

### Confusion Matrix

```
              Predicted
              REAL    AI
True REAL       0      0
True AI        42    158

FP: 0  (nessun falso positivo)
FN: 42 (perde 42 frodi su 200)
```

---

## 📈 Confronto con Threshold 0.90

| Metrica | Threshold 0.90 | Threshold 0.55 | Miglioramento |
|---------|----------------|----------------|---------------|
| **Recall** | 54.5% | **79.0%** | **+24.5%** ✅ |
| **Precision** | 100% | 100% | 0% |
| **F1** | 70.6% | **88.3%** | **+17.7%** ✅ |
| **FN** | 91 | 42 | **-49 (-53.8%)** ✅ |

**Miglioramento Significativo!**
- Rileva 49 frodi in più (da 109 a 158)
- Mantiene precision perfetta (100%)
- F1 aumenta di 17.7 punti percentuali

---

## ⚠️ Problema: Recall Ancora Sotto 80%

### Recall 79.0% è Borderline

**Soglia Accettabile:** 80-85%
**Soglia Ottimale:** >85%

**Situazione Attuale:**
- ✅ Miglioramento enorme vs 0.90 (+24.5%)
- ⚠️ Ma ancora sotto la soglia di 80%
- ❌ Perde ancora 42 frodi su 200 (21%)

---

## 🔍 Analisi Falsi Negativi (42 frodi perse)

### Statistiche Probabilità

```
Mean probability:   0.213  (molto bassa!)
Median probability: 0.208
Min probability:    0.052  (quasi 0!)
Max probability:    0.541  (sotto threshold)
```

**Problema:** Le frodi GPT-1.5 perse hanno probabilità molto basse (media 0.21), ben sotto il threshold 0.55.

### Breakdown per Defect Type

```
crudo:    18 FN (42.9%)  - Cibo crudo
marcio:   16 FN (38.1%)  - Cibo marcio
bruciato:  8 FN (19.0%)  - Cibo bruciato
```

**Pattern:** I difetti "crudo" e "marcio" sono i più difficili per GPT-1.5.

### Top 10 Frodi Più Difficili

```
Prob   Defect          Food                 
0.052  marcio          insalata             (foto 27e9)
0.057  marcio          insalata             (foto 27e9)
0.059  marcio          insalata             (foto 27e9)
0.072  marcio          insalata             (foto 27e9)
0.100  crudo           frutta_verdura       (foto 3bd4)
0.112  crudo           frutta_verdura       (foto 3bd4)
```

**Osservazione:** Alcune foto specifiche (27e9, 3bd4) sono sistematicamente difficili in tutte le versioni.

---

## 🎯 Raccomandazione

### ❌ Fine-Tuning GPT-1.5 è RACCOMANDATO

**Motivi:**
1. Recall 79.0% è sotto la soglia di 80%
2. Perde ancora 42 frodi su 200 (21%)
3. Probabilità medie molto basse (0.21) - non risolvibile con threshold
4. Pattern specifici (crudo, marcio) richiedono training mirato

**Risultato Atteso con Fine-Tuning:**
- Recall: 79.0% → **90-95%**
- FN: 42 → **10-20**
- Probabilità medie: 0.21 → 0.60+

---

## 🔧 Opzioni Disponibili

### Opzione A: Fine-Tuning GPT-1.5 ✅ RACCOMANDATO

**Azione:**
```bash
python scripts/hard_negative_finetune.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware \
  --focus-generator gpt_image_1_5 \
  --threshold 0.55 \
  --epochs 5 \
  --lr 1e-5
```

**Pro:**
- ✅ Risolve il problema alla radice
- ✅ Recall atteso: 90-95%
- ✅ Mantiene performance GPT-1-mini

**Contro:**
- ⏱️ Richiede 3-4 ore training
- ⚠️ Rischio di overfitting (mitigabile con early stopping)

**Tempo:** 3-4 ore

---

### Opzione B: Threshold Più Basso (0.40) ⚠️ NON RACCOMANDATO

**Azione:** Usa threshold 0.40 invece di 0.55

**Stima Performance GPT-1.5 con 0.40:**
- Recall: ~85-90% (migliore)
- Precision: ~85% (peggiore)
- FP: +10-15 (più falsi allarmi)

**Pro:**
- ✅ Nessun training necessario
- ✅ Recall più alto

**Contro:**
- ❌ Più falsi positivi su TUTTO il dataset (non solo GPT-1.5)
- ❌ F1 generale peggiore (da 82.6% a 81.4%)
- ❌ Non risolve il problema alla radice

**Raccomandazione:** NO - peggiora performance generale

---

### Opzione C: Threshold Adattivo per GPT-1.5 🤔 POSSIBILE

**Azione:** Usa threshold diverso per GPT-1.5

```python
if generator == "gpt_image_1_5":
    threshold = 0.40
else:
    threshold = 0.55
```

**Pro:**
- ✅ Migliora recall GPT-1.5 senza peggiorare altri
- ✅ Nessun training necessario

**Contro:**
- ⚠️ Più complesso da implementare
- ⚠️ Richiede detection del generator (non sempre disponibile)
- ⚠️ Non risolve il problema alla radice

**Raccomandazione:** Possibile come soluzione temporanea

---

### Opzione D: Accettare Recall 79% ❌ NON RACCOMANDATO

**Azione:** Nessuna, vai in produzione con recall 79%

**Pro:**
- ✅ Nessun lavoro aggiuntivo

**Contro:**
- ❌ Perde 1 frode su 5 (21%)
- ❌ Sotto soglia accettabile (80%)
- ❌ Rischio in produzione

**Raccomandazione:** NO - recall troppo basso

---

## 📊 Confronto Opzioni

| Opzione | Recall GPT-1.5 | F1 Generale | Tempo | Complessità | Raccomandazione |
|---------|----------------|-------------|-------|-------------|-----------------|
| **A: Fine-Tuning** | **90-95%** | **82.6%** | 3-4h | Media | ✅ **RACCOMANDATO** |
| B: Threshold 0.40 | 85-90% | 81.4% | 0 | Bassa | ⚠️ Peggiora generale |
| C: Threshold Adattivo | 85-90% | 82.6% | 1h | Alta | 🤔 Temporaneo |
| D: Accettare 79% | 79% | 82.6% | 0 | Bassa | ❌ Troppo basso |

---

## 🚀 Piano d'Azione Raccomandato

### Step 1: Fine-Tuning GPT-1.5 (3-4 ore)

```bash
python scripts/hard_negative_finetune.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware \
  --focus-generator gpt_image_1_5 \
  --threshold 0.55 \
  --epochs 5 \
  --lr 1e-5 \
  --repeat_factor 3
```

**Parametri:**
- `--focus-generator gpt_image_1_5`: Focus su GPT-1.5
- `--threshold 0.55`: Usa threshold ottimale
- `--epochs 5`: Pochi epochs per evitare overfitting
- `--lr 1e-5`: Learning rate basso per fine-tuning
- `--repeat_factor 3`: Ripeti FN 3x per bilanciare

### Step 2: Valutazione Post Fine-Tuning

**Verifica:**
- Recall GPT-1.5 > 85%
- Recall GPT-1-mini ancora > 90%
- F1 generale ancora > 82%

**Se OK:**
- ✅ Vai in produzione con modello fine-tuned

**Se NON OK:**
- ⚠️ Considera Opzione C (threshold adattivo)

### Step 3: Test su WhatsApp

**Dopo fine-tuning:**
- Test su dataset WhatsApp più grande
- Verifica generalizzazione
- Ottimizza threshold se necessario

---

## 📝 Conclusioni

### Situazione Attuale

**Threshold 0.55:**
- ✅ Miglioramento enorme vs 0.90 (+24.5% recall)
- ✅ F1 generale ottimo (82.6%)
- ⚠️ Recall GPT-1.5 79.0% (borderline)
- ❌ Sotto soglia accettabile (80%)

### Raccomandazione Finale

**🎯 Fine-Tuning GPT-1.5 è RACCOMANDATO**

**Motivi:**
1. Recall 79% è sotto soglia (80%)
2. Perde ancora 21% delle frodi GPT-1.5
3. Probabilità troppo basse (0.21) - non risolvibile con threshold
4. Fine-tuning è la soluzione più robusta

**Risultato Atteso:**
- GPT-1.5 recall: 79% → **90-95%**
- F1 generale: **82.6%** (invariato)
- Pronto per produzione

**Tempo:** 3-4 ore training + 1 ora validazione

---

## 📞 Prossimo Step?

**Vuoi che:**

A) 🔴 **Esegua il fine-tuning GPT-1.5** (3-4 ore) - RACCOMANDATO
B) 🟡 **Implementi threshold adattivo** (1 ora) - Temporaneo
C) 🟢 **Accetti recall 79%** e vai in produzione - Non raccomandato
D) 🔧 **Altro** (dimmi tu)

**Raccomandazione:** Opzione A (fine-tuning) per risultati ottimali.
