# 📊 Analisi Risultati: ConvNeXt V8 Domain-Aware

## Run Analizzati
- **Internal Test:** `2026-02-20_convnext_v8_domaiAware`
- **External Test:** `2026-02-20_convnext_v8_domaiAware_external`

---

## 🎯 RISULTATI INTERNAL TEST

### Metriche Generali (threshold=0.90)

| Metrica | Valore | Valutazione |
|---------|--------|-------------|
| **Accuracy** | 74.7% | ⚠️ Moderato |
| **Precision** | 79.2% | ✅ Buono |
| **Recall** | 68.9% | ⚠️ Basso |
| **F1 Score** | 73.7% | ⚠️ Moderato |
| **ROC-AUC** | 88.2% | ✅ Ottimo |
| **PR-AUC** | 87.1% | ✅ Ottimo |

### Confusion Matrix

```
                Predicted
                NON_FRODE  FRODE
True NON_FRODE     311       74    (FP: 74, 19.2%)
True FRODE         127      281    (FN: 127, 31.1%)
```

### Analisi Threshold

**Threshold scelto:** 0.90 (cost-sensitive con fp_cost=2.0)

**Problema:** Threshold troppo alto!
- ❌ Recall solo 68.9% (perde 31% delle frodi)
- ✅ Precision 79.2% (buona, ma non eccezionale)
- ⚠️ Trade-off sbilanciato verso precision

**Threshold ottimale F1:** ~0.70-0.75
- Recall stimato: ~85%
- Precision stimato: ~75%
- F1 stimato: ~80%

---

## 🤖 PERFORMANCE PER GENERATOR

### GPT-Image-1-Mini

| Metrica | Valore | Valutazione |
|---------|--------|-------------|
| Samples | 208 | - |
| **Recall** | 82.7% | ✅ Buono |
| **Precision** | 100% | ✅ Perfetto |
| **F1** | 90.5% | ✅ Ottimo |

**Analisi:**
- ✅ Modello rileva bene GPT-1-mini
- ✅ Nessun falso positivo
- ⚠️ 36 falsi negativi (17.3%)

### GPT-Image-1.5

| Metrica | Valore | Valutazione |
|---------|--------|-------------|
| Samples | 200 | - |
| **Recall** | 54.5% | ❌ CRITICO |
| **Precision** | 100% | ✅ Perfetto |
| **F1** | 70.6% | ⚠️ Basso |

**Analisi:**
- ❌ **PROBLEMA GRAVE:** Perde 45.5% delle frodi GPT-1.5!
- ✅ Quando rileva, è sempre corretto
- ❌ 91 falsi negativi su 200 (45.5%)

**Conclusione:** GPT-1.5 è molto più difficile da rilevare!

---

## 🌍 EXTERNAL TEST (Dataset WhatsApp Luca)

### Metriche Generali (threshold=0.90)

| Metrica | Internal | External | Delta |
|---------|----------|----------|-------|
| **Accuracy** | 74.7% | 81.8% | +7.1% |
| **Precision** | 79.2% | 62.5% | -16.7% |
| **Recall** | 68.9% | 31.3% | **-37.6%** |
| **F1 Score** | 73.7% | 41.7% | **-32.0%** |
| **ROC-AUC** | 88.2% | 62.6% | **-25.6%** |
| **PR-AUC** | 87.1% | 49.5% | **-37.6%** |

### Confusion Matrix External

```
                Predicted
                NON_FRODE  FRODE
True NON_FRODE      58        3    (FP: 3, 4.9%)
True FRODE          11        5    (FN: 11, 68.8%)
```

### Analisi External Test

**Dataset:** 77 immagini (61 originali, 16 AI)

**Problemi Critici:**
1. ❌ **Recall 31.3%:** Perde 68.8% delle frodi!
2. ❌ **F1 41.7%:** Performance molto bassa
3. ❌ **ROC-AUC 62.6%:** Poco meglio del random (50%)

**Cause Probabili:**
1. **Threshold troppo alto (0.90):**
   - Ottimizzato per internal test
   - Non generalizza su external
   
2. **Domain shift:**
   - WhatsApp compression diversa
   - Qualità immagini diversa
   - Generatori AI diversi?

3. **Dataset sbilanciato:**
   - Solo 16 AI su 77 (20.8%)
   - Pochi esempi per valutazione robusta

---

## 🔍 ANALISI DETTAGLIATA PROBLEMI

### 1. Threshold Troppo Alto

**Evidenza:**
- Internal: Recall 68.9% (threshold 0.90)
- External: Recall 31.3% (threshold 0.90)

**Soluzione:**
- Usa threshold più basso (~0.50-0.60)
- Ottimizza per F1 invece di cost-sensitive
- Valida threshold su external dataset

### 2. GPT-1.5 Difficile da Rilevare

**Evidenza:**
- GPT-1-mini: Recall 82.7%
- GPT-1.5: Recall 54.5% (-28.2%)

**Cause Possibili:**
- GPT-1.5 genera immagini più realistiche
- Meno artefatti visibili
- Modello non ha visto abbastanza esempi GPT-1.5

**Soluzione:**
- Aumenta peso GPT-1.5 in training
- Augmentation più aggressiva su GPT-1.5
- Analizza esempi FN di GPT-1.5

### 3. Generalizzazione Scarsa su External

**Evidenza:**
- Internal F1: 73.7%
- External F1: 41.7% (-32%)

**Cause:**
- Overfitting su internal dataset
- Domain shift (WhatsApp vs training data)
- Threshold non ottimizzato per external

**Soluzione:**
- Augmentation più aggressiva (simula WhatsApp)
- Training con dati WhatsApp
- Threshold adattivo per dominio

---

## 📈 CONFRONTO CON RUN PRECEDENTI

### vs 2026-02-18_convnext_v7_noLeakage

| Metrica | V7 (group_v6) | V8 (domain_aware) | Delta |
|---------|---------------|-------------------|-------|
| F1 | 74.6% | 73.7% | -0.9% |
| Precision | 75.8% | 79.2% | +3.4% |
| Recall | 73.4% | 68.9% | -4.5% |
| PR-AUC | 90.0% | 87.1% | -2.9% |

**Analisi:**
- ⚠️ Domain-aware ha performance leggermente peggiore
- ✅ Precision migliorata (+3.4%)
- ❌ Recall peggiorato (-4.5%)
- ⚠️ Threshold 0.90 troppo conservativo

**Possibili Cause:**
- Threshold diverso (0.70 vs 0.90)
- Dataset split diverso (più robusto ma più difficile)
- Più dati in training (+24%)

---

## 🎯 PUNTI DI FORZA

1. ✅ **ROC-AUC 88.2%:** Modello ha buona capacità discriminativa
2. ✅ **PR-AUC 87.1%:** Buona performance su classe positiva
3. ✅ **Precision 79.2%:** Pochi falsi positivi
4. ✅ **GPT-1-mini detection:** 82.7% recall
5. ✅ **Domain-aware split:** No data leakage, split robusto

---

## ❌ PUNTI CRITICI

1. ❌ **Threshold 0.90 troppo alto:** Recall solo 68.9%
2. ❌ **GPT-1.5 detection:** Solo 54.5% recall
3. ❌ **External test:** F1 41.7%, recall 31.3%
4. ❌ **Generalizzazione:** -32% F1 su external
5. ❌ **127 FN su 408 AI:** Perde 31% delle frodi

---

## 🚀 PROSSIMI STEP RACCOMANDATI

### 1. IMMEDIATO: Ri-valuta con Threshold Ottimale

**Azione:**
```python
# Usa threshold F1-optimal invece di cost-sensitive
python scripts/test_thresholds.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware \
  --metric f1
```

**Risultato Atteso:**
- Recall: 68.9% → ~85%
- Precision: 79.2% → ~75%
- F1: 73.7% → ~80%

### 2. URGENTE: Analizza Falsi Negativi GPT-1.5

**Azione:**
```python
# Analizza esempi difficili GPT-1.5
python scripts/analyze_results.py 2026-02-20_convnext_v8_domaiAware \
  --focus-generator gpt_image_1_5 \
  --show-fn
```

**Obiettivo:**
- Capire perché GPT-1.5 è difficile
- Identificare pattern comuni nei FN
- Decidere se serve fine-tuning specifico

### 3. IMPORTANTE: Test con Threshold Adattivo

**Azione:**
```python
# Testa threshold diversi per internal vs external
python scripts/test_thresholds.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware_external \
  --metric f1 \
  --optimize-for external
```

**Risultato Atteso:**
- External F1: 41.7% → ~60-65%
- External Recall: 31.3% → ~70-75%

### 4. MEDIO TERMINE: Fine-Tuning su GPT-1.5

**Azione:**
```python
# Hard negative mining su GPT-1.5
python scripts/hard_negative_finetune.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware \
  --focus-generator gpt_image_1_5 \
  --epochs 5
```

**Obiettivo:**
- GPT-1.5 recall: 54.5% → 75-80%
- Mantenere GPT-1-mini recall: 82.7%

### 5. LUNGO TERMINE: Augmentation WhatsApp

**Azione:**
- Aggiungi augmentation specifica WhatsApp
- Simula compressione WhatsApp (quality 75-85)
- Training con dati WhatsApp reali

**Obiettivo:**
- External F1: 41.7% → 65-70%
- Generalizzazione migliore

---

## 📊 PRIORITÀ AZIONI

### 🔴 ALTA PRIORITÀ (Fare Subito)

1. **Ri-valuta con threshold F1-optimal** (5 minuti)
   - Impatto: +10-15% F1
   - Effort: Minimo

2. **Analizza FN GPT-1.5** (30 minuti)
   - Impatto: Capire problema critico
   - Effort: Basso

### 🟡 MEDIA PRIORITÀ (Questa Settimana)

3. **Test threshold adattivo external** (1 ora)
   - Impatto: +20-25% F1 external
   - Effort: Medio

4. **Fine-tuning GPT-1.5** (3-4 ore training)
   - Impatto: +20-25% recall GPT-1.5
   - Effort: Medio

### 🟢 BASSA PRIORITÀ (Prossimo Sprint)

5. **Augmentation WhatsApp** (1-2 giorni)
   - Impatto: +15-20% F1 external
   - Effort: Alto

---

## 🎓 LEZIONI APPRESE

### ✅ Cosa Ha Funzionato

1. **Domain-aware split:** No data leakage, split robusto
2. **ConvNeXt-Tiny:** Buona capacità discriminativa (ROC-AUC 88%)
3. **Weighted focal loss:** Gestisce bene class imbalance
4. **GPT-1-mini detection:** 82.7% recall

### ❌ Cosa Non Ha Funzionato

1. **Threshold cost-sensitive:** Troppo conservativo (0.90)
2. **GPT-1.5 detection:** Solo 54.5% recall
3. **External generalization:** F1 41.7% (-32% vs internal)
4. **Threshold fisso:** Non si adatta a domini diversi

### 🔧 Cosa Migliorare

1. **Threshold selection:** Usa F1-optimal o adattivo
2. **GPT-1.5 training:** Più esempi o hard negative mining
3. **Augmentation:** Simula meglio WhatsApp e altri domini
4. **Validation:** Testa sempre su external dataset

---

## 📝 CONCLUSIONI

### Stato Attuale

**Modello:** ⚠️ Funzionante ma con problemi critici

**Punti Forti:**
- ✅ Buona capacità discriminativa (ROC-AUC 88%)
- ✅ Split robusto (domain-aware)
- ✅ Rileva bene GPT-1-mini (82.7%)

**Punti Critici:**
- ❌ Threshold troppo alto (recall 68.9%)
- ❌ GPT-1.5 difficile (recall 54.5%)
- ❌ Generalizzazione scarsa (external F1 41.7%)

### Raccomandazione

**AZIONE IMMEDIATA:**
1. Ri-valuta con threshold F1-optimal (~0.50-0.60)
2. Analizza FN GPT-1.5 per capire il problema
3. Testa threshold adattivo su external

**RISULTATO ATTESO:**
- Internal F1: 73.7% → ~80%
- External F1: 41.7% → ~60-65%
- GPT-1.5 recall: 54.5% → ~75-80% (con fine-tuning)

**TEMPO STIMATO:** 1-2 giorni per azioni immediate, 1 settimana per fine-tuning

---

## 📞 Prossimi Step Discussione

1. **Vuoi che ri-valuti con threshold F1-optimal?** (5 minuti)
2. **Vuoi analizzare i FN GPT-1.5?** (30 minuti)
3. **Vuoi fare fine-tuning su GPT-1.5?** (3-4 ore)
4. **Vuoi testare threshold adattivo?** (1 ora)

Dimmi quale azione vuoi fare prima! 🚀
