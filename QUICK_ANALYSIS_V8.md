# ⚡ Quick Analysis: ConvNeXt V8 Domain-Aware

## 📊 RISULTATI IN SINTESI

### Internal Test (threshold=0.90)
```
✅ ROC-AUC: 88.2%  (Ottimo)
✅ PR-AUC:  87.1%  (Ottimo)
⚠️ F1:      73.7%  (Moderato)
⚠️ Recall:  68.9%  (Basso - perde 31% frodi!)
✅ Precision: 79.2% (Buono)
```

### External Test (threshold=0.90)
```
❌ F1:      41.7%  (CRITICO - 32% peggio!)
❌ Recall:  31.3%  (CRITICO - perde 69% frodi!)
⚠️ Precision: 62.5% (Moderato)
❌ ROC-AUC: 62.6%  (Poco meglio del random)
```

---

## 🤖 PERFORMANCE PER GENERATOR

```
GPT-1-mini:  Recall 82.7% ✅ (Buono)
GPT-1.5:     Recall 54.5% ❌ (CRITICO - perde 45%!)
```

**Problema:** GPT-1.5 è MOLTO più difficile da rilevare!

---

## ❌ PROBLEMI CRITICI

### 1. Threshold Troppo Alto (0.90)
- Ottimizzato per minimizzare FP
- Risultato: Perde troppe frodi (31% internal, 69% external)
- **Fix:** Usa threshold ~0.50-0.60 (F1-optimal)

### 2. GPT-1.5 Detection Scarsa
- Recall solo 54.5% (vs 82.7% GPT-1-mini)
- 91 FN su 200 immagini GPT-1.5
- **Fix:** Fine-tuning specifico su GPT-1.5

### 3. Generalizzazione Scarsa
- Internal F1: 73.7%
- External F1: 41.7% (-32%!)
- **Fix:** Threshold adattivo + augmentation WhatsApp

---

## 🚀 AZIONI IMMEDIATE (Priorità)

### 🔴 1. Ri-valuta con Threshold F1-Optimal (5 min)
```bash
# Cambia threshold da 0.90 a ~0.55
python scripts/test_thresholds.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware \
  --metric f1
```
**Impatto Atteso:**
- Internal F1: 73.7% → ~80% (+6%)
- Internal Recall: 68.9% → ~85% (+16%)

### 🔴 2. Analizza FN GPT-1.5 (30 min)
```bash
# Capire perché GPT-1.5 è difficile
python scripts/analyze_results.py 2026-02-20_convnext_v8_domaiAware \
  --focus-generator gpt_image_1_5
```
**Obiettivo:** Identificare pattern comuni nei falsi negativi

### 🟡 3. Test Threshold External (1 ora)
```bash
# Ottimizza threshold per external dataset
python scripts/test_thresholds.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware_external \
  --metric f1
```
**Impatto Atteso:**
- External F1: 41.7% → ~60-65% (+20%)
- External Recall: 31.3% → ~70-75% (+40%)

### 🟡 4. Fine-Tuning GPT-1.5 (3-4 ore)
```bash
# Hard negative mining su GPT-1.5
python scripts/hard_negative_finetune.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware \
  --focus-generator gpt_image_1_5
```
**Impatto Atteso:**
- GPT-1.5 Recall: 54.5% → ~75-80% (+20%)

---

## 📈 CONFRONTO CON V7

| Metrica | V7 (group_v6) | V8 (domain_aware) | Delta |
|---------|---------------|-------------------|-------|
| F1 | 74.6% | 73.7% | -0.9% |
| Recall | 73.4% | 68.9% | -4.5% ❌ |
| Precision | 75.8% | 79.2% | +3.4% ✅ |

**Causa:** Threshold diverso (0.70 vs 0.90)

---

## 🎯 RACCOMANDAZIONE

### Cosa Fare ORA

1. **Ri-valuta con threshold 0.55** (5 minuti)
   - Risolve problema recall basso
   - F1 da 73.7% a ~80%

2. **Analizza FN GPT-1.5** (30 minuti)
   - Capire problema critico
   - Decidere se serve fine-tuning

3. **Testa threshold external** (1 ora)
   - Migliora external da 41.7% a ~60%
   - Threshold adattivo per dominio

### Risultato Atteso

**Dopo azioni immediate:**
- Internal F1: 73.7% → **~80%** (+6%)
- External F1: 41.7% → **~60-65%** (+20%)
- GPT-1.5 Recall: 54.5% → **~75-80%** (+20%, con fine-tuning)

**Tempo:** 1-2 giorni

---

## 💡 LEZIONI CHIAVE

1. ✅ **Domain-aware split funziona** (no data leakage)
2. ❌ **Threshold cost-sensitive troppo conservativo** (usa F1-optimal)
3. ❌ **GPT-1.5 è molto più difficile** (serve fine-tuning)
4. ❌ **Generalizzazione scarsa** (serve augmentation WhatsApp)

---

## 📞 PROSSIMO STEP?

**Dimmi quale azione vuoi fare:**

A) 🔴 Ri-valuta con threshold F1-optimal (5 min) - **RACCOMANDATO**
B) 🔴 Analizza FN GPT-1.5 (30 min)
C) 🟡 Test threshold external (1 ora)
D) 🟡 Fine-tuning GPT-1.5 (3-4 ore)
E) 🟢 Altro (dimmi tu)

**Raccomandazione:** Inizia con A, poi B, poi C. 🚀
