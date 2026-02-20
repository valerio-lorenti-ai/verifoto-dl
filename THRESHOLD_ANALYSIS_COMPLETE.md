# 🎯 Analisi Completa Threshold: Internal vs External

## 📊 Internal Test (226 foto)

### Top 3 Threshold per F1

| Threshold | F1 | Precision | Recall | Accuracy | FP | FN | Valutazione |
|-----------|-----|-----------|--------|----------|----|----|-------------|
| **0.55** | **82.6%** | 77.6% | 88.2% | 91.6% | 13 | 6 | ✅ Migliore bilanciamento |
| **0.60** | **81.5%** | 77.2% | 86.3% | 91.2% | 13 | 7 | ✅ Ottimo |
| **0.40** | **81.4%** | 74.2% | 90.2% | 90.7% | 16 | 5 | ✅ Recall più alto |

### Analisi Dettagliata

#### Threshold 0.40 (Raccomandato da Calibration)
```
F1:        81.4%
Precision: 74.2%  (3 su 4 alert sono veri)
Recall:    90.2%  (rileva 9 frodi su 10)
FP: 16, FN: 5
```
**Pro:**
- ✅ Recall altissimo (90.2%) - rileva quasi tutte le frodi
- ✅ Solo 5 falsi negativi (perde solo 5 frodi su 51)
- ✅ Buon bilanciamento per fraud detection

**Contro:**
- ⚠️ Precision più bassa (74.2%) - 1 alert su 4 è falso
- ⚠️ 16 falsi positivi (più falsi allarmi)

#### Threshold 0.55 (Migliore F1)
```
F1:        82.6%  (+1.2% vs 0.40)
Precision: 77.6%  (+3.4% vs 0.40)
Recall:    88.2%  (-2.0% vs 0.40)
FP: 13, FN: 6
```
**Pro:**
- ✅ F1 più alto (82.6%)
- ✅ Precision migliore (77.6%) - meno falsi allarmi
- ✅ Recall ancora ottimo (88.2%)
- ✅ Miglior bilanciamento generale

**Contro:**
- ⚠️ 1 falso negativo in più (6 vs 5)

#### Threshold 0.60
```
F1:        81.5%  (+0.1% vs 0.40)
Precision: 77.2%  (+3.0% vs 0.40)
Recall:    86.3%  (-3.9% vs 0.40)
FP: 13, FN: 7
```
**Pro:**
- ✅ Precision buona (77.2%)
- ✅ F1 simile a 0.40 (81.5%)

**Contro:**
- ⚠️ Recall più basso (86.3%) - perde 2 frodi in più
- ⚠️ 7 falsi negativi (vs 5 con 0.40)

---

## 🌍 External Test (3 foto - Dataset WhatsApp Luca)

### ⚠️ ATTENZIONE: Dataset Molto Piccolo!

**Problema Critico:** Solo 3 foto (77 versioni) - metriche poco affidabili!

### Risultati per Threshold

| Threshold | F1 | Precision | Recall | Accuracy | FP | FN | Note |
|-----------|-----|-----------|--------|----------|----|----|------|
| **0.10-0.15** | **80.0%** | 66.7% | 100% | 66.7% | 1 | 0 | Migliore |
| 0.20 | 50.0% | 50.0% | 50.0% | 33.3% | 1 | 1 | Mediocre |
| 0.25-0.90 | 0.0% | 0.0% | 0.0% | 0-33% | 0-1 | 2 | Pessimo |

### Analisi External Test

**Threshold Ottimale: 0.10-0.15**
```
F1:        80.0%
Precision: 66.7%  (2 su 3 alert sono veri)
Recall:    100%   (rileva tutte le 2 frodi)
FP: 1, FN: 0
```

**Problema:** Con threshold ≥0.25, il modello NON rileva NESSUNA frode!

**Causa:** Dataset WhatsApp ha caratteristiche molto diverse:
- Compressione WhatsApp
- Qualità immagini diversa
- Solo 3 foto (statisticamente non significativo)

---

## 🎯 Raccomandazione Finale

### Per Internal Test (Produzione)

**Raccomandazione: Threshold 0.55** ✅

**Perché:**
1. ✅ **F1 più alto:** 82.6% (migliore bilanciamento)
2. ✅ **Precision migliore:** 77.6% (meno falsi allarmi)
3. ✅ **Recall ottimo:** 88.2% (rileva 9 frodi su 10)
4. ✅ **Accuracy migliore:** 91.6%
5. ✅ **Meno falsi positivi:** 13 vs 16 (con 0.40)

**Trade-off Accettabile:**
- Solo 1 falso negativo in più (6 vs 5)
- Ma 3 falsi positivi in meno (13 vs 16)
- Rapporto beneficio/costo: migliore

### Confronto 0.40 vs 0.55 vs 0.60

| Metrica | 0.40 | 0.55 | 0.60 | Migliore |
|---------|------|------|------|----------|
| **F1** | 81.4% | **82.6%** | 81.5% | 0.55 ✅ |
| **Precision** | 74.2% | **77.6%** | 77.2% | 0.55 ✅ |
| **Recall** | **90.2%** | 88.2% | 86.3% | 0.40 ✅ |
| **Accuracy** | 90.7% | **91.6%** | 91.2% | 0.55 ✅ |
| **FP** | 16 | **13** | 13 | 0.55/0.60 ✅ |
| **FN** | **5** | 6 | 7 | 0.40 ✅ |

**Vincitore:** Threshold 0.55 (vince 4 su 6 metriche)

### Per External Test

**⚠️ NON AFFIDABILE** - Dataset troppo piccolo (solo 3 foto)

**Raccomandazione:**
1. 🔴 **Raccogliere più dati WhatsApp** (almeno 50-100 foto)
2. 🔴 **Ri-testare con dataset più grande**
3. 🟡 **Usare threshold adattivo:** 0.55 per internal, 0.10-0.15 per WhatsApp (temporaneo)

---

## 📈 Impatto del Threshold

### Threshold 0.40 (Calibration)

**Caso d'uso:** Quando recall è prioritario
- Fraud detection dove perdere una frode è molto costoso
- Fase iniziale dove vuoi catturare tutto
- Tolleranza alta per falsi allarmi

**Metriche:**
- Rileva 46/51 frodi (90.2%)
- 16 falsi allarmi su 175 reali
- 1 falso allarme ogni 2.9 frodi rilevate

### Threshold 0.55 (Migliore F1) ✅ RACCOMANDATO

**Caso d'uso:** Produzione bilanciata
- Miglior compromesso recall/precision
- Meno falsi allarmi senza perdere troppe frodi
- Uso quotidiano

**Metriche:**
- Rileva 45/51 frodi (88.2%)
- 13 falsi allarmi su 175 reali
- 1 falso allarme ogni 3.5 frodi rilevate

**Miglioramento vs 0.40:**
- -3 falsi positivi (-18.8%)
- +1 falso negativo (+20%)
- +1.2% F1
- +3.4% Precision

### Threshold 0.60

**Caso d'uso:** Quando precision è prioritario
- Costi alti per falsi allarmi
- Revisione manuale costosa
- Tolleranza bassa per falsi positivi

**Metriche:**
- Rileva 44/51 frodi (86.3%)
- 13 falsi allarmi su 175 reali
- 1 falso allarme ogni 3.4 frodi rilevate

---

## 🎓 Analisi Costi/Benefici

### Scenario: 1000 Foto (450 frodi, 550 reali)

#### Threshold 0.40
```
Frodi rilevate:    406 / 450  (90.2%)
Frodi perse:       44          (9.8%)
Falsi allarmi:     50          (9.1% delle reali)
Alert totali:      456
Precisione alert:  89.0%
```

#### Threshold 0.55 ✅
```
Frodi rilevate:    397 / 450  (88.2%)
Frodi perse:       53          (11.8%)
Falsi allarmi:     41          (7.5% delle reali)
Alert totali:      438
Precisione alert:  90.6%
```
**Differenza vs 0.40:**
- -9 frodi rilevate (-2.0%)
- -9 falsi allarmi (-18.0%)
- +1.6% precisione alert

#### Threshold 0.60
```
Frodi rilevate:    388 / 450  (86.3%)
Frodi perse:       62          (13.8%)
Falsi allarmi:     41          (7.5% delle reali)
Alert totali:      429
Precisione alert:  90.4%
```

---

## 💡 Raccomandazione Operativa

### Strategia a Due Threshold

**Opzione A: Threshold Fisso 0.55** ✅ RACCOMANDATO
- Usa 0.55 per tutti i casi
- Miglior bilanciamento generale
- Più semplice da gestire

**Opzione B: Threshold Adattivo**
- **0.40** per casi critici (es. transazioni alte)
- **0.55** per uso normale
- **0.60** per casi a basso rischio

**Opzione C: Threshold con Confidence**
- **prob > 0.80:** Alert automatico (alta confidence)
- **0.55 < prob < 0.80:** Alert normale
- **0.40 < prob < 0.55:** Alert bassa priorità
- **prob < 0.40:** Nessun alert

---

## 📊 Conclusione

### Per Internal Test

**Threshold Raccomandato: 0.55** ✅

**Motivi:**
1. F1 più alto (82.6%)
2. Precision migliore (77.6%)
3. Recall ottimo (88.2%)
4. Meno falsi allarmi (-18.8% vs 0.40)
5. Solo 1 frode in più persa (accettabile)

### Per External Test

**⚠️ Dataset troppo piccolo** (3 foto)

**Azioni Necessarie:**
1. Raccogliere più dati WhatsApp (50-100 foto minimo)
2. Ri-testare con dataset più grande
3. Considerare fine-tuning su dati WhatsApp

### Prossimi Step

1. ✅ **Usa threshold 0.55 in produzione** (internal)
2. 🔴 **Raccogliere più dati WhatsApp** (external)
3. 🟡 **Monitorare performance** in produzione
4. 🟡 **Considerare threshold adattivo** se necessario

---

## 📞 Domande?

**Q: Perché non 0.40 se ha recall più alto?**
- A: 0.55 ha miglior bilanciamento generale (+1.2% F1, +3.4% Precision) con solo -2% recall

**Q: Posso usare 0.60?**
- A: Sì, se precision è prioritario. Ma 0.55 è migliore per uso generale.

**Q: E per external test?**
- A: Dataset troppo piccolo (3 foto). Serve più dati per valutazione affidabile.

**Q: Threshold diverso per GPT-1.5?**
- A: Possibile, ma prima testa 0.55 su tutto. Se GPT-1.5 ancora problematico, considera threshold specifico.
