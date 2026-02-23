# Prossimi Passi Operativi - v8.1

## 🎉 Ottimi Risultati!

✅ **No data leakage** - Split identici (100% overlap)  
✅ **Performance buona** - F1 84.8%, Precision 85.7%, Recall 84.0%  
✅ **Hard negative efficace** - Miglioramento +7.5% F1  

## 📋 Azioni Immediate

### 1. Espandi External Test (PRIORITÀ MASSIMA) 🔴

**Problema:** Solo 3 foto WhatsApp - non statisticamente significativo

**Cosa fare:**
1. Raccogli 50-100 foto WhatsApp reali da Luca o altri utenti
2. Mix di:
   - Originali (buone): ~30-50 foto
   - Modificate (frodi): ~20-50 foto
3. Organizza in cartelle:
   ```
   whatsapp_test_large/
     originali/
     modificate/
   ```
4. Testa con:
   ```python
   python -m src.eval \
     --config configs/convnext_v8.yaml \
     --run_name 2026-02-23_v8.1_external_large \
     --checkpoint_path checkpoints/.../best.pt \
     --threshold 0.75 \
     --external_test_dataset /path/to/whatsapp_test_large
   ```

**Target:**
- F1 ≥75% (tolleranza -5% vs internal)
- Precision ≥75%
- Recall ≥75%

**Se raggiunti → READY FOR PRODUCTION** ✅

### 2. Analizza False Negatives 🟡

**Problema:** FN aumentati da 4 a 8 con hard negative

**Cosa fare:**
```python
# Vedi quali foto sono FN
python scripts/analyze_results.py \
    2026-02-23_convnext_v8.1_domainAware_hard_negative \
    --show-fn

# Identifica pattern:
# - Quali generatori? (GPT-1.5 vs GPT-1-mini)
# - Quali difetti? (crudo, marcio, bruciato)
# - Quali food categories?
```

**Obiettivo:** Capire se serve ulteriore fine-tuning

### 3. Test GPT-1.5 Performance 🟡

**Cosa fare:**
```python
python scripts/analyze_gpt15_threshold.py \
    --run 2026-02-23_convnext_v8.1_domainAware_hard_negative \
    --threshold 0.75
```

**Target:**
- Recall GPT-1.5: ≥85%
- Precision GPT-1.5: ≥90%

**Se sotto target:** Considera ulteriore hard negative su GPT-1.5

### 4. Deploy in Staging 🟢

**Prerequisiti:**
- ✅ No data leakage
- ✅ F1 ≥80%
- ⏳ External test con dataset grande

**Quando fare:**
- Dopo aver completato external test grande
- Se performance external ≥75%

**Come fare:**
1. Usa checkpoint: `2026-02-23_convnext_v8.1_domainAware_hard_negative/best.pt`
2. Threshold: 0.75
3. Monitora performance su foto reali

## 📊 Metriche Attuali

### Internal Test (221 foto)
```
Original:
  Precision: 66.7%
  Recall:    92.0%
  F1:        77.3%

Hard Negative:
  Precision: 85.7% (+19.0%)
  Recall:    84.0% (-8.0%)
  F1:        84.8% (+7.5%)
```

### External Test (3 foto) ⚠️ TROPPO PICCOLO
```
Precision: 100%
Recall:    100%
F1:        100%
```

## 🎯 Raccomandazioni

### Per Produzione

**Modello:** `2026-02-23_convnext_v8.1_domainAware_hard_negative`  
**Threshold:** 0.75  
**Performance attesa:** F1 ~85%, Precision ~86%, Recall ~84%

### Limitazioni Note

1. External test limitato (solo 3 foto)
2. 8 false negatives (1.6% delle frodi)
3. Performance varia con threshold

## ✅ Checklist

- [x] Training completato
- [x] Hard negative completato
- [x] No data leakage verificato
- [x] Performance internal ≥80%
- [ ] External test con 50-100 foto
- [ ] Analisi false negatives
- [ ] Test GPT-1.5 performance
- [ ] Deploy in staging
- [ ] Monitoraggio produzione

## 🚀 Timeline Suggerita

**Questa settimana:**
1. Raccogli 50-100 foto WhatsApp
2. Run external test grande
3. Analizza false negatives

**Prossima settimana:**
1. Se external test OK → Deploy staging
2. Monitora performance
3. Se tutto OK → Production

## 📞 Domande?

Se hai dubbi o serve aiuto:
1. Controlla `ANALISI_RISULTATI_V8.1.md` per dettagli
2. Esegui gli script di analisi
3. Chiedi se serve chiarimento

**Ottimo lavoro finora!** 🎉
