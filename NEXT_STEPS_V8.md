# 🚀 Prossimi Passi - ConvNeXt V8

## ✅ Completato

1. ✅ **Threshold ottimizzato:** Aggiornato a 0.55 in `configs/convnext_v8.yaml`
2. ✅ **Config rinominato:** `convnext_v7_improved.yaml` → `convnext_v8.yaml`
3. ✅ **Documentazione:** Scelta del threshold documentata nel config
4. ✅ **Notebook aggiornato:** External test ora include photo-level analysis
5. ✅ **File generati:** External test ha tutti i file necessari

---

## 🎯 Prossimi Passi Raccomandati

### 🔴 ALTA PRIORITÀ (Fare Subito)

#### 1. Test del Nuovo Config (30 min)

**Obiettivo:** Verificare che il config V8 funzioni correttamente

**Azione:**
```bash
# Test rapido con quick_test.yaml (se esiste)
python -m src.train_v7 \
  --config configs/convnext_v8.yaml \
  --run_name test_v8_config \
  --checkpoint_dir checkpoints
```

**Verifica:**
- Training parte correttamente
- Usa threshold 0.55
- Usa domain-aware split
- Salva checkpoint

**Tempo:** 5-10 minuti (solo per verificare che funzioni)

---

#### 2. Analizza GPT-1.5 con Threshold 0.55 (30 min)

**Obiettivo:** Capire se threshold 0.55 risolve il problema GPT-1.5

**Azione:**
```bash
python scripts/analyze_results.py 2026-02-20_convnext_v8_domaiAware \
  --focus-generator gpt_image_1_5 \
  --threshold 0.55 \
  --show-fn
```

**Cosa Cercare:**
- Recall GPT-1.5 con threshold 0.55 (atteso: ~85% vs 54.5% con 0.90)
- Quali GPT-1.5 sono ancora difficili
- Pattern comuni nei falsi negativi

**Decisione:**
- Se recall GPT-1.5 > 80%: ✅ Problema risolto, vai in produzione
- Se recall GPT-1.5 < 80%: ⚠️ Considera fine-tuning (Step 5)

---

#### 3. Raccogliere Più Dati WhatsApp (Critico!)

**Obiettivo:** External test ha solo 3 foto - troppo poco!

**Azione:**
1. Chiedi a Luca di inviare più foto WhatsApp
2. Target: Almeno 50-100 foto (mix reali + AI)
3. Organizza in cartelle:
   ```
   whatsapp_dataset/
   ├── originali/
   │   ├── foto1.jpg
   │   ├── foto2.jpg
   │   └── ...
   └── modificate/
       ├── ai_foto1.jpg
       ├── ai_foto2.jpg
       └── ...
   ```

**Perché è Importante:**
- 3 foto non sono statisticamente significative
- Non puoi validare generalizzazione su WhatsApp
- Rischi di avere sorprese in produzione

**Tempo:** Dipende da Luca (raccolta dati)

---

### 🟡 MEDIA PRIORITÀ (Questa Settimana)

#### 4. Training Completo con Config V8 (3-4 ore)

**Obiettivo:** Training completo con threshold ottimizzato

**Azione:**
```bash
# Su Colab con A100
python -m src.train_v7 \
  --config configs/convnext_v8.yaml \
  --run_name 2026-02-21_convnext_v8_optimized \
  --checkpoint_dir /content/drive/MyDrive/verifoto_checkpoints
```

**Cosa Aspettarsi:**
- Training usa domain-aware split
- Trova threshold durante training (poi usa 0.55)
- Photo-level analysis usa threshold 0.55
- Metriche migliori rispetto a V7

**Quando Farlo:**
- Dopo aver verificato che config funziona (Step 1)
- Dopo aver analizzato GPT-1.5 (Step 2)
- Quando hai tempo per training completo

---

#### 5. Fine-Tuning GPT-1.5 (Solo se Necessario)

**Quando:** Solo se Step 2 mostra recall GPT-1.5 < 80% con threshold 0.55

**Azione:**
```bash
python scripts/hard_negative_finetune.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware \
  --focus-generator gpt_image_1_5 \
  --threshold 0.55 \
  --epochs 5 \
  --lr 1e-5
```

**Obiettivo:**
- GPT-1.5 recall: ~85% → ~92-95%
- Mantenere GPT-1-mini recall: ~95%

**Tempo:** 3-4 ore training

---

#### 6. Test su Dataset WhatsApp Più Grande

**Quando:** Dopo aver raccolto più dati (Step 3)

**Azione:**
```bash
python -m src.eval \
  --config configs/convnext_v8.yaml \
  --run_name 2026-02-21_convnext_v8_whatsapp_large \
  --checkpoint_path checkpoints/2026-02-21_convnext_v8_optimized/best.pt \
  --threshold 0.55 \
  --external_test_dataset /path/to/whatsapp_dataset_large
```

**Cosa Cercare:**
- F1 su WhatsApp con threshold 0.55
- Confronto con internal test
- Se serve threshold diverso per WhatsApp

**Decisione:**
- Se F1 > 70%: ✅ Generalizza bene
- Se F1 < 70%: ⚠️ Considera threshold adattivo o fine-tuning

---

### 🟢 BASSA PRIORITÀ (Prossimo Sprint)

#### 7. Threshold Adattivo per Domini Diversi

**Obiettivo:** Threshold diverso per internal vs WhatsApp

**Implementazione:**
```python
def get_optimal_threshold(domain):
    if domain == "internal":
        return 0.55
    elif domain == "whatsapp":
        return 0.40  # O quello trovato in Step 6
    else:
        return 0.55  # Default
```

**Quando:** Dopo Step 6, se WhatsApp ha threshold ottimale diverso

---

#### 8. Augmentation WhatsApp-Specific

**Obiettivo:** Simulare compressione WhatsApp durante training

**Implementazione:**
```python
# In src/utils/augmentations.py
def whatsapp_augmentation(image):
    # Simula compressione WhatsApp
    image = jpeg_compression(image, quality=75-85)
    # Simula resize WhatsApp
    image = resize_with_aspect_ratio(image, max_size=1600)
    return image
```

**Quando:** Se Step 6 mostra scarsa generalizzazione su WhatsApp

---

#### 9. Monitoraggio Produzione

**Obiettivo:** Tracciare performance in produzione

**Metriche da Monitorare:**
- Distribuzione probabilità
- Numero alert per giorno
- Feedback utenti (veri positivi vs falsi positivi)
- Drift del modello nel tempo

**Tool:** MLflow, Weights & Biases, o custom dashboard

---

## 📊 Roadmap Completa

### Settimana 1 (Ora)
- ✅ Config V8 creato
- 🔴 Test config V8 (Step 1)
- 🔴 Analizza GPT-1.5 con 0.55 (Step 2)
- 🔴 Richiedi più dati WhatsApp (Step 3)

### Settimana 2
- 🟡 Training completo V8 (Step 4)
- 🟡 Fine-tuning GPT-1.5 se necessario (Step 5)
- 🟡 Test su WhatsApp grande (Step 6)

### Settimana 3-4
- 🟢 Threshold adattivo (Step 7)
- 🟢 Augmentation WhatsApp (Step 8)
- 🟢 Setup monitoraggio (Step 9)

---

## 🎯 Decisioni Chiave

### Decisione 1: Fine-Tuning GPT-1.5?

**Dipende da:** Step 2 (analisi GPT-1.5 con threshold 0.55)

**Se recall GPT-1.5 > 80%:**
- ✅ NO fine-tuning necessario
- Vai direttamente in produzione con threshold 0.55

**Se recall GPT-1.5 < 80%:**
- ⚠️ Considera fine-tuning (Step 5)
- O accetta recall più basso

### Decisione 2: Threshold Adattivo?

**Dipende da:** Step 6 (test WhatsApp grande)

**Se threshold ottimale WhatsApp ≈ 0.55:**
- ✅ Usa threshold fisso 0.55 per tutto
- Più semplice da gestire

**Se threshold ottimale WhatsApp molto diverso (es. 0.40):**
- ⚠️ Implementa threshold adattivo (Step 7)
- Più complesso ma migliori risultati

### Decisione 3: Augmentation WhatsApp?

**Dipende da:** Step 6 (generalizzazione WhatsApp)

**Se F1 WhatsApp > 70%:**
- ✅ NO augmentation necessaria
- Modello generalizza bene

**Se F1 WhatsApp < 70%:**
- ⚠️ Implementa augmentation WhatsApp (Step 8)
- Ri-training con augmentation

---

## 📝 Checklist Immediata

Prima del prossimo training:

- [ ] Verifica config V8 funziona (Step 1)
- [ ] Analizza GPT-1.5 con threshold 0.55 (Step 2)
- [ ] Richiedi più dati WhatsApp a Luca (Step 3)
- [ ] Aggiorna notebook per usare config V8
- [ ] Documenta risultati analisi GPT-1.5

Prima di andare in produzione:

- [ ] Training completo V8 completato (Step 4)
- [ ] GPT-1.5 recall > 80% (con 0.55)
- [ ] Test su WhatsApp grande completato (Step 6)
- [ ] Threshold finale deciso (0.55 o adattivo)
- [ ] Documentazione completa

---

## 🚀 Quick Start

**Per iniziare subito:**

```bash
# 1. Test config V8
python -m src.train_v7 \
  --config configs/convnext_v8.yaml \
  --run_name test_v8_config \
  --checkpoint_dir checkpoints

# 2. Analizza GPT-1.5
python scripts/analyze_results.py 2026-02-20_convnext_v8_domaiAware \
  --focus-generator gpt_image_1_5 \
  --threshold 0.55 \
  --show-fn

# 3. Quando pronto, training completo
python -m src.train_v7 \
  --config configs/convnext_v8.yaml \
  --run_name 2026-02-21_convnext_v8_optimized \
  --checkpoint_dir /content/drive/MyDrive/verifoto_checkpoints
```

---

## 📞 Domande?

**Q: Devo rifare il training subito?**
- A: No, prima testa il config (Step 1) e analizza GPT-1.5 (Step 2)

**Q: Quando posso andare in produzione?**
- A: Dopo training V8 completo (Step 4) e test WhatsApp (Step 6)

**Q: Cosa faccio con il run V7?**
- A: Tienilo come backup. V8 è un miglioramento incrementale.

**Q: Devo cambiare qualcosa nel codice?**
- A: No, basta usare il nuovo config V8. Il codice supporta già threshold 0.55.

**Q: E se GPT-1.5 è ancora problematico?**
- A: Fine-tuning (Step 5) o accetta recall più basso. Dipende dai requisiti.

---

## 🎉 Conclusione

Hai ora:
- ✅ Config V8 ottimizzato con threshold 0.55
- ✅ Documentazione completa della scelta
- ✅ Roadmap chiara per i prossimi passi
- ✅ Decisioni chiave identificate

**Prossima azione:** Esegui Step 1 (test config) e Step 2 (analisi GPT-1.5) per decidere se serve fine-tuning.

Buon lavoro! 🚀
