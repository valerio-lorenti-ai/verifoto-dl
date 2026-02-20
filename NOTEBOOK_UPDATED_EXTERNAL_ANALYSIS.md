# ✅ Notebook Aggiornato: External Test Analysis

## 🎯 Cosa è Stato Fatto

Ho aggiornato il notebook `scripts/notebooks/verifoto_dl.ipynb` per includere l'analisi photo-level e calibration anche per l'external test.

---

## 📝 Modifiche Apportate

### Nuova Cella Aggiunta

**Posizione:** Dopo la cella "Test con dataset esterno" e prima della cella "PUSH RESULTS TO GITHUB"

**Funzionalità:**

1. **Photo-Level Analysis per External Test:**
   - Esegue `scripts/analyze_by_photo.py` sul dataset external
   - Trova threshold ottimale per external dataset
   - Calcola metriche photo-level aggregate
   - Genera file: `threshold_sweep.csv`, `photo_level_metrics.json`, `chosen_threshold.json`

2. **Calibration per External Test:**
   - Copia i file di calibration dal training run
   - Nota che la calibration è già applicata (modello usa T dal training)
   - Genera file: `calibration_T.json`, `calibration_report.json`

3. **Output Visualizzato:**
   - Threshold raccomandato per external
   - Confronto metriche: threshold originale vs ottimale
   - Miglioramenti attesi (F1, Recall, Precision)

---

## 📊 File Generati per External Test

Dopo l'aggiornamento, l'external test avrà questi file aggiuntivi:

```
outputs/runs/{EXPERIMENT_NAME}_external/
├── metrics.json                    ✅ (già presente)
├── predictions.csv                 ✅ (già presente)
├── threshold_sweep.csv             ✨ NUOVO
├── photo_level_metrics.json        ✨ NUOVO
├── photo_level_predictions.csv     ✨ NUOVO
├── chosen_threshold.json           ✨ NUOVO
├── photo_hard_fp.csv               ✨ NUOVO
├── photo_hard_fn.csv               ✨ NUOVO
├── threshold_sweep.png             ✨ NUOVO
├── calibration_T.json              ✨ NUOVO (copiato)
└── calibration_report.json         ✨ NUOVO (copiato)
```

---

## 🚀 Come Usare il Notebook Aggiornato

### Per Run Future

Il notebook ora esegue automaticamente:

1. Training interno → Photo-level analysis + Calibration
2. External test → Photo-level analysis + Calibration
3. Backup di tutti i file su Drive
4. Push di tutti i file su GitHub

**Non serve fare nulla di diverso!** Basta eseguire il notebook normalmente.

---

## 🔧 Per Generare i File Mancanti (Run Corrente)

Se vuoi generare i file per il run corrente `2026-02-20_convnext_v8_domaiAware_external`:

### Opzione 1: Esegui Localmente

```bash
# Photo-level analysis
python scripts/analyze_by_photo.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware_external \
  --min-recall 0.90

# Copia calibration files
copy outputs\runs\2026-02-20_convnext_v8_domaiAware\calibration_T.json outputs\runs\2026-02-20_convnext_v8_domaiAware_external\
copy outputs\runs\2026-02-20_convnext_v8_domaiAware\calibration_report.json outputs\runs\2026-02-20_convnext_v8_domaiAware_external\
```

### Opzione 2: Esegui su Colab

Aggiungi e esegui questa cella nel notebook Colab:

```python
# Genera file mancanti per external test corrente
EXTERNAL_RUN_NAME = "2026-02-20_convnext_v8_domaiAware_external"
EXTERNAL_OUTPUT_DIR = f"outputs/runs/{EXTERNAL_RUN_NAME}"

!python scripts/analyze_by_photo.py \
    --run {EXTERNAL_OUTPUT_DIR} \
    --min-recall 0.90

# Copia calibration files
import shutil
shutil.copy("outputs/runs/2026-02-20_convnext_v8_domaiAware/calibration_T.json", 
            f"{EXTERNAL_OUTPUT_DIR}/calibration_T.json")
shutil.copy("outputs/runs/2026-02-20_convnext_v8_domaiAware/calibration_report.json", 
            f"{EXTERNAL_OUTPUT_DIR}/calibration_report.json")

print("✓ File generati per external test")
```

---

## 📈 Output Atteso

Quando esegui il notebook (o generi i file manualmente), vedrai:

```
================================================================================
EXTERNAL TEST: PHOTO-LEVEL ANALYSIS
================================================================================
[... output di analyze_by_photo.py ...]

================================================================================
EXTERNAL TEST: THRESHOLD RECOMMENDATION
================================================================================
🎯 Recommended threshold: 0.XXX
   F1-optimal threshold: 0.XXX
   Precision-optimal (recall≥90%): 0.XXX

Rationale: Use precision-optimal if precision > 0.75, otherwise use F1-optimal

================================================================================
EXTERNAL TEST: PHOTO-LEVEL METRICS
================================================================================
Total photos: XX

Original threshold (from training):
  Threshold: 0.900
  Precision: XX.X%
  Recall:    XX.X%
  F1:        XX.X%
  FP: XX, FN: XX

Optimal threshold for external (0.XXX):
  Precision: XX.X% (+X.X%)
  Recall:    XX.X% (+X.X%)
  F1:        XX.X% (+X.X%)
  FP: XX (+X), FN: XX (-X)

✓ External photo-level analysis complete

================================================================================
EXTERNAL TEST: CALIBRATION
================================================================================
⚠️  Note: External test uses model calibrated on internal validation data
    Calibration parameters from training run are already applied.
✓ Copied calibration_T.json from training run
✓ Copied calibration_report.json from training run

✓ External test analysis complete
================================================================================
```

---

## 🎯 Benefici dell'Aggiornamento

1. **Completezza:** Tutti i file necessari per analisi completa
2. **Threshold Ottimale:** Trova il threshold migliore per external dataset
3. **Metriche Realistiche:** Photo-level metrics invece di image-level
4. **Confronto Diretto:** Confronta internal vs external con stesse metriche
5. **Riproducibilità:** Stesso workflow per internal ed external
6. **Automatico:** Nessuna azione manuale richiesta per run future

---

## 📝 Note Importanti

1. **Calibration:** L'external test usa la calibration del training (T=1.0000 nel tuo caso)
2. **Threshold Diverso:** L'external potrebbe avere threshold ottimale diverso dall'internal
3. **Photo-Level:** Se external ha solo 1 versione per foto, photo-level = image-level
4. **Backup Automatico:** I nuovi file saranno inclusi nel backup Drive e push GitHub

---

## ✅ Verifica

Per verificare che l'aggiornamento sia stato applicato correttamente:

1. **Controlla il notebook:**
   ```bash
   # Cerca la nuova cella
   grep -A 5 "EXTERNAL TEST: PHOTO-LEVEL ANALYSIS" scripts/notebooks/verifoto_dl.ipynb
   ```

2. **Esegui il notebook su Colab** (per run future)

3. **Genera i file per il run corrente** (opzionale, vedi sopra)

---

## 🚀 Prossimi Step

### Per il Run Corrente (2026-02-20_convnext_v8_domaiAware_external)

**Opzione A:** Genera i file localmente (5 min)
```bash
python scripts/analyze_by_photo.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware_external \
  --min-recall 0.90
```

**Opzione B:** Aspetta il prossimo training (i file saranno generati automaticamente)

### Per Run Future

✅ **Nessuna azione richiesta!** Il notebook ora genera automaticamente tutti i file.

---

## 📞 Domande?

- **Q: Devo rifare il training?**
  - A: No, puoi generare i file per il run corrente con il comando sopra

- **Q: I file saranno pushati su GitHub?**
  - A: Sì, il notebook fa `git add -f {OUTPUT_DIR}` che include tutti i file

- **Q: Cosa succede se external test non esiste?**
  - A: La cella salta l'analisi con un messaggio "External test not found"

- **Q: Posso usare threshold diverso per external?**
  - A: Sì! Il file `chosen_threshold.json` contiene il threshold ottimale per external

---

## 🎉 Conclusione

Il notebook è stato aggiornato con successo! Ora genera automaticamente tutti i file necessari per l'analisi completa di internal ed external test.

Per il run corrente, puoi generare i file mancanti con il comando sopra, oppure aspettare il prossimo training dove saranno generati automaticamente.
