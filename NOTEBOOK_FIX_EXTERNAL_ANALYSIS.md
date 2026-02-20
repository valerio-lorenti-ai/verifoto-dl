# 🔧 Fix: Aggiungere Photo-Level Analysis per External Test

## 🔍 Problema

L'external test (`2026-02-20_convnext_v8_domaiAware_external`) non ha:
- ❌ `threshold_sweep.csv`
- ❌ `photo_level_metrics.json`
- ❌ `chosen_threshold.json`
- ❌ File di calibration

**Causa:** Il notebook esegue photo-level analysis e calibration solo per il training interno, non per l'external test.

---

## ✅ Soluzione

Aggiungere una nuova cella nel notebook dopo l'external test che esegue:
1. Photo-level analysis
2. Temperature scaling calibration

---

## 📝 Codice da Aggiungere al Notebook

### Posizione

Aggiungere questa cella **DOPO** la cella "Test con dataset esterno - AntiLeakage" e **PRIMA** della cella "PUSH RESULTS TO GITHUB".

### Codice della Nuova Cella

```python
# ============================================================================
# EXTERNAL TEST: PHOTO-LEVEL ANALYSIS & CALIBRATION
# ============================================================================

EXTERNAL_RUN_NAME = f"{EXPERIMENT_NAME}_external"
EXTERNAL_OUTPUT_DIR = f"outputs/runs/{EXTERNAL_RUN_NAME}"

if os.path.exists(f"{EXTERNAL_OUTPUT_DIR}/metrics.json"):
    print("="*80)
    print("EXTERNAL TEST: PHOTO-LEVEL ANALYSIS")
    print("="*80)
    
    # Photo-level analysis
    !python scripts/analyze_by_photo.py \
        --run {EXTERNAL_OUTPUT_DIR} \
        --min-recall 0.90
    
    # Display chosen threshold
    if os.path.exists(f"{EXTERNAL_OUTPUT_DIR}/chosen_threshold.json"):
        import json
        with open(f"{EXTERNAL_OUTPUT_DIR}/chosen_threshold.json", 'r') as f:
            threshold_info = json.load(f)
        
        print("\n" + "="*80)
        print("EXTERNAL TEST: THRESHOLD RECOMMENDATION")
        print("="*80)
        print(f"🎯 Recommended threshold: {threshold_info['recommendation']:.3f}")
        print(f"   F1-optimal threshold: {threshold_info['f1_optimal']:.3f}")
        print(f"   Precision-optimal (recall≥90%): {threshold_info['precision_optimal_recall_90']:.3f}")
        print(f"\nRationale: {threshold_info['rationale']}")
        
        # Display photo-level metrics
        with open(f"{EXTERNAL_OUTPUT_DIR}/photo_level_metrics.json", 'r') as f:
            photo_metrics = json.load(f)
        
        print("\n" + "="*80)
        print("EXTERNAL TEST: PHOTO-LEVEL METRICS")
        print("="*80)
        print(f"Total photos: {photo_metrics['n_photos']}")
        
        print("\nOriginal threshold (from training):")
        orig = photo_metrics['original_threshold']
        print(f"  Threshold: {orig['threshold']:.3f}")
        print(f"  Precision: {orig['precision']:.1%}")
        print(f"  Recall:    {orig['recall']:.1%}")
        print(f"  F1:        {orig['f1']:.1%}")
        print(f"  FP: {orig['fp']}, FN: {orig['fn']}")
        
        print(f"\nOptimal threshold for external ({threshold_info['recommendation']:.3f}):")
        opt = photo_metrics['optimal_f1_threshold']
        print(f"  Precision: {opt['precision']:.1%} ({opt['precision']-orig['precision']:+.1%})")
        print(f"  Recall:    {opt['recall']:.1%} ({opt['recall']-orig['recall']:+.1%})")
        print(f"  F1:        {opt['f1']:.1%} ({opt['f1']-orig['f1']:+.1%})")
        print(f"  FP: {opt['fp']} ({int(opt['fp']-orig['fp']):+d}), FN: {opt['fn']} ({int(opt['fn']-orig['fn']):+d})")
        
        print("\n✓ External photo-level analysis complete")
    
    # Temperature scaling calibration (if validation data available)
    print("\n" + "="*80)
    print("EXTERNAL TEST: CALIBRATION")
    print("="*80)
    print("⚠️  Note: External test uses model calibrated on internal validation data")
    print("    Calibration parameters from training run are already applied.")
    
    # Copy calibration files from training run
    import shutil
    training_cal_T = f"{OUTPUT_DIR}/calibration_T.json"
    training_cal_report = f"{OUTPUT_DIR}/calibration_report.json"
    
    if os.path.exists(training_cal_T):
        shutil.copy(training_cal_T, f"{EXTERNAL_OUTPUT_DIR}/calibration_T.json")
        print(f"✓ Copied calibration_T.json from training run")
    
    if os.path.exists(training_cal_report):
        shutil.copy(training_cal_report, f"{EXTERNAL_OUTPUT_DIR}/calibration_report.json")
        print(f"✓ Copied calibration_report.json from training run")
    
    print("\n✓ External test analysis complete")
    print("="*80)
else:
    print("\n⚠️  External test not found - skipping analysis")
```

---

## 🎯 Cosa Fa Questa Cella

1. **Photo-Level Analysis:**
   - Aggrega le predizioni per foto (se ci sono versioni multiple)
   - Trova threshold ottimale per external dataset
   - Calcola metriche photo-level
   - Salva `threshold_sweep.csv`, `photo_level_metrics.json`, `chosen_threshold.json`

2. **Calibration:**
   - Copia i file di calibration dal training run
   - Nota che la calibration è già applicata (modello usa T dal training)

3. **Output:**
   - Mostra threshold raccomandato per external
   - Confronta metriche con threshold originale vs ottimale
   - Salva tutti i file necessari

---

## 📊 File Creati

Dopo aver aggiunto questa cella, l'external test avrà:

```
outputs/runs/2026-02-20_convnext_v8_domaiAware_external/
├── metrics.json                    ✅ (già presente)
├── predictions.csv                 ✅ (già presente)
├── threshold_sweep.csv             ✨ (NUOVO)
├── photo_level_metrics.json        ✨ (NUOVO)
├── photo_level_predictions.csv     ✨ (NUOVO)
├── chosen_threshold.json           ✨ (NUOVO)
├── photo_hard_fp.csv               ✨ (NUOVO)
├── photo_hard_fn.csv               ✨ (NUOVO)
├── threshold_sweep.png             ✨ (NUOVO)
├── calibration_T.json              ✨ (NUOVO - copiato)
└── calibration_report.json         ✨ (NUOVO - copiato)
```

---

## 🚀 Come Applicare il Fix

### Opzione 1: Manuale (Raccomandato)

1. Apri il notebook `scripts/notebooks/verifoto_dl.ipynb`
2. Trova la cella "Test con dataset esterno - AntiLeakage"
3. Aggiungi una nuova cella dopo di essa
4. Copia-incolla il codice sopra
5. Salva il notebook

### Opzione 2: Eseguire Localmente

Se hai già i risultati e vuoi solo generare i file mancanti:

```bash
# Photo-level analysis per external test
python scripts/analyze_by_photo.py \
  --run outputs/runs/2026-02-20_convnext_v8_domaiAware_external \
  --min-recall 0.90

# Copia calibration files
cp outputs/runs/2026-02-20_convnext_v8_domaiAware/calibration_T.json \
   outputs/runs/2026-02-20_convnext_v8_domaiAware_external/

cp outputs/runs/2026-02-20_convnext_v8_domaiAware/calibration_report.json \
   outputs/runs/2026-02-20_convnext_v8_domaiAware_external/
```

---

## 📈 Risultati Attesi

Dopo aver eseguito l'analisi, vedrai:

```
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
```

---

## 🎯 Benefici

1. **Threshold Ottimale per External:** Trova il threshold migliore per il dataset WhatsApp
2. **Metriche Realistiche:** Photo-level metrics invece di image-level
3. **Confronto Diretto:** Confronta internal vs external con stesse metriche
4. **Completezza:** Tutti i file necessari per analisi completa
5. **Riproducibilità:** Stesso workflow per internal ed external

---

## 📝 Note Importanti

1. **Calibration:** L'external test usa la calibration del training (T=1.0000 nel tuo caso)
2. **Threshold Diverso:** L'external potrebbe avere threshold ottimale diverso dall'internal
3. **Photo-Level:** Se external ha solo 1 versione per foto, photo-level = image-level
4. **Backup:** I nuovi file saranno inclusi nel backup Drive e push GitHub

---

## ✅ Verifica

Dopo aver applicato il fix, verifica che esistano:

```bash
ls outputs/runs/2026-02-20_convnext_v8_domaiAware_external/threshold_sweep.csv
ls outputs/runs/2026-02-20_convnext_v8_domaiAware_external/photo_level_metrics.json
ls outputs/runs/2026-02-20_convnext_v8_domaiAware_external/chosen_threshold.json
```

Se vedi tutti e 3 i file, il fix è stato applicato correttamente! ✅
