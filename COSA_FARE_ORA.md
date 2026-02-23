# Cosa Fare Ora - Istruzioni Passo-Passo

## Problema

Anche usando `domain_aware` in entrambi i run, il test set è DIVERSO perché `domain_aware` ha randomness interno.

## Soluzione

Ho implementato il **salvataggio e caricamento dello split** per garantire split identici.

## 🔧 Passi da Seguire

### 1. Git Pull (Prendi Nuovo Codice)

```python
%cd /content/verifoto-dl
!git pull
```

### 2. Elimina Run Vecchi

```python
# Elimina training originale
!rm -rf outputs/runs/2026-02-20_convnext_v8_domaiAware
!rm -rf /content/drive/MyDrive/verifoto_checkpoints/2026-02-20_convnext_v8_domaiAware

# Elimina hard negative
!rm -rf outputs/runs/2026-02-20_convnext_v8_domaiAware_hard_negative
!rm -rf /content/drive/MyDrive/verifoto_checkpoints/2026-02-20_convnext_v8_domaiAware_hard_negative
```

### 3. Re-Train Modello Originale

Esegui la cella di training nel notebook.

Il nuovo codice salverà lo split in:
```
outputs/runs/2026-02-20_convnext_v8_domaiAware/split/
  train_split.csv
  val_split.csv
  test_split.csv
```

### 4. Verifica Split Salvato

```python
import os
split_dir = "outputs/runs/2026-02-20_convnext_v8_domaiAware/split"
print(f"Split saved: {os.path.exists(split_dir)}")
if os.path.exists(split_dir):
    print(f"Files: {os.listdir(split_dir)}")
```

Dovresti vedere:
```
Split saved: True
Files: ['train_split.csv', 'val_split.csv', 'test_split.csv']
```

### 5. Run Hard Negative

Esegui la cella hard negative nel notebook.

Il nuovo codice caricherà lo split salvato e vedrai:
```
✓ Loading split from outputs/runs/.../split/ (ensures same test set as original)
  Train: 3503 images
  Val:   760 images
  Test:  788 images
```

### 6. Verifica Test Set Identico

```python
import pandas as pd

orig_pred = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware/photo_level_predictions.csv")
hn_pred = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware_hard_negative/photo_level_predictions.csv")

orig_photos = set(orig_pred['photo_id'].values)
hn_photos = set(hn_pred['photo_id'].values)

print(f"Original test: {len(orig_photos)} photos")
print(f"Hard neg test: {len(hn_photos)} photos")
print(f"Overlap: {len(orig_photos & hn_photos)}/{len(orig_photos)}")
print(f"Missing: {len(orig_photos - hn_photos)}")
print(f"New: {len(hn_photos - orig_photos)}")
```

**DEVE mostrare:**
```
Original test: 226 photos
Hard neg test: 226 photos
Overlap: 226/226  ← 100%!
Missing: 0
New: 0
```

### 7. Analizza Risultati

```python
!python scripts/analyze_hard_negative_results.py
```

Dovresti vedere:
```
✅ Test sets are identical (good)
✅ F1 improvement is realistic (+3-7%)
✅ Moderate fix rate - plausible (40-60%)
```

## 📊 Risultati Attesi

```
Original:
  Precision: 74.2%
  F1:        81.4%

Hard Negative (atteso):
  Precision: ~78-85% (+4-11%)
  F1:        ~83-88% (+2-7%)
  Fix rate:  ~40-60%
```

**NON** aspettarti:
- F1 91% (troppo alto)
- Fix rate 94% (irrealistico)

## ⚠️ Importante

- **DEVI re-trainare** il modello originale (il vecchio non ha salvato lo split)
- **NON puoi** usare il training vecchio con hard negative nuovo
- Il re-training richiede ~8-10 ore su A100

## ✅ Checklist

- [ ] Git pull
- [ ] Elimina run vecchi
- [ ] Re-train originale (salva split)
- [ ] Verifica split salvato
- [ ] Run hard negative (carica split)
- [ ] Verifica test set identico (100% overlap)
- [ ] Analizza risultati (F1 ~85%)

## Conclusione

Questa è l'UNICA soluzione robusta per garantire split identici con `domain_aware`.

Dopo il re-training, i risultati saranno finalmente validi e realistici!
