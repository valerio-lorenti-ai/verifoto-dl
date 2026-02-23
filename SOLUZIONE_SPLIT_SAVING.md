# Soluzione: Split Saving per Prevenire Data Leakage

**Date:** 2026-02-20  
**Status:** ✅ SOLUZIONE IMPLEMENTATA

## Problema

Anche usando `domain_aware` in entrambi i run, il test set è COMPLETAMENTE DIVERSO:
- Original: 226 foto
- Hard Negative: 221 foto
- Overlap: Solo 35 foto (15%)!

**Causa:** `domain_aware_group_split_v1` ha randomness interno e genera split diversi ogni volta, anche con lo stesso seed.

## Soluzione Implementata

### 1. train_v7.py Salva lo Split

Dopo aver generato lo split, ora lo salva:

```python
# Save split for reproducibility
split_dir = output_dir / "split"
split_dir.mkdir(exist_ok=True)
train_df.to_csv(split_dir / "train_split.csv", index=False)
val_df.to_csv(split_dir / "val_split.csv", index=False)
test_df.to_csv(split_dir / "test_split.csv", index=False)
print(f"✓ Saved split to {split_dir}/ (for reproducibility)")
```

Questo crea:
```
outputs/runs/2026-02-20_convnext_v8_domaiAware/
  split/
    train_split.csv
    val_split.csv
    test_split.csv
```

### 2. hard_negative_finetune.py Carica lo Split

Prima di generare un nuovo split, prova a caricare quello originale:

```python
# Try to load split from original run
split_dir = Path(args.run) / "split"
if split_dir.exists() and (split_dir / "train_split.csv").exists():
    print(f"✓ Loading split from {split_dir}/")
    train_df = pd.read_csv(split_dir / "train_split.csv")
    val_df = pd.read_csv(split_dir / "val_split.csv")
    test_df = pd.read_csv(split_dir / "test_split.csv")
else:
    print(f"⚠️  WARNING: Split files not found")
    # Fallback: generate new split
    ...
```

Questo GARANTISCE che hard negative usi lo STESSO split del training originale.

## Cosa Fare Ora

### Step 1: Re-Train Original (con Split Saving)

Il training originale NON ha salvato lo split (il codice è stato aggiunto ora).

Devi re-trainare da zero:

```python
# Su Colab
%cd /content/verifoto-dl
!git pull  # Prendi il nuovo codice

# Elimina run vecchio
!rm -rf outputs/runs/2026-02-20_convnext_v8_domaiAware
!rm -rf /content/drive/MyDrive/verifoto_checkpoints/2026-02-20_convnext_v8_domaiAware

# Re-train (ora salverà lo split)
!python -m src.train_v7 \
  --config configs/convnext_v8.yaml \
  --run_name 2026-02-20_convnext_v8_domaiAware \
  --checkpoint_dir /content/drive/MyDrive/verifoto_checkpoints
```

### Step 2: Verifica Split Salvato

Dopo il training, verifica:

```python
import os
split_dir = "outputs/runs/2026-02-20_convnext_v8_domaiAware/split"
print(f"Split saved: {os.path.exists(split_dir)}")
print(f"Files: {os.listdir(split_dir) if os.path.exists(split_dir) else 'N/A'}")
```

Dovresti vedere:
```
Split saved: True
Files: ['train_split.csv', 'val_split.csv', 'test_split.csv']
```

### Step 3: Run Hard Negative (con Split Loading)

Ora hard negative caricherà lo split salvato:

```python
# Elimina run vecchio
!rm -rf outputs/runs/2026-02-20_convnext_v8_domaiAware_hard_negative

# Run hard negative (caricherà lo split salvato)
# Esegui la cella hard negative nel notebook
```

### Step 4: Verifica Test Set Identico

Dopo hard negative, verifica:

```python
import pandas as pd

orig_pred = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware/photo_level_predictions.csv")
hn_pred = pd.read_csv("outputs/runs/2026-02-20_convnext_v8_domaiAware_hard_negative/photo_level_predictions.csv")

orig_photos = set(orig_pred['photo_id'].values)
hn_photos = set(hn_pred['photo_id'].values)

print(f"Original test: {len(orig_photos)} photos")
print(f"Hard neg test: {len(hn_photos)} photos")
print(f"Overlap: {len(orig_photos & hn_photos)}/{len(orig_photos)} ({len(orig_photos & hn_photos)/len(orig_photos)*100:.1f}%)")
print(f"Missing: {len(orig_photos - hn_photos)}")
print(f"New: {len(hn_photos - orig_photos)}")
```

Dovresti vedere:
```
Original test: 226 photos
Hard neg test: 226 photos
Overlap: 226/226 (100.0%)  ← PERFETTO!
Missing: 0
New: 0
```

## Risultati Attesi

Con split identico, i risultati saranno più realistici:

```
Original:
  Precision: 74.2%
  Recall:    90.2%
  F1:        81.4%

Hard Negative (atteso):
  Precision: ~78-85% (+4-11%)
  Recall:    ~88-92% (-2% to +2%)
  F1:        ~83-88% (+2-7%)
  Fix rate:  ~40-60% (non 94%)
```

## Vantaggi della Soluzione

✅ **Garantisce split identico** (100% overlap)  
✅ **Previene data leakage** (nessuna foto si sposta tra train/test)  
✅ **Riproducibile** (stesso split ogni volta)  
✅ **Robusto** (funziona con qualsiasi split strategy)  
✅ **Backward compatible** (fallback se split non salvato)

## Conclusione

🚨 **Devi re-trainare** il modello originale con il nuovo codice

✅ **Soluzione implementata** (split saving + loading)

⏳ **Dopo re-training** hard negative userà lo split corretto

📊 **Risultati saranno realistici** (F1 ~85%, non 91%)

Questa è l'UNICA soluzione robusta per garantire split identici con `domain_aware`.
