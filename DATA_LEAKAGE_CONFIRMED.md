# Data Leakage Confermato - Analisi Completa

**Date:** 2026-02-20  
**Status:** 🚨 DATA LEAKAGE CONFERMATO

## 🚨 Risultati dell'Analisi

### Indicatori Critici

1. **Test set size cambiato: 226 → 221 foto (-5)**
   - ✅ PROVA DEFINITIVA di split diverso
   - Le foto sono finite in train/val invece che test

2. **F1 improvement: +12.5%**
   - Troppo alto per essere realistico
   - Threshold: >10% è sospetto

3. **Hard FP fix rate: 93.8% (15/16 foto)**
   - Irrealistico che il fine-tuning fissi il 94%
   - Threshold: >80% è sospetto

4. **Foto mancanti nel test set**
   - Molte foto originali hard FP hanno `nan` nel nuovo run
   - Significa che NON sono più nel test set
   - Probabilmente finite nel training set

### Confronto Metriche

```
Original (threshold=0.4):
  Precision: 74.2%
  Recall:    90.2%
  F1:        81.4%
  FP: 16, FN: 5

Hard Negative (threshold=0.9):
  Precision: 95.8%
  Recall:    92.0%
  F1:        93.9%
  FP: 2, FN: 4

Improvement:
  Precision: +21.6% ← TROPPO ALTO!
  F1:        +12.5% ← TROPPO ALTO!
```

### Hard FP Analysis

```
Original hard FP: 16 photos
  ['74ec', '7b56', 'aa12', '414d', 'b9be', 'e99d', 'b04d', 
   'k3xa', '63ad', '5cb6', 'dd44', 'f5d4', 'e156', '474b', 
   'b9a9', 's4md']

Hard Negative hard FP: 2 photos
  ['01bd', '7b56']

✅ Fixed: 15 photos (93.8%) ← SOSPETTO!
⚠️  Still FP: 1 photo (7b56)
🆕 New FP: 1 photo (01bd)
```

### Foto Mancanti (NaN)

Queste foto erano nel test set originale ma NON sono nel test set hard negative:
- 414d, 474b, 5cb6, 63ad, 74ec, aa12, b04d, b9a9, b9be, e156, e99d, k3xa, s4md, f5d4

**Dove sono finite?** Probabilmente in train/val → DATA LEAKAGE!

## Perché il Fix Non Ha Funzionato?

### Problema: Git Pull Non Eseguito

Il fix è stato implementato in `scripts/hard_negative_finetune.py`, ma:

1. Il codice è stato committato su GitHub
2. **Ma il notebook su Colab NON ha fatto `git pull`**
3. Quindi ha eseguito la versione VECCHIA del codice
4. Che usa ancora `group_based_split_v6` hardcoded

### Verifica

```python
# Codice VECCHIO (eseguito su Colab):
train_df, val_df, test_df = group_based_split_v6(...)  # ❌

# Codice NUOVO (nel repo):
split_strategy = config.get('split_strategy', 'group_v6')
if split_strategy == 'domain_aware':
    train_df, val_df, test_df = domain_aware_group_split_v1(...)  # ✅
```

## Soluzione

### Step 1: Verifica Git Pull su Colab

Nel notebook, nella cella di setup, assicurati che ci sia:

```python
%cd /content/verifoto-dl
!git pull  # ← IMPORTANTE!
!pip install -q -r requirements.txt
```

### Step 2: Elimina Risultati Invalidi

```bash
# Su Colab
!rm -rf outputs/runs/2026-02-20_convnext_v8_domaiAware_hard_negative
!rm -rf /content/drive/MyDrive/verifoto_checkpoints/2026-02-20_convnext_v8_domaiAware_hard_negative
```

### Step 3: Re-Run con Codice Aggiornato

1. Esegui la cella di setup (con `git pull`)
2. Verifica che il codice sia aggiornato:
   ```python
   !grep -A 5 "split_strategy" scripts/hard_negative_finetune.py
   ```
   Dovresti vedere il codice che legge dal config

3. Esegui di nuovo la cella hard negative

### Step 4: Verifica Risultati

Dopo il re-run, verifica:

```python
# Test set size dovrebbe essere UGUALE
original_metrics = json.load(open("outputs/runs/2026-02-20_convnext_v8_domaiAware/photo_level_metrics.json"))
hn_metrics = json.load(open("outputs/runs/2026-02-20_convnext_v8_domaiAware_hard_negative/photo_level_metrics.json"))

print(f"Original test photos: {original_metrics['n_photos']}")
print(f"Hard neg test photos: {hn_metrics['n_photos']}")

# Dovrebbero essere UGUALI (226 = 226)
```

## Risultati Attesi (Realistici)

Dopo il fix corretto:

```
Original:
  Precision: 74.2%
  Recall:    90.2%
  F1:        81.4%

Hard Negative (atteso):
  Precision: ~78-82% (+4-8%)
  Recall:    ~88-92% (-2% to +2%)
  F1:        ~83-87% (+2-6%)
```

**NON** aspettarti:
- Precision 95.8% (troppo alto)
- F1 93.9% (troppo alto)
- Fix rate 93.8% (irrealistico)

## Conclusione

🚨 **I risultati attuali NON sono validi** (data leakage confermato)

✅ **Il fix è stato implementato** ma non è stato eseguito su Colab

⏳ **Devi fare git pull su Colab e re-runnare**

📊 **Aspettati F1 ~85%** (miglioramento moderato +3-6%, non +12%)

## Checklist

Prima di re-runnare:
- [ ] Git pull su Colab
- [ ] Verifica codice aggiornato (`grep split_strategy`)
- [ ] Elimina risultati vecchi
- [ ] Re-run hard negative
- [ ] Verifica test set size (deve essere 226 = 226)
- [ ] Verifica F1 improvement (<10%)
- [ ] Verifica fix rate (<80%)
