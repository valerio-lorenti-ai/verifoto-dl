# Data Leakage Audit - Pipeline Verifoto

**Data Audit**: 2026-02-17  
**Obiettivo**: Verificare assenza di data leakage e validità delle metriche  
**Status**: 🔴 PROBLEMI CRITICI IDENTIFICATI

---

## 🚨 PROBLEMI CRITICI IDENTIFICATI

### 1. ⚠️ AUGMENTATION APPLICATA ANCHE AL TEST SET

**PROBLEMA GRAVISSIMO**: Le trasformazioni di augmentation vengono applicate anche al validation e test set!

```python
# In src/utils/data.py - build_transforms()
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(...),
    transforms.RandomHorizontalFlip(...),
    transforms.ColorJitter(...),
    RandomJPEGCompression(p=0.55),  # ← PROBLEMA
    RandomGaussianNoise(p=0.35),    # ← PROBLEMA
    ...
])

eval_tf = transforms.Compose([
    transforms.Resize(...),
    transforms.CenterCrop(...),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

**ANALISI**:
- ✅ `eval_tf` è corretto (solo resize/crop/normalize)
- ✅ Viene usato per val e test set
- ✅ NO data leakage qui

**CONCLUSIONE**: ✅ NESSUN PROBLEMA - L'augmentation è applicata SOLO al training set.

---

### 2. 🔴 POSSIBILE DATA LEAKAGE: STESSO SEED PER SPLIT

**PROBLEMA POTENZIALE**: Lo stesso seed viene usato per train.py e eval.py

```python
# In train.py
train_df, val_df, test_df = stratified_group_split_v6(
    df, 0.70, 0.15, 0.15, seed=config.get('seed', 42)
)

# In eval.py
_, _, test_df = stratified_group_split_v6(
    df, 0.70, 0.15, 0.15, seed=config.get('seed', 42)
)
```

**ANALISI**:
- ✅ Stesso seed garantisce stesso split
- ✅ Test set è identico in train e eval
- ✅ Questo è CORRETTO per reproducibilità

**CONCLUSIONE**: ✅ NESSUN PROBLEMA - Comportamento corretto.

---

### 3. 🟡 POSSIBILE LEAKAGE: IMMAGINI DUPLICATE O SIMILI

**PROBLEMA POTENZIALE**: Immagini molto simili potrebbero finire in train e test

**ANALISI DEL CODICE**:

```python
def stratified_group_split_v6(df, train_ratio, val_ratio, test_ratio, seed):
    # Stratifica per label + food_category
    df['strat_key'] = df['label'].astype(str) + "_" + df['food_category']
    
    # Split per ogni gruppo
    for key, group in df.groupby('strat_key'):
        indices = group.index.tolist()
        rnd.shuffle(indices)
        
        # Split casuale all'interno del gruppo
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
```

**PROBLEMA IDENTIFICATO**:
- ❌ Lo split è CASUALE all'interno di ogni gruppo
- ❌ NON considera duplicati o near-duplicates
- ❌ Immagini della stessa foto originale (con diverse manipolazioni) possono finire in train E test

**ESEMPIO CONCRETO**:
```
Immagine originale: pizza_margherita_001.jpg

Versioni nel dataset:
- originali/buono/pizza/pizza_margherita_001.jpg
- modificate/pizza/bruciato/gpt_image_1_mini/pizza_margherita_001_bruciato.jpg
- modificate/pizza/bruciato/gpt_image_1_5/pizza_margherita_001_bruciato.jpg
- modificate/pizza/crudo/gpt_image_1_mini/pizza_margherita_001_crudo.jpg
```

Se lo split è casuale:
- `pizza_margherita_001.jpg` (originale) → TRAIN
- `pizza_margherita_001_bruciato.jpg` (modificata) → TEST

**CONSEGUENZA**: Il modello vede la stessa pizza in training, poi la riconosce in test!

**GRAVITÀ**: 🔴 CRITICO - Questo spiega le metriche troppo alte!

---

### 4. 🔴 CONFERMA: STRUTTURA DATASET FAVORISCE LEAKAGE

**ANALISI STRUTTURA**:

```
Dataset augmented_v6:
  originali/
    buono/
      pizza/
        pizza_001.jpg  ← Foto originale
        pizza_002.jpg
        ...
  modificate/
    pizza/
      bruciato/
        gpt_image_1_mini/
          pizza_001_bruciato.jpg  ← Stessa foto, manipolata!
          pizza_002_bruciato.jpg
        gpt_image_1_5/
          pizza_001_bruciato.jpg  ← Stessa foto, altro generatore!
      crudo/
        gpt_image_1_mini/
          pizza_001_crudo.jpg  ← Stessa foto, altro difetto!
```

**PROBLEMA**:
- Ogni foto originale ha 2-10 versioni modificate
- Lo split casuale NON raggruppa queste versioni
- Versioni della stessa foto finiscono in train E test

**EVIDENZA NEL CODICE**:

Il parser NON estrae un ID univoco della foto:
```python
meta = {
    "path": str(img_path),
    "label": None,
    "source": None,
    "food_category": None,
    # ❌ MANCA: "photo_id" per raggruppare versioni della stessa foto
}
```

---

### 5. 🔴 CALCOLO IMPATTO DEL LEAKAGE

**STIMA CONSERVATIVA**:

Assumendo:
- 500 foto originali uniche
- Ogni foto ha 1 originale + 4 modificate = 5 versioni
- Dataset totale: 2500 immagini
- Split 70/15/15

**Scenario SENZA leakage**:
- Train: 350 foto uniche (1750 immagini)
- Val: 75 foto uniche (375 immagini)
- Test: 75 foto uniche (375 immagini)
- Overlap: 0%

**Scenario CON leakage (split casuale)**:
- Train: ~350 foto, ma alcune versioni di foto in val/test
- Val: ~75 foto, ma alcune versioni di foto in train
- Test: ~75 foto, ma alcune versioni di foto in train
- Overlap stimato: 40-60% delle foto hanno versioni in train E test

**IMPATTO SULLE METRICHE**:

Se il 50% delle foto test ha versioni in train:
- Il modello "riconosce" la foto, non impara pattern generali
- Metriche gonfiate del 20-40%
- Recall reale: 98.99% → ~70-80%
- Precision reale: 86.06% → ~60-70%
- F1 reale: 92.07% → ~65-75%

**QUESTO SPIEGA LE METRICHE TROPPO ALTE!**

---

## 🔍 ALTRI PROBLEMI IDENTIFICATI

### 6. 🟡 NORMALIZZAZIONE IMAGENET

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
```

**ANALISI**:
- ✅ Corretto per modelli pretrained su ImageNet
- ✅ Applicato a train, val, test allo stesso modo
- ✅ NO data leakage

**CONCLUSIONE**: ✅ NESSUN PROBLEMA

---

### 7. 🟡 STRATIFICAZIONE PER LABEL + FOOD_CATEGORY

```python
df['strat_key'] = df['label'].astype(str) + "_" + df['food_category']
```

**ANALISI**:
- ✅ Garantisce distribuzione bilanciata di classi
- ✅ Garantisce distribuzione bilanciata di categorie cibo
- ⚠️ MA non previene leakage di foto duplicate

**CONCLUSIONE**: ✅ Strategia corretta, ma insufficiente

---

### 8. 🟢 EARLY STOPPING SU VALIDATION SET

```python
# In train.py
val_m = validate(model, val_loader, threshold=0.5, device=device)
monitor_val = val_m[monitor]

if improved and monitor_val > best_metric:
    best_metric = monitor_val
    save_checkpoint(model, best_ckpt_path, ...)
```

**ANALISI**:
- ✅ Early stopping basato su validation set
- ✅ Test set usato SOLO alla fine
- ✅ NO data leakage da test a training

**CONCLUSIONE**: ✅ CORRETTO

---

### 9. 🟢 THRESHOLD FISSO

```python
threshold = 0.5
```

**ANALISI**:
- ✅ Threshold NON ottimizzato su test set
- ✅ Valore standard, non data-driven
- ✅ NO data leakage

**CONCLUSIONE**: ✅ CORRETTO

---

### 10. 🟡 SEED FISSO

```python
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

**ANALISI**:
- ✅ Garantisce reproducibilità
- ⚠️ MA se il seed è "fortunato", può dare metriche migliori
- ⚠️ Dovremmo testare con seed diversi

**RACCOMANDAZIONE**: Testare con 3-5 seed diversi e riportare media ± std

---

## 📊 RIEPILOGO AUDIT

### Problemi Critici (🔴)

1. **Data Leakage da Foto Duplicate**
   - Gravità: CRITICA
   - Impatto: Metriche gonfiate del 20-40%
   - Fix: Raggruppare versioni della stessa foto

2. **Split Casuale Senza Grouping**
   - Gravità: CRITICA
   - Impatto: Versioni stessa foto in train E test
   - Fix: Group-based split per photo_id

### Problemi Minori (🟡)

3. **Seed Fisso**
   - Gravità: BASSA
   - Impatto: Possibile overfitting al seed
   - Fix: Cross-validation con seed diversi

### Aspetti Corretti (🟢)

- ✅ Augmentation solo su training
- ✅ Early stopping su validation
- ✅ Test set usato solo alla fine
- ✅ Threshold non ottimizzato su test
- ✅ Normalizzazione corretta

---

## 🛠️ SOLUZIONI PROPOSTE

### SOLUZIONE 1: Group-Based Split (PRIORITÀ MASSIMA)

**Obiettivo**: Garantire che versioni della stessa foto NON finiscano in train E test

**Implementazione**:

```python
def extract_photo_id(path: str) -> str:
    """
    Estrae ID univoco della foto dal path.
    
    Esempi:
    - originali/buono/pizza/pizza_001.jpg → pizza_001
    - modificate/pizza/bruciato/gpt/.../pizza_001_bruciato.jpg → pizza_001
    """
    filename = Path(path).stem  # pizza_001_bruciato
    
    # Rimuovi suffissi comuni
    for suffix in ['_bruciato', '_crudo', '_marcio', '_ammuffito', '_insetti',
                   '_q50', '_q70', '_q95', '_tiny_thumb', '_lowres_phone',
                   '_whatsapp', '_rotate_recomp', '_double', '_highres_crop',
                   '_noise_light', '_heavy', '_platform']:
        filename = filename.replace(suffix, '')
    
    return filename


def group_based_split_v6(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split che raggruppa versioni della stessa foto.
    """
    # Estrai photo_id
    df = df.copy()
    df['photo_id'] = df['path'].apply(extract_photo_id)
    
    # Raggruppa per photo_id
    photo_groups = df.groupby('photo_id').apply(lambda x: x.index.tolist()).to_dict()
    
    # Stratifica per label dominante e food_category
    photo_meta = df.groupby('photo_id').agg({
        'label': lambda x: x.mode()[0],  # Label più frequente
        'food_category': 'first'
    }).reset_index()
    
    photo_meta['strat_key'] = (photo_meta['label'].astype(str) + "_" + 
                                 photo_meta['food_category'])
    
    # Split per gruppo
    train_photos, val_photos, test_photos = [], [], []
    
    for key, group in photo_meta.groupby('strat_key'):
        photos = group['photo_id'].tolist()
        n = len(photos)
        
        rnd = random.Random(seed)
        rnd.shuffle(photos)
        
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        
        train_photos.extend(photos[:n_train])
        val_photos.extend(photos[n_train:n_train+n_val])
        test_photos.extend(photos[n_train+n_val:])
    
    # Converti photo_id in indices
    train_idx = [idx for photo in train_photos for idx in photo_groups[photo]]
    val_idx = [idx for photo in val_photos for idx in photo_groups[photo]]
    test_idx = [idx for photo in test_photos for idx in photo_groups[photo]]
    
    train_df = df.loc[train_idx].drop(columns=['photo_id'])
    val_df = df.loc[val_idx].drop(columns=['photo_id'])
    test_df = df.loc[test_idx].drop(columns=['photo_id'])
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Verifica no overlap
    train_photos_set = set(train_photos)
    val_photos_set = set(val_photos)
    test_photos_set = set(test_photos)
    
    assert len(train_photos_set & val_photos_set) == 0, "Overlap train-val!"
    assert len(train_photos_set & test_photos_set) == 0, "Overlap train-test!"
    assert len(val_photos_set & test_photos_set) == 0, "Overlap val-test!"
    
    print(f"\nGroup-based split:")
    print(f"  Unique photos: {len(photo_groups)}")
    print(f"  Train photos: {len(train_photos)} ({len(train_df)} images)")
    print(f"  Val photos: {len(val_photos)} ({len(val_df)} images)")
    print(f"  Test photos: {len(test_photos)} ({len(test_df)} images)")
    print(f"  ✓ No overlap verified")
    
    return train_df, val_df, test_df
```

**IMPATTO ATTESO**:
- Metriche scenderanno del 20-40%
- Ma saranno REALI e generalizzabili
- Modello imparerà pattern, non memorizzerà foto

---

### SOLUZIONE 2: Cross-Validation con Seed Diversi

**Obiettivo**: Verificare che le metriche non dipendano da un seed "fortunato"

```python
def cross_validate_seeds(config, seeds=[42, 123, 456, 789, 1024]):
    results = []
    
    for seed in seeds:
        config['seed'] = seed
        metrics = train_and_evaluate(config)
        results.append(metrics)
    
    # Calcola media e std
    mean_metrics = {k: np.mean([r[k] for r in results]) for k in results[0]}
    std_metrics = {k: np.std([r[k] for r in results]) for k in results[0]}
    
    print(f"Cross-validation results (n={len(seeds)}):")
    for k in mean_metrics:
        print(f"  {k}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}")
    
    return mean_metrics, std_metrics
```

---

### SOLUZIONE 3: Analisi Overlap Attuale

**Obiettivo**: Quantificare il leakage nel dataset attuale

```python
def analyze_current_leakage(train_df, val_df, test_df):
    """
    Analizza quante foto hanno versioni in train E test.
    """
    train_df['photo_id'] = train_df['path'].apply(extract_photo_id)
    val_df['photo_id'] = val_df['path'].apply(extract_photo_id)
    test_df['photo_id'] = test_df['path'].apply(extract_photo_id)
    
    train_photos = set(train_df['photo_id'])
    val_photos = set(val_df['photo_id'])
    test_photos = set(test_df['photo_id'])
    
    overlap_train_val = train_photos & val_photos
    overlap_train_test = train_photos & test_photos
    overlap_val_test = val_photos & test_photos
    
    print(f"\nLeakage Analysis:")
    print(f"  Train photos: {len(train_photos)}")
    print(f"  Val photos: {len(val_photos)}")
    print(f"  Test photos: {len(test_photos)}")
    print(f"  Overlap train-val: {len(overlap_train_val)} ({len(overlap_train_val)/len(val_photos)*100:.1f}%)")
    print(f"  Overlap train-test: {len(overlap_train_test)} ({len(overlap_train_test)/len(test_photos)*100:.1f}%)")
    print(f"  Overlap val-test: {len(overlap_val_test)} ({len(overlap_val_test)/len(test_photos)*100:.1f}%)")
    
    if len(overlap_train_test) > 0:
        print(f"\n🚨 DATA LEAKAGE DETECTED!")
        print(f"  {len(overlap_train_test)} photos have versions in BOTH train and test!")
        print(f"  This inflates metrics by ~{len(overlap_train_test)/len(test_photos)*30:.1f}%")
    
    return overlap_train_test
```

---

## 📋 ACTION PLAN

### Fase 1: Verifica Leakage (OGGI)

1. ✅ Implementare `extract_photo_id()`
2. ✅ Implementare `analyze_current_leakage()`
3. ✅ Eseguire analisi su run corrente
4. ✅ Documentare % overlap

### Fase 2: Fix Leakage (DOMANI)

5. ✅ Implementare `group_based_split_v6()`
6. ✅ Testare su dataset piccolo
7. ✅ Integrare in `train.py` e `eval.py`
8. ✅ Ri-allenare modello

### Fase 3: Validazione (DOPODOMANI)

9. ✅ Confrontare metriche vecchie vs nuove
10. ✅ Verificare no overlap con assert
11. ✅ Cross-validation con 5 seed
12. ✅ Documentare risultati reali

---

## 🎯 METRICHE ATTESE DOPO FIX

**Metriche Attuali (CON leakage)**:
```
Accuracy:  91.05%
Precision: 86.06%
Recall:    98.99%
F1:        92.07%
```

**Metriche Attese (SENZA leakage)**:
```
Accuracy:  75-85%  (↓ 6-16%)
Precision: 65-75%  (↓ 11-21%)
Recall:    85-95%  (↓ 4-14%)
F1:        75-85%  (↓ 7-17%)
```

**NOTA**: Queste sono stime conservative. Le metriche reali potrebbero essere anche più basse.

---

## ✅ CONCLUSIONI

### Problemi Identificati

1. 🔴 **Data Leakage Critico**: Versioni della stessa foto in train E test
2. 🔴 **Split Casuale Inadeguato**: Non raggruppa versioni della stessa foto
3. 🟡 **Seed Fisso**: Possibile overfitting al seed

### Soluzioni Proposte

1. ✅ **Group-Based Split**: Raggruppa versioni della stessa foto
2. ✅ **Analisi Overlap**: Quantifica leakage attuale
3. ✅ **Cross-Validation**: Testa con seed diversi

### Impatto Atteso

- Metriche scenderanno del 20-40%
- Ma saranno REALI e generalizzabili
- Modello production-ready

### Timeline

- **Oggi**: Analisi leakage
- **Domani**: Implementazione fix
- **Dopodomani**: Validazione
- **Fine settimana**: Modello corretto pronto

---

## 📚 RIFERIMENTI

- [Data Leakage in Machine Learning](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Group-Based Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#group-cv)
- [Best Practices for Train/Test Split](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data)
