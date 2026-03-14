# ❓ Dataset Transformations - FAQ & Troubleshooting

## Domande Frequenti

### Q1: Devo rigenerare i dataset ogni volta?
**No!** I dataset trasformati vengono salvati su Google Drive. Le celle controllano se esistono già e saltano la generazione.

Per rigenerare:
1. Elimina la cartella su Drive
2. Riesegui la cella

### Q2: Posso modificare i parametri delle trasformazioni?
**Sì!** Modifica i parametri nelle celle:

```python
# High-pass: cambia kernel Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)  # Default: 3

# Median Residual: cambia kernel size
MEDIAN_KERNEL = 7  # Default: 5

# Gaussian Residual: cambia kernel e sigma
GAUSSIAN_KERNEL = 7  # Default: 5
GAUSSIAN_SIGMA = 2.0  # Default: 1.5
```

### Q3: Le trasformazioni mantengono le dimensioni originali?
**Sì!** Ogni immagine trasformata ha esattamente le stesse dimensioni dell'originale.

### Q4: Posso usare dataset trasformati con altri config?
**Sì!** I dataset trasformati funzionano con qualsiasi config (convnext_v7, convnext_v8, etc.).

### Q5: Quanto spazio occupano i dataset trasformati?
Circa lo stesso del dataset originale:
- Dataset originale: ~2-3 GB
- Ogni trasformato: ~2-3 GB
- Totale (4 dataset): ~8-12 GB

### Q6: Posso eliminare i dataset trasformati dopo il training?
**Sì**, ma dovrai rigenerarli se vuoi rifare esperimenti. Consiglio di tenerli se hai spazio.

### Q7: Le trasformazioni sono reversibili?
**No**, sono lossy. Mantieni sempre il dataset RGB originale.

### Q8: Posso combinare più trasformazioni?
Non con le celle attuali, ma puoi creare una nuova cella che applica trasformazioni in sequenza.

## Troubleshooting

### Problema: "Dataset già esistente"
**Causa**: La cartella esiste già su Drive

**Soluzione**:
1. Se vuoi rigenerare: elimina la cartella su Drive
2. Se vuoi usare quello esistente: salta la cella

### Problema: "Impossibile caricare immagine"
**Causa**: File corrotto o formato non supportato

**Soluzione**:
- La cella continua con le altre immagini
- Controlla il conteggio errori alla fine
- Se molti errori, verifica dataset originale

### Problema: "Out of memory"
**Causa**: Immagini troppo grandi o RAM insufficiente

**Soluzione**:
```python
# Aggiungi resize prima della trasformazione
def apply_highpass(image):
    # Resize se troppo grande
    max_size = 1024
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
    
    # ... resto della trasformazione
```

### Problema: "Permission denied" su Drive
**Causa**: Drive non montato correttamente

**Soluzione**:
1. Riesegui mount di Google Drive
2. Verifica permessi di scrittura
3. Prova a creare una cartella manualmente su Drive

### Problema: Immagini trasformate appaiono tutte grigie
**Causa**: Normalizzazione errata o immagini senza varianza

**Soluzione**:
- Verifica che `abs_max > 1e-8`
- Controlla che le immagini originali non siano uniformi
- Aumenta kernel size se troppo piccolo

### Problema: Training non migliora con dataset trasformati
**Causa**: Possibili cause multiple

**Analisi**:
1. Verifica che le trasformazioni siano corrette (visualizza alcune immagini)
2. Controlla che il dataloader carichi correttamente
3. Confronta loss curves RGB vs trasformato
4. Analizza se il modello converge

**Possibili soluzioni**:
- Aumenta learning rate
- Aumenta numero epoche
- Prova data augmentation diverso
- Considera dual-input invece di solo noise

### Problema: Trasformazione troppo lenta
**Causa**: Molte immagini o immagini grandi

**Soluzione**:
```python
# Aggiungi resize per velocizzare
def apply_transformation(image):
    # Resize temporaneo per processing
    h, w = image.shape[:2]
    scale = 512 / max(h, w)
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
    
    # Trasformazione
    transformed = ...
    
    # Resize back a dimensioni originali
    transformed = cv2.resize(transformed, (w, h))
    
    return transformed
```

### Problema: Struttura cartelle non mantenuta
**Causa**: Bug nel path relativo

**Verifica**:
```python
# Stampa path per debug
print(f"Source: {img_path}")
print(f"Relative: {rel_path}")
print(f"Target: {target_path}")
```

**Soluzione**:
```python
# Assicurati che rel_path sia corretto
rel_path = img_path.relative_to(source_path)
target_path = Path(TARGET_ROOT) / rel_path
target_path.parent.mkdir(parents=True, exist_ok=True)
```

## Verifica Trasformazioni

### Visualizza Esempi
Aggiungi questa cella dopo la generazione per verificare:

```python
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# Carica un'immagine da ogni dataset
datasets = [
    "exp_3_augmented_v6.1_categorized",
    "exp_3_augmented_v6.1_categorized_highpass",
    "exp_3_augmented_v6.1_categorized_median_residual",
    "exp_3_augmented_v6.1_categorized_gaussian_residual"
]

# Prendi prima immagine da originali
sample_path = "originali/food_category_1/image1.jpg"  # Modifica con path reale

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for idx, dataset in enumerate(datasets):
    img_path = f"/content/drive/MyDrive/DatasetVerifoto/images/{dataset}/{sample_path}"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    axes[idx].imshow(img)
    axes[idx].set_title(dataset.split('_')[-1] if '_' in dataset else 'RGB')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

### Verifica Statistiche
Controlla che le trasformazioni abbiano senso:

```python
import numpy as np

for dataset in datasets:
    img_path = f"/content/drive/MyDrive/DatasetVerifoto/images/{dataset}/{sample_path}"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"\n{dataset}:")
    print(f"  Shape: {img.shape}")
    print(f"  Min: {img.min()}, Max: {img.max()}")
    print(f"  Mean: {img.mean():.2f}, Std: {img.std():.2f}")
```

**Valori attesi**:
- RGB: Mean ~127, Std ~50-70
- High-pass: Mean ~127, Std ~20-40
- Median Residual: Mean ~127, Std ~10-30
- Gaussian Residual: Mean ~127, Std ~10-30

### Verifica Conteggio File
Assicurati che tutti i file siano stati trasformati:

```python
from pathlib import Path

for dataset in datasets:
    root = Path(f"/content/drive/MyDrive/DatasetVerifoto/images/{dataset}")
    n_files = len(list(root.rglob("*.jpg")))
    print(f"{dataset}: {n_files} files")
```

Tutti i dataset dovrebbero avere lo stesso numero di file.

## Best Practices

### 1. Genera dataset in ordine
1. High-pass (più veloce)
2. Median Residual
3. Gaussian Residual

### 2. Verifica prima di training
- Visualizza alcune immagini
- Controlla statistiche
- Verifica conteggio file

### 3. Documenta parametri
Annota i parametri usati per ogni dataset:
```python
# High-pass: ksize=3
# Median: kernel=5
# Gaussian: kernel=5, sigma=1.5
```

### 4. Backup su Drive
I dataset sono già su Drive, ma considera backup locale se hai spazio.

### 5. Naming convention
Usa nomi descrittivi:
```python
# Buono
"exp_3_augmented_v6.1_categorized_median_residual_k5"

# Meglio
"exp_3_augmented_v6.1_categorized_median_k5"

# Evita
"dataset_noise_1"
```

## Performance Tips

### Velocizzare Generazione
```python
# 1. Usa tqdm per progress bar (già incluso)
# 2. Processa in batch se hai RAM
# 3. Usa multiprocessing (avanzato)

from multiprocessing import Pool

def process_image(img_path):
    # ... trasformazione
    pass

with Pool(4) as p:
    p.map(process_image, image_files)
```

### Ridurre Spazio
```python
# Salva con compressione JPEG
cv2.imwrite(str(target_path), transformed_bgr, 
            [cv2.IMWRITE_JPEG_QUALITY, 90])  # Default: 95
```

### Monitorare Progresso
```python
# Aggiungi timestamp
import time
start = time.time()

# ... processing

elapsed = time.time() - start
print(f"Tempo totale: {elapsed/60:.1f} minuti")
print(f"Tempo per immagine: {elapsed/len(image_files):.2f} secondi")
```

## Checklist Pre-Training

Prima di iniziare il training con dataset trasformati:

- [ ] Dataset generato completamente (no errori)
- [ ] Stesso numero di file del dataset originale
- [ ] Visualizzate alcune immagini (sembrano corrette)
- [ ] Statistiche verificate (mean ~127)
- [ ] Struttura cartelle mantenuta
- [ ] DATASET_NAME aggiornato nel notebook
- [ ] Config file corretto
- [ ] Spazio sufficiente su Drive per checkpoint

---

**Tutto pronto per gli esperimenti!** 🎯
