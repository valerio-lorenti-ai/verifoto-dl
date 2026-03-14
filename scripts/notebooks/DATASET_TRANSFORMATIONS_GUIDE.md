# 🔬 Dataset Transformations - Guida Integrazione

## Obiettivo

Aggiungere 3 celle al notebook `verifoto_dl.ipynb` per generare versioni trasformate del dataset RGB usando tecniche di noise analysis.

## Posizione delle Celle

Inserire le 3 nuove celle **subito dopo** la cella di mount di Google Drive:

```python
# Dopo questa cella:
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted")

# ← INSERIRE QUI LE 3 NUOVE CELLE
```

## Celle da Aggiungere

### CELLA 1: High-Pass Filter Dataset

```python
# ============================================================================
# DATASET TRANSFORMATION 1: HIGH-PASS FILTER
# ============================================================================
"""
Genera dataset con filtro high-pass (Laplacian edge detection).
Evidenzia bordi e dettagli ad alta frequenza.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import os

# Configurazione
SOURCE_DATASET = "exp_3_augmented_v6.1_categorized"  # Dataset RGB originale
TARGET_DATASET = f"{SOURCE_DATASET}_highpass"  # Nuovo dataset

SOURCE_ROOT = f"/content/drive/MyDrive/DatasetVerifoto/images/{SOURCE_DATASET}"
TARGET_ROOT = f"/content/drive/MyDrive/DatasetVerifoto/images/{TARGET_DATASET}"

print("="*80)
print("HIGH-PASS FILTER DATASET GENERATION")
print("="*80)
print(f"Source: {SOURCE_ROOT}")
print(f"Target: {TARGET_ROOT}")

# Check se già esiste
if os.path.exists(TARGET_ROOT):
    print(f"\n⚠️  Dataset già esistente: {TARGET_ROOT}")
    print("Saltando generazione. Elimina la cartella per rigenerare.")
else:
    print("\n🔬 Generazione dataset High-pass...")
    
    def apply_highpass(image):
        """Applica filtro high-pass (Laplacian)."""
        # Converti a grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Applica Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        
        # Normalizza (centered: [-max, +max] → [0, 255])
        abs_max = max(abs(laplacian.min()), abs(laplacian.max()))
        if abs_max > 1e-8:
            normalized = ((laplacian / abs_max) + 1.0) * 127.5
        else:
            normalized = np.ones_like(laplacian) * 127.5
        
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Converti a RGB (3 canali identici)
        rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        
        return rgb
    
    # Trova tutte le immagini
    source_path = Path(SOURCE_ROOT)
    image_files = list(source_path.rglob("*.jpg")) + list(source_path.rglob("*.jpeg")) + list(source_path.rglob("*.png"))
    
    print(f"Trovate {len(image_files)} immagini")
    
    # Processa ogni immagine
    processed = 0
    errors = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # Path relativo
            rel_path = img_path.relative_to(source_path)
            target_path = Path(TARGET_ROOT) / rel_path
            
            # Crea directory
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Carica
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Impossibile caricare: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Trasforma
            transformed = apply_highpass(img)
            
            # Salva
            transformed_bgr = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(target_path), transformed_bgr)
            
            processed += 1
            
        except Exception as e:
            print(f"\n✗ Errore {img_path.name}: {e}")
            errors += 1
            continue
    
    print(f"\n✓ Generazione completata!")
    print(f"  Processate: {processed}/{len(image_files)}")
    print(f"  Errori: {errors}")
    print(f"  Dataset salvato in: {TARGET_ROOT}")

print("\n" + "="*80)
print("Per usare questo dataset, modifica DATASET_NAME:")
print(f'DATASET_NAME = "{TARGET_DATASET}"')
print("="*80)
```

### CELLA 2: Median Residual Dataset

```python
# ============================================================================
# DATASET TRANSFORMATION 2: MEDIAN RESIDUAL
# ============================================================================
"""
Genera dataset con Median Residual noise (Forensically-style).
Sottrae versione denoised con median filter.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import os

# Configurazione
SOURCE_DATASET = "exp_3_augmented_v6.1_categorized"  # Dataset RGB originale
TARGET_DATASET = f"{SOURCE_DATASET}_median_residual"  # Nuovo dataset

SOURCE_ROOT = f"/content/drive/MyDrive/DatasetVerifoto/images/{SOURCE_DATASET}"
TARGET_ROOT = f"/content/drive/MyDrive/DatasetVerifoto/images/{TARGET_DATASET}"

MEDIAN_KERNEL = 5  # Kernel size

print("="*80)
print("MEDIAN RESIDUAL DATASET GENERATION")
print("="*80)
print(f"Source: {SOURCE_ROOT}")
print(f"Target: {TARGET_ROOT}")
print(f"Median kernel: {MEDIAN_KERNEL}")

# Check se già esiste
if os.path.exists(TARGET_ROOT):
    print(f"\n⚠️  Dataset già esistente: {TARGET_ROOT}")
    print("Saltando generazione. Elimina la cartella per rigenerare.")
else:
    print("\n🔬 Generazione dataset Median Residual...")
    
    def apply_median_residual(image, kernel_size=5):
        """Applica Median Residual (forensic noise analysis)."""
        # Float [0, 1]
        img_float = image.astype(np.float32) / 255.0
        
        # Median blur
        img_uint8 = (img_float * 255).astype(np.uint8)
        denoised = cv2.medianBlur(img_uint8, kernel_size)
        denoised_float = denoised.astype(np.float32) / 255.0
        
        # Residuo RAW
        residual = img_float - denoised_float
        
        # Normalizza (centered: [-max, +max] → [0, 1])
        abs_max = max(abs(residual.min()), abs(residual.max()))
        if abs_max > 1e-8:
            normalized = (residual / (2 * abs_max)) + 0.5
        else:
            normalized = np.ones_like(residual) * 0.5
        
        normalized = np.clip(normalized, 0, 1)
        
        # Uint8
        result = (normalized * 255).astype(np.uint8)
        
        return result
    
    # Trova immagini
    source_path = Path(SOURCE_ROOT)
    image_files = list(source_path.rglob("*.jpg")) + list(source_path.rglob("*.jpeg")) + list(source_path.rglob("*.png"))
    
    print(f"Trovate {len(image_files)} immagini")
    
    # Processa
    processed = 0
    errors = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            rel_path = img_path.relative_to(source_path)
            target_path = Path(TARGET_ROOT) / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Impossibile caricare: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            transformed = apply_median_residual(img, MEDIAN_KERNEL)
            
            transformed_bgr = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(target_path), transformed_bgr)
            
            processed += 1
            
        except Exception as e:
            print(f"\n✗ Errore {img_path.name}: {e}")
            errors += 1
            continue
    
    print(f"\n✓ Generazione completata!")
    print(f"  Processate: {processed}/{len(image_files)}")
    print(f"  Errori: {errors}")
    print(f"  Dataset salvato in: {TARGET_ROOT}")

print("\n" + "="*80)
print("Per usare questo dataset, modifica DATASET_NAME:")
print(f'DATASET_NAME = "{TARGET_DATASET}"')
print("="*80)
```

### CELLA 3: Gaussian Residual Dataset

```python
# ============================================================================
# DATASET TRANSFORMATION 3: GAUSSIAN RESIDUAL
# ============================================================================
"""
Genera dataset con Gaussian Residual noise.
Sottrae versione denoised con Gaussian blur.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import os

# Configurazione
SOURCE_DATASET = "exp_3_augmented_v6.1_categorized"  # Dataset RGB originale
TARGET_DATASET = f"{SOURCE_DATASET}_gaussian_residual"  # Nuovo dataset

SOURCE_ROOT = f"/content/drive/MyDrive/DatasetVerifoto/images/{SOURCE_DATASET}"
TARGET_ROOT = f"/content/drive/MyDrive/DatasetVerifoto/images/{TARGET_DATASET}"

GAUSSIAN_KERNEL = 5  # Kernel size
GAUSSIAN_SIGMA = 1.5  # Sigma

print("="*80)
print("GAUSSIAN RESIDUAL DATASET GENERATION")
print("="*80)
print(f"Source: {SOURCE_ROOT}")
print(f"Target: {TARGET_ROOT}")
print(f"Gaussian kernel: {GAUSSIAN_KERNEL}, sigma: {GAUSSIAN_SIGMA}")

# Check se già esiste
if os.path.exists(TARGET_ROOT):
    print(f"\n⚠️  Dataset già esistente: {TARGET_ROOT}")
    print("Saltando generazione. Elimina la cartella per rigenerare.")
else:
    print("\n🔬 Generazione dataset Gaussian Residual...")
    
    def apply_gaussian_residual(image, kernel_size=5, sigma=1.5):
        """Applica Gaussian Residual (forensic noise analysis)."""
        # Float [0, 1]
        img_float = image.astype(np.float32) / 255.0
        
        # Gaussian blur
        denoised = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
        
        # Residuo RAW
        residual = img_float - denoised
        
        # Normalizza (centered: [-max, +max] → [0, 1])
        abs_max = max(abs(residual.min()), abs(residual.max()))
        if abs_max > 1e-8:
            normalized = (residual / (2 * abs_max)) + 0.5
        else:
            normalized = np.ones_like(residual) * 0.5
        
        normalized = np.clip(normalized, 0, 1)
        
        # Uint8
        result = (normalized * 255).astype(np.uint8)
        
        return result
    
    # Trova immagini
    source_path = Path(SOURCE_ROOT)
    image_files = list(source_path.rglob("*.jpg")) + list(source_path.rglob("*.jpeg")) + list(source_path.rglob("*.png"))
    
    print(f"Trovate {len(image_files)} immagini")
    
    # Processa
    processed = 0
    errors = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            rel_path = img_path.relative_to(source_path)
            target_path = Path(TARGET_ROOT) / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Impossibile caricare: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            transformed = apply_gaussian_residual(img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
            
            transformed_bgr = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(target_path), transformed_bgr)
            
            processed += 1
            
        except Exception as e:
            print(f"\n✗ Errore {img_path.name}: {e}")
            errors += 1
            continue
    
    print(f"\n✓ Generazione completata!")
    print(f"  Processate: {processed}/{len(image_files)}")
    print(f"  Errori: {errors}")
    print(f"  Dataset salvato in: {TARGET_ROOT}")

print("\n" + "="*80)
print("Per usare questo dataset, modifica DATASET_NAME:")
print(f'DATASET_NAME = "{TARGET_DATASET}"')
print("="*80)
```

## Come Usare

### 1. Generare i Dataset Trasformati

Esegui le 3 celle in sequenza. Ogni cella:
- Controlla se il dataset esiste già
- Se non esiste, lo genera
- Mantiene la struttura originale del dataset

### 2. Usare un Dataset Trasformato

Nella cella di configurazione del notebook, modifica `DATASET_NAME`:

```python
# Per RGB originale (default)
DATASET_NAME = "exp_3_augmented_v6.1_categorized"

# Per High-pass
DATASET_NAME = "exp_3_augmented_v6.1_categorized_highpass"

# Per Median Residual
DATASET_NAME = "exp_3_augmented_v6.1_categorized_median_residual"

# Per Gaussian Residual
DATASET_NAME = "exp_3_augmented_v6.1_categorized_gaussian_residual"
```

Il resto del notebook funziona identicamente!

## Struttura Dataset Generati

Ogni dataset trasformato mantiene la stessa struttura:

```
exp_3_augmented_v6.1_categorized_highpass/
├── originali/
│   ├── food_category_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── food_category_2/
│       └── ...
└── modificate/
    ├── food_category_1/
    │   ├── image1_edit.jpg
    │   └── image2_edit.jpg
    └── food_category_2/
        └── ...
```

## Caratteristiche Tecniche

### High-Pass Filter
- **Tecnica**: Laplacian edge detection
- **Output**: Bordi e dettagli ad alta frequenza
- **Normalizzazione**: Centered ([-max, +max] → [0, 255])
- **Canali**: 3 (RGB identici)

### Median Residual
- **Tecnica**: image - median_blur(image)
- **Kernel**: 5x5
- **Output**: Noise pattern forensico
- **Normalizzazione**: Centered ([-max, +max] → [0, 255])
- **Canali**: 3 (RGB)

### Gaussian Residual
- **Tecnica**: image - gaussian_blur(image)
- **Kernel**: 5x5, sigma=1.5
- **Output**: Noise pattern forensico
- **Normalizzazione**: Centered ([-max, +max] → [0, 255])
- **Canali**: 3 (RGB)

## Ottimizzazioni Memoria

- Processa **una immagine alla volta**
- Salva immediatamente dopo trasformazione
- Non carica tutto il dataset in memoria
- Progress bar con `tqdm`

## Gestione Errori

- Se un'immagine fallisce, continua con le altre
- Conta errori e li mostra alla fine
- Non blocca l'intero processo

## Rigenerazione

Per rigenerare un dataset:
1. Elimina la cartella su Google Drive
2. Riesegui la cella corrispondente

## Tempo di Esecuzione

Per ~5000 immagini:
- High-pass: ~10-15 minuti
- Median Residual: ~15-20 minuti
- Gaussian Residual: ~15-20 minuti

## Spazio su Drive

Ogni dataset trasformato occupa circa lo stesso spazio del dataset originale (~2-3 GB per 5000 immagini).

---

**Pronto per l'uso!** Le celle sono standalone e non modificano il resto del notebook.
