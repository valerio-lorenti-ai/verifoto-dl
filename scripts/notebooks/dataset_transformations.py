# -*- coding: utf-8 -*-
"""
Dataset Transformation Cells for verifoto_dl.ipynb

Inserire queste 3 celle nel notebook subito dopo il mount di Google Drive.
Ogni cella crea una versione trasformata del dataset RGB originale.

IMPORTANTE: Queste celle NON modificano il resto del notebook.
Cambiano solo il dataset utilizzato modificando DATASET_NAME.
"""

# ============================================================================
# CELLA 1: HIGH-PASS FILTER DATASET
# ============================================================================
"""
Crea dataset con filtro high-pass (Laplacian edge detection).
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
        """
        Applica filtro high-pass (Laplacian).
        
        Args:
            image: RGB image [0-255] uint8
        
        Returns:
            Filtered image [0-255] uint8
        """
        # Converti a grayscale per Laplacian
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Applica Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        
        # Normalizza per visualizzazione (centered)
        # Range: [-max, +max] → [0, 255]
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
            # Calcola path relativo
            rel_path = img_path.relative_to(source_path)
            target_path = Path(TARGET_ROOT) / rel_path
            
            # Crea directory se non esiste
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Carica immagine
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Impossibile caricare: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Applica trasformazione
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


# ============================================================================
# CELLA 2: MEDIAN RESIDUAL DATASET (Forensically-style)
# ============================================================================
"""
Crea dataset con Median Residual noise.
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

MEDIAN_KERNEL = 5  # Kernel size per median filter

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
        """
        Applica Median Residual (forensic noise analysis).
        
        Args:
            image: RGB image [0-255] uint8
            kernel_size: Median filter kernel size
        
        Returns:
            Residual image [0-255] uint8
        """
        # Converti a float [0, 1]
        img_float = image.astype(np.float32) / 255.0
        
        # Applica median blur
        img_uint8 = (img_float * 255).astype(np.uint8)
        denoised = cv2.medianBlur(img_uint8, kernel_size)
        denoised_float = denoised.astype(np.float32) / 255.0
        
        # Calcola residuo (RAW)
        residual = img_float - denoised_float
        
        # Normalizza per visualizzazione (centered)
        # Range: [-max, +max] → [0, 1]
        abs_max = max(abs(residual.min()), abs(residual.max()))
        if abs_max > 1e-8:
            normalized = (residual / (2 * abs_max)) + 0.5
        else:
            normalized = np.ones_like(residual) * 0.5
        
        normalized = np.clip(normalized, 0, 1)
        
        # Converti a uint8
        result = (normalized * 255).astype(np.uint8)
        
        return result
    
    # Trova tutte le immagini
    source_path = Path(SOURCE_ROOT)
    image_files = list(source_path.rglob("*.jpg")) + list(source_path.rglob("*.jpeg")) + list(source_path.rglob("*.png"))
    
    print(f"Trovate {len(image_files)} immagini")
    
    # Processa ogni immagine
    processed = 0
    errors = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # Calcola path relativo
            rel_path = img_path.relative_to(source_path)
            target_path = Path(TARGET_ROOT) / rel_path
            
            # Crea directory se non esiste
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Carica immagine
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Impossibile caricare: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Applica trasformazione
            transformed = apply_median_residual(img, MEDIAN_KERNEL)
            
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


# ============================================================================
# CELLA 3: GAUSSIAN RESIDUAL DATASET
# ============================================================================
"""
Crea dataset con Gaussian Residual noise.
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
        """
        Applica Gaussian Residual (forensic noise analysis).
        
        Args:
            image: RGB image [0-255] uint8
            kernel_size: Gaussian filter kernel size
            sigma: Gaussian sigma
        
        Returns:
            Residual image [0-255] uint8
        """
        # Converti a float [0, 1]
        img_float = image.astype(np.float32) / 255.0
        
        # Applica Gaussian blur
        denoised = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
        
        # Calcola residuo (RAW)
        residual = img_float - denoised
        
        # Normalizza per visualizzazione (centered)
        # Range: [-max, +max] → [0, 1]
        abs_max = max(abs(residual.min()), abs(residual.max()))
        if abs_max > 1e-8:
            normalized = (residual / (2 * abs_max)) + 0.5
        else:
            normalized = np.ones_like(residual) * 0.5
        
        normalized = np.clip(normalized, 0, 1)
        
        # Converti a uint8
        result = (normalized * 255).astype(np.uint8)
        
        return result
    
    # Trova tutte le immagini
    source_path = Path(SOURCE_ROOT)
    image_files = list(source_path.rglob("*.jpg")) + list(source_path.rglob("*.jpeg")) + list(source_path.rglob("*.png"))
    
    print(f"Trovate {len(image_files)} immagini")
    
    # Processa ogni immagine
    processed = 0
    errors = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # Calcola path relativo
            rel_path = img_path.relative_to(source_path)
            target_path = Path(TARGET_ROOT) / rel_path
            
            # Crea directory se non esiste
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Carica immagine
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Impossibile caricare: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Applica trasformazione
            transformed = apply_gaussian_residual(img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
            
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
