# -*- coding: utf-8 -*-
"""
# 🔬 Forensic Noise Analysis - Verifoto

Copia questo file su Colab e eseguilo cella per cella.
Oppure usa: File > Upload notebook e carica questo .py

## 🎯 Obiettivo
Identificare quale tecnica di noise detection distingue meglio immagini originali da AI-edited.

## ⚠️ PRINCIPIO FORENSE
- ✅ PRESERVARE anomalie forensi
- ✅ Metriche su RAW data  
- ✅ Normalizzazioni SOLO per visualizzazione
"""

# ============================================================================
# CELLA 1: Installazione
# ============================================================================
# !pip install -q opencv-python-headless pillow numpy matplotlib scipy

# ============================================================================
# CELLA 2: Import
# ============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
print("✓ Setup completato")

# ============================================================================
# CELLA 3: Upload Immagini
# ============================================================================
from google.colab import files

print("📤 Carica immagine ORIGINALE:")
uploaded_orig = files.upload()
original_path = list(uploaded_orig.keys())[0]

print("\n📤 Carica immagine MODIFICATA (AI-edited):")
uploaded_mod = files.upload()
modified_path = list(uploaded_mod.keys())[0]

print(f"\n✓ Original: {original_path}")
print(f"✓ Modified: {modified_path}")


# ============================================================================
# CELLA 4: Classi di Analisi
# ============================================================================
@dataclass
class NoiseMetrics:
    """Metriche calcolate su RAW data."""
    std: float
    mean_abs: float
    range: float
    snr: float
    entropy: float
    percentile_95: float

class NoiseAnalyzer:
    """Analisi forense - Restituisce sempre RAW data."""
    
    def __init__(self, median_kernel=5, gaussian_kernel=5, gaussian_sigma=1.5, ela_quality=90):
        self.median_kernel = median_kernel
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
        self.ela_quality = ela_quality
    
    def load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Impossibile caricare: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0
    
    def median_noise_residual(self, image):
        """Median Residual - RAW."""
        img_uint8 = (image * 255).astype(np.uint8)
        denoised = cv2.medianBlur(img_uint8, self.median_kernel)
        denoised = denoised.astype(np.float32) / 255.0
        return image - denoised
    
    def gaussian_residual_noise(self, image):
        """Gaussian Residual - RAW."""
        denoised = cv2.GaussianBlur(image, (self.gaussian_kernel, self.gaussian_kernel), self.gaussian_sigma)
        return image - denoised
    
    def high_pass_filter(self, image):
        """High-pass Filter - RAW."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian = laplacian.astype(np.float32) / 255.0
        if len(image.shape) == 3:
            laplacian = np.stack([laplacian] * 3, axis=-1)
        return laplacian
    
    def error_level_analysis(self, image):
        """ELA - RAW."""
        img_uint8 = (image * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=self.ela_quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        compressed = np.array(compressed_img).astype(np.float32) / 255.0
        return np.abs(image - compressed)
    
    def compute_metrics(self, noise):
        """Calcola metriche su RAW data."""
        if len(noise.shape) == 3:
            noise_gray = cv2.cvtColor((np.clip(noise, -1, 1) * 127.5 + 127.5).astype(np.uint8), 
                                     cv2.COLOR_RGB2GRAY).astype(np.float32) / 127.5 - 1.0
        else:
            noise_gray = noise
        
        std = float(np.std(noise_gray))
        mean_abs = float(np.mean(np.abs(noise_gray)))
        range_val = float(np.max(noise_gray) - np.min(noise_gray))
        signal_power = np.mean(noise_gray ** 2)
        noise_power = np.var(noise_gray)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        noise_norm = (noise_gray - noise_gray.min()) / (noise_gray.max() - noise_gray.min() + 1e-10)
        hist, _ = np.histogram(noise_norm.flatten(), bins=256, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        percentile_95 = float(np.percentile(np.abs(noise_gray), 95))
        
        return NoiseMetrics(std, mean_abs, range_val, float(snr), float(entropy), percentile_95)
    
    def analyze_image(self, image):
        """Applica tutte le tecniche."""
        results = {}
        results['Median Residual'] = self.median_noise_residual(image)
        results['Gaussian Residual'] = self.gaussian_residual_noise(image)
        results['High-pass Filter'] = self.high_pass_filter(image)
        results['ELA'] = self.error_level_analysis(image)
        
        results_with_metrics = {}
        for technique, noise in results.items():
            metrics = self.compute_metrics(noise)
            results_with_metrics[technique] = (noise, metrics)
        
        return results_with_metrics

print("✓ NoiseAnalyzer definito")


# ============================================================================
# CELLA 5: Funzioni Visualizzazione
# ============================================================================
def normalize_for_display(noise, method="centered"):
    """Normalizza SOLO per visualizzazione."""
    if method == "centered":
        abs_max = max(abs(noise.min()), abs(noise.max()))
        if abs_max < 1e-8:
            return np.ones_like(noise) * 0.5
        normalized = (noise / (2 * abs_max)) + 0.5
        return np.clip(normalized, 0, 1)
    else:  # minmax
        range_val = noise.max() - noise.min()
        if range_val < 1e-8:
            return np.zeros_like(noise)
        return (noise - noise.min()) / range_val

def to_grayscale_display(noise, method="centered"):
    """Converte a grayscale per visualizzazione neutra."""
    normalized = normalize_for_display(noise, method)
    if len(normalized.shape) == 3:
        gray = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return gray.astype(np.float32) / 255.0
    return normalized

print("✓ Funzioni visualizzazione definite")

# ============================================================================
# CELLA 6: Esegui Analisi
# ============================================================================
analyzer = NoiseAnalyzer()

print("📂 Caricamento immagini...")
original = analyzer.load_image(original_path)
modified = analyzer.load_image(modified_path)
print(f"  Original: {original.shape}")
print(f"  Modified: {modified.shape}")

print("\n🔬 Applicazione tecniche forensi...")
original_results = analyzer.analyze_image(original)
modified_results = analyzer.analyze_image(modified)
print("✓ Analisi completata")

# ============================================================================
# CELLA 7: Metriche RAW
# ============================================================================
print("="*80)
print("METRICHE QUANTITATIVE (calcolate su RAW data)")
print("="*80)

for technique in original_results.keys():
    print(f"\n{technique}:")
    orig_noise, orig_metrics = original_results[technique]
    mod_noise, mod_metrics = modified_results[technique]
    
    print(f"  Original  → STD: {orig_metrics.std:.5f} | Entropy: {orig_metrics.entropy:.2f} | Range: {orig_metrics.range:.5f}")
    print(f"  Modified  → STD: {mod_metrics.std:.5f} | Entropy: {mod_metrics.entropy:.2f} | Range: {mod_metrics.range:.5f}")
    
    delta_std = abs(orig_metrics.std - mod_metrics.std)
    delta_entropy = abs(orig_metrics.entropy - mod_metrics.entropy)
    print(f"  Δ STD: {delta_std:.5f} | Δ Entropy: {delta_entropy:.2f}")
    
    # Interpretazione
    if delta_std > 0.01:
        print(f"  ✅ Δ STD > 0.01: Buona separazione!")
    if delta_entropy > 0.5:
        print(f"  ✅ Δ Entropy > 0.5: Pattern diversi!")


# ============================================================================
# CELLA 8: Visualizzazione Comparison
# ============================================================================
techniques = list(original_results.keys())
n_cols = len(techniques) * 2 + 1  # +1 per immagine originale

fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

# Row 1: Original
axes[0, 0].imshow(original)
axes[0, 0].set_title('ORIGINAL\nImage', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

col_idx = 1
for technique in techniques:
    noise, metrics = original_results[technique]
    
    # Determina metodo
    if 'Residual' in technique or 'High-pass' in technique:
        norm_method = 'centered'
        cmap = 'RdBu_r'
    else:
        norm_method = 'minmax'
        cmap = 'hot'
    
    # Neutral
    gray = to_grayscale_display(noise, norm_method)
    axes[0, col_idx].imshow(gray, cmap='gray', vmin=0, vmax=1)
    axes[0, col_idx].set_title(f'{technique}\n(Neutral)', fontsize=10)
    axes[0, col_idx].axis('off')
    
    # Enhanced
    colored = normalize_for_display(noise, norm_method)
    axes[0, col_idx + 1].imshow(colored, cmap=cmap)
    axes[0, col_idx + 1].set_title(f'{technique}\n(Enhanced)\nSTD:{metrics.std:.4f}', fontsize=10)
    axes[0, col_idx + 1].axis('off')
    
    col_idx += 2

# Row 2: Modified
axes[1, 0].imshow(modified)
axes[1, 0].set_title('MODIFIED\nImage', fontsize=12, fontweight='bold', color='red')
axes[1, 0].axis('off')

col_idx = 1
for technique in techniques:
    noise, metrics = modified_results[technique]
    
    if 'Residual' in technique or 'High-pass' in technique:
        norm_method = 'centered'
        cmap = 'RdBu_r'
    else:
        norm_method = 'minmax'
        cmap = 'hot'
    
    # Neutral
    gray = to_grayscale_display(noise, norm_method)
    axes[1, col_idx].imshow(gray, cmap='gray', vmin=0, vmax=1)
    axes[1, col_idx].set_title(f'{technique}\n(Neutral)', fontsize=10)
    axes[1, col_idx].axis('off')
    
    # Enhanced
    colored = normalize_for_display(noise, norm_method)
    axes[1, col_idx + 1].imshow(colored, cmap=cmap)
    axes[1, col_idx + 1].set_title(f'{technique}\n(Enhanced)\nSTD:{metrics.std:.4f}', fontsize=10)
    axes[1, col_idx + 1].axis('off')
    
    col_idx += 2

plt.suptitle('Forensic Noise Analysis: Original vs Modified', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

print("\n✓ Comparison plot completato")


# ============================================================================
# CELLA 9: Difference Maps
# ============================================================================
fig, axes = plt.subplots(1, len(techniques), figsize=(5 * len(techniques), 5))
if len(techniques) == 1:
    axes = [axes]

for idx, technique in enumerate(techniques):
    orig_noise, _ = original_results[technique]
    mod_noise, _ = modified_results[technique]
    
    # Differenza assoluta RAW
    diff = np.abs(orig_noise - mod_noise)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    # Visualizza
    diff_display = normalize_for_display(diff, 'minmax')
    axes[idx].imshow(diff_display, cmap='hot')
    axes[idx].set_title(f'{technique}\nMean Δ: {mean_diff:.4f}\nMax Δ: {max_diff:.4f}',
                       fontsize=11, fontweight='bold')
    axes[idx].axis('off')

plt.suptitle('Difference Maps: |Original Noise - Modified Noise|\n(Higher values = Better discrimination)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n✓ Difference maps completate")

# ============================================================================
# CELLA 10: Metrics Comparison
# ============================================================================
metrics_names = ['std', 'mean_abs', 'entropy', 'percentile_95']
metrics_labels = ['Standard Deviation', 'Mean Absolute', 'Entropy', '95th Percentile']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
    orig_values = [getattr(original_results[t][1], metric_name) for t in techniques]
    mod_values = [getattr(modified_results[t][1], metric_name) for t in techniques]
    
    x = np.arange(len(techniques))
    width = 0.35
    
    axes[idx].bar(x - width/2, orig_values, width, label='Original', alpha=0.8, color='steelblue')
    axes[idx].bar(x + width/2, mod_values, width, label='Modified', alpha=0.8, color='coral')
    
    # Aggiungi delta
    for i, (orig, mod) in enumerate(zip(orig_values, mod_values)):
        delta = abs(orig - mod)
        axes[idx].text(i, max(orig, mod) * 1.05, f'Δ={delta:.4f}', 
                     ha='center', fontsize=8, fontweight='bold')
    
    axes[idx].set_xlabel('Technique', fontsize=11)
    axes[idx].set_ylabel(metric_label, fontsize=11)
    axes[idx].set_title(f'{metric_label} (RAW metrics)', fontsize=12, fontweight='bold')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(techniques, rotation=45, ha='right')
    axes[idx].legend()
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Quantitative Metrics Comparison (calculated on RAW data)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✓ Metrics comparison completato")

# ============================================================================
# CELLA 11: Interpretazione Finale
# ============================================================================
print("\n" + "="*80)
print("📊 INTERPRETAZIONE RISULTATI")
print("="*80)

print("\n🎯 Cosa cercare:")
print("  ✅ Δ STD > 0.01 (buon segno)")
print("  ✅ Δ Entropy > 0.5 (buon segno)")
print("  ✅ Difference maps con valori alti (colori caldi)")
print("  ✅ Pattern visivi chiaramente diversi tra original e modified")

print("\n🏆 Tecnica migliore:")
best_technique = None
best_delta_std = 0

for technique in techniques:
    orig_metrics = original_results[technique][1]
    mod_metrics = modified_results[technique][1]
    delta_std = abs(orig_metrics.std - mod_metrics.std)
    
    if delta_std > best_delta_std:
        best_delta_std = delta_std
        best_technique = technique

print(f"  → {best_technique} (Δ STD: {best_delta_std:.5f})")

print("\n💡 Prossimi passi:")
print("  1. Se Δ STD > 0.01: La tecnica è promettente!")
print("  2. Testare su più coppie per conferma")
print("  3. Integrare nella pipeline di training come:")
print("     - Preprocessing (usa solo noise map)")
print("     - Dual-input (RGB + noise map)")
print("     - Multi-channel (concatena come canali)")

print("\n" + "="*80)
print("✓ ANALISI COMPLETATA")
print("="*80)
