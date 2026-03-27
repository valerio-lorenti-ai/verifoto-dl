# -*- coding: utf-8 -*-
"""
# 🔬 Forensic Noise Analysis - Memory Efficient Edition

Versione OTTIMIZZATA per analizzare molte coppie senza esaurire la RAM.

## 🎯 Obiettivo
Analizzare N coppie (anche 50+) su Colab senza crash di memoria.

## 💾 Ottimizzazioni
- ✅ Elaborazione streaming (coppia per coppia)
- ✅ Salva solo metriche, non immagini
- ✅ Garbage collection esplicito
- ✅ Salvataggio progressivo su disco
- ✅ Modalità lightweight batch
- ✅ Resize opzionale controllato

## 📋 Workflow
1. Carica e matcha dataset
2. Elabora coppia per coppia (streaming)
3. Salva solo metriche aggregate
4. Conserva poche coppie campione per visualizzazione
5. Genera report finale
"""

# ============================================================================
# CELLA 1: Installazione
# ============================================================================
!pip install -q opencv-python-headless pillow numpy matplotlib scipy

# ============================================================================
# CELLA 2: Import e Configurazione
# ============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from dataclasses import dataclass, asdict
from scipy import stats as scipy_stats
from pathlib import Path
from collections import defaultdict
import random
import warnings
import json
import gc  # Garbage collection
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
print("✓ Setup completato")

# ============================================================================
# CELLA 3: Configurazione Dataset e Memoria
# ============================================================================
# 🔧 CONFIGURA QUESTI PARAMETRI

# === PATH DATASET ===
ORIGINAL_DIR = "/content/drive/MyDrive/dataset/originali"
MODIFIED_DIR = "/content/drive/MyDrive/dataset/modificate"

# === ANALISI ===
NUM_PAIRS = 50  # Numero coppie da analizzare (ora puoi usare valori alti!)
SELECTION_MODE = "first"  # "first", "random", "specific"
SPECIFIC_IDS = []  # Solo per mode="specific"

# === MEMORIA ===
LIGHTWEIGHT_BATCH_MODE = True  # ✅ CONSIGLIATO per molte coppie
NUM_EXAMPLES_TO_STORE = 3  # Numero coppie da salvare per visualizzazione
MAX_IMAGE_SIZE = None  # Es. 512 per resize a 512x512, None = no resize

# === SALVATAGGIO ===
SAVE_METRICS_TO_FILE = True  # Salva metriche progressivamente
METRICS_FILE = "noise_analysis_metrics.json"

# === ANALYZER ===
MEDIAN_KERNEL = 5
GAUSSIAN_KERNEL = 5
GAUSSIAN_SIGMA = 1.5
ELA_QUALITY = 90

print(f"✓ Configurazione:")
print(f"  Coppie da analizzare: {NUM_PAIRS}")
print(f"  Modalità lightweight: {LIGHTWEIGHT_BATCH_MODE}")
print(f"  Esempi da salvare: {NUM_EXAMPLES_TO_STORE}")
print(f"  Max image size: {MAX_IMAGE_SIZE if MAX_IMAGE_SIZE else 'Original'}")
print(f"  Salvataggio su file: {SAVE_METRICS_TO_FILE}")


# ============================================================================
# CELLA 4: Funzioni Dataset (invariate)
# ============================================================================
def extract_id(filename):
    """Estrae ID (primi 8 caratteri) dal filename."""
    return Path(filename).stem[:8]

def load_dataset_pairs(original_dir, modified_dir):
    """Carica e matcha automaticamente le coppie."""
    original_dir = Path(original_dir)
    modified_dir = Path(modified_dir)
    
    original_files = list(original_dir.glob("*.jpg")) + list(original_dir.glob("*.jpeg")) + list(original_dir.glob("*.png"))
    modified_files = list(modified_dir.glob("*.jpg")) + list(modified_dir.glob("*.jpeg")) + list(modified_dir.glob("*.png"))
    
    original_dict = {extract_id(f.name): f for f in original_files}
    modified_dict = {extract_id(f.name): f for f in modified_files}
    
    common_ids = set(original_dict.keys()) & set(modified_dict.keys())
    pairs = [(id, original_dict[id], modified_dict[id]) for id in sorted(common_ids)]
    
    stats = {
        'total_original': len(original_files),
        'total_modified': len(modified_files),
        'matched_pairs': len(pairs),
        'unmatched_original': len(original_dict) - len(common_ids),
        'unmatched_modified': len(modified_dict) - len(common_ids)
    }
    
    return pairs, stats

def select_pairs(pairs, mode="first", num_pairs=10, specific_ids=None):
    """Seleziona N coppie."""
    if mode == "first":
        return pairs[:num_pairs]
    elif mode == "random":
        return random.sample(pairs, min(num_pairs, len(pairs)))
    elif mode == "specific":
        if not specific_ids:
            raise ValueError("specific_ids richiesto")
        pairs_dict = {id: (id, orig, mod) for id, orig, mod in pairs}
        return [pairs_dict[id] for id in specific_ids if id in pairs_dict]
    else:
        raise ValueError(f"Modalità sconosciuta: {mode}")

print("✓ Funzioni dataset definite")

# ============================================================================
# CELLA 5: Carica e Matcha Dataset
# ============================================================================
print("📂 Caricamento dataset...")
pairs, stats = load_dataset_pairs(ORIGINAL_DIR, MODIFIED_DIR)

print(f"\n📊 Statistiche:")
print(f"  Coppie matchate: {stats['matched_pairs']}")
print(f"  Originali senza match: {stats['unmatched_original']}")
print(f"  Modificate senza match: {stats['unmatched_modified']}")

selected_pairs = select_pairs(pairs, mode=SELECTION_MODE, num_pairs=NUM_PAIRS, 
                              specific_ids=SPECIFIC_IDS if SELECTION_MODE == "specific" else None)

print(f"\n🎯 Coppie selezionate: {len(selected_pairs)}")
print(f"\n📋 Preview (prime 5):")
for id, orig, mod in selected_pairs[:5]:
    print(f"  {id}: {orig.name} ↔ {mod.name}")
if len(selected_pairs) > 5:
    print(f"  ... e altre {len(selected_pairs) - 5}")

print("\n✓ Dataset pronto")


# ============================================================================
# CELLA 6: Classi Analisi (Memory Efficient)
# ============================================================================
@dataclass
class NoiseMetrics:
    """Metriche leggere (solo numeri, no array)."""
    std: float
    mean_abs: float
    range: float
    snr: float
    entropy: float
    percentile_95: float
    
    def to_dict(self):
        return asdict(self)

class MemoryEfficientNoiseAnalyzer:
    """Analyzer ottimizzato per memoria."""
    
    def __init__(self, median_kernel=5, gaussian_kernel=5, gaussian_sigma=1.5, 
                 ela_quality=90, max_size=None):
        self.median_kernel = median_kernel
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
        self.ela_quality = ela_quality
        self.max_size = max_size
    
    def load_and_resize(self, path):
        """Carica e opzionalmente ridimensiona."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Impossibile caricare: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Resize opzionale
        if self.max_size and max(img.shape[:2]) > self.max_size:
            h, w = img.shape[:2]
            scale = self.max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return img
    
    def ensure_same_shape(self, img1, img2):
        """Uniforma shape."""
        if img1.shape != img2.shape:
            h, w = img1.shape[:2]
            img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)
        return img2
    
    def median_noise_residual(self, image):
        """Median Residual - RAW."""
        img_uint8 = (image * 255).astype(np.uint8)
        denoised = cv2.medianBlur(img_uint8, self.median_kernel)
        denoised = denoised.astype(np.float32) / 255.0
        return image - denoised
    
    def gaussian_residual_noise(self, image):
        """Gaussian Residual - RAW."""
        denoised = cv2.GaussianBlur(image, (self.gaussian_kernel, self.gaussian_kernel), 
                                     self.gaussian_sigma)
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
    
    def analyze_pair_lightweight(self, orig_path, mod_path):
        """
        Analizza coppia in modalità LIGHTWEIGHT.
        
        Restituisce SOLO metriche, non immagini o noise maps.
        Questo è il metodo chiave per efficienza memoria.
        """
        # Carica
        original = self.load_and_resize(orig_path)
        modified = self.load_and_resize(mod_path)
        modified = self.ensure_same_shape(original, modified)
        
        # Analizza e calcola metriche
        techniques = ['Median Residual', 'Gaussian Residual', 'High-pass Filter', 'ELA']
        results = {}
        
        for technique in techniques:
            if technique == 'Median Residual':
                noise = self.median_noise_residual(original)
                noise_mod = self.median_noise_residual(modified)
            elif technique == 'Gaussian Residual':
                noise = self.gaussian_residual_noise(original)
                noise_mod = self.gaussian_residual_noise(modified)
            elif technique == 'High-pass Filter':
                noise = self.high_pass_filter(original)
                noise_mod = self.high_pass_filter(modified)
            else:  # ELA
                noise = self.error_level_analysis(original)
                noise_mod = self.error_level_analysis(modified)
            
            # Calcola metriche
            metrics_orig = self.compute_metrics(noise)
            metrics_mod = self.compute_metrics(noise_mod)
            
            results[technique] = {
                'original': metrics_orig,
                'modified': metrics_mod
            }
            
            # Libera memoria immediatamente
            del noise, noise_mod
        
        # Libera immagini
        del original, modified
        
        return results

print("✓ MemoryEfficientNoiseAnalyzer definito")


# ============================================================================
# CELLA 7: Analisi Streaming (Memory Efficient)
# ============================================================================
print("🔬 Inizio analisi STREAMING (memory efficient)...")
print(f"  Modalità lightweight: {LIGHTWEIGHT_BATCH_MODE}")
print(f"  Coppie da analizzare: {len(selected_pairs)}")

# Inizializza analyzer
analyzer = MemoryEfficientNoiseAnalyzer(
    median_kernel=MEDIAN_KERNEL,
    gaussian_kernel=GAUSSIAN_KERNEL,
    gaussian_sigma=GAUSSIAN_SIGMA,
    ela_quality=ELA_QUALITY,
    max_size=MAX_IMAGE_SIZE
)

# Storage LEGGERO (solo metriche)
aggregate_metrics = defaultdict(lambda: {'original': [], 'modified': []})
pair_results = []  # Solo ID e metriche, NO immagini
examples_stored = []  # Poche coppie complete per visualizzazione

# File per salvataggio progressivo
if SAVE_METRICS_TO_FILE:
    metrics_log = []

# Elabora coppia per coppia (STREAMING)
successful = 0
for idx, (pair_id, orig_path, mod_path) in enumerate(selected_pairs, 1):
    print(f"\rProcessing {idx}/{len(selected_pairs)}: {pair_id}", end="", flush=True)
    
    try:
        # Analizza in modalità lightweight
        results = analyzer.analyze_pair_lightweight(orig_path, mod_path)
        
        # Salva solo metriche (LEGGERO)
        pair_results.append({
            'pair_id': pair_id,
            'metrics': results
        })
        
        # Aggrega metriche
        for technique, metrics_dict in results.items():
            aggregate_metrics[technique]['original'].append(metrics_dict['original'])
            aggregate_metrics[technique]['modified'].append(metrics_dict['modified'])
        
        # Salva su file progressivamente
        if SAVE_METRICS_TO_FILE:
            metrics_log.append({
                'pair_id': pair_id,
                'metrics': {tech: {
                    'original': m['original'].to_dict(),
                    'modified': m['modified'].to_dict()
                } for tech, m in results.items()}
            })
        
        # Salva poche coppie complete per visualizzazione
        if len(examples_stored) < NUM_EXAMPLES_TO_STORE:
            # Solo per questi pochi esempi, carica immagini complete
            original = analyzer.load_and_resize(orig_path)
            modified = analyzer.load_and_resize(mod_path)
            modified = analyzer.ensure_same_shape(original, modified)
            
            examples_stored.append({
                'pair_id': pair_id,
                'original': original,
                'modified': modified,
                'metrics': results
            })
        
        successful += 1
        
        # Garbage collection esplicito ogni 10 coppie
        if idx % 10 == 0:
            gc.collect()
        
    except Exception as e:
        print(f"\n  ✗ ERRORE {pair_id}: {e}")
        continue

print(f"\n\n✓ Analisi completata: {successful}/{len(selected_pairs)} coppie")

# Salva metriche su file
if SAVE_METRICS_TO_FILE:
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f"✓ Metriche salvate in: {METRICS_FILE}")

# Garbage collection finale
gc.collect()
print("✓ Memoria liberata")


# ============================================================================
# CELLA 8: Visualizzazione Esempi Campione
# ============================================================================
if len(examples_stored) > 0:
    print(f"\n📊 Visualizzazione {len(examples_stored)} esempi campione...")
    
    def normalize_for_display(noise, method="centered"):
        """Normalizza SOLO per visualizzazione."""
        if method == "centered":
            abs_max = max(abs(noise.min()), abs(noise.max()))
            if abs_max < 1e-8:
                return np.ones_like(noise) * 0.5
            normalized = (noise / (2 * abs_max)) + 0.5
            return np.clip(normalized, 0, 1)
        else:
            range_val = noise.max() - noise.min()
            if range_val < 1e-8:
                return np.zeros_like(noise)
            return (noise - noise.min()) / range_val
    
    # Mostra solo primo esempio (per non sovraccaricare)
    example = examples_stored[0]
    pair_id = example['pair_id']
    original = example['original']
    modified = example['modified']
    
    # Ricalcola noise maps solo per visualizzazione
    techniques = ['Median Residual', 'Gaussian Residual', 'High-pass Filter', 'ELA']
    
    fig, axes = plt.subplots(2, len(techniques) + 1, figsize=(4 * (len(techniques) + 1), 8))
    
    # Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f'ORIGINAL\n{pair_id}', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Modified
    axes[1, 0].imshow(modified)
    axes[1, 0].set_title(f'MODIFIED\n{pair_id}', fontsize=11, fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    # Noise maps
    for col_idx, technique in enumerate(techniques, 1):
        if technique == 'Median Residual':
            noise_orig = analyzer.median_noise_residual(original)
            noise_mod = analyzer.median_noise_residual(modified)
            norm_method = 'centered'
            cmap = 'RdBu_r'
        elif technique == 'Gaussian Residual':
            noise_orig = analyzer.gaussian_residual_noise(original)
            noise_mod = analyzer.gaussian_residual_noise(modified)
            norm_method = 'centered'
            cmap = 'RdBu_r'
        elif technique == 'High-pass Filter':
            noise_orig = analyzer.high_pass_filter(original)
            noise_mod = analyzer.high_pass_filter(modified)
            norm_method = 'centered'
            cmap = 'RdBu_r'
        else:  # ELA
            noise_orig = analyzer.error_level_analysis(original)
            noise_mod = analyzer.error_level_analysis(modified)
            norm_method = 'minmax'
            cmap = 'hot'
        
        # Original noise
        display_orig = normalize_for_display(noise_orig, norm_method)
        axes[0, col_idx].imshow(display_orig, cmap=cmap)
        axes[0, col_idx].set_title(technique, fontsize=10)
        axes[0, col_idx].axis('off')
        
        # Modified noise
        display_mod = normalize_for_display(noise_mod, norm_method)
        axes[1, col_idx].imshow(display_mod, cmap=cmap)
        axes[1, col_idx].set_title(technique, fontsize=10)
        axes[1, col_idx].axis('off')
        
        # Libera
        del noise_orig, noise_mod, display_orig, display_mod
    
    plt.suptitle(f'Example: {pair_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.close()  # Chiudi figura per liberare memoria
    
    print(f"✓ Esempio visualizzato")
    
    # Libera esempi dopo visualizzazione se in lightweight mode
    if LIGHTWEIGHT_BATCH_MODE:
        del examples_stored
        gc.collect()
        print("✓ Esempi liberati dalla memoria")
else:
    print("\n⚠️  Nessun esempio salvato per visualizzazione")


# ============================================================================
# CELLA 9: Metriche Aggregate Finali
# ============================================================================
print("\n" + "="*80)
print("📊 METRICHE AGGREGATE SU TUTTE LE COPPIE")
print("="*80)
print(f"Coppie analizzate: {successful}")

techniques = list(aggregate_metrics.keys())

for technique in techniques:
    print(f"\n{technique}:")
    print(f"{'─'*80}")
    
    orig_metrics_list = aggregate_metrics[technique]['original']
    mod_metrics_list = aggregate_metrics[technique]['modified']
    
    # Statistiche
    orig_std_mean = np.mean([m.std for m in orig_metrics_list])
    orig_std_std = np.std([m.std for m in orig_metrics_list])
    mod_std_mean = np.mean([m.std for m in mod_metrics_list])
    mod_std_std = np.std([m.std for m in mod_metrics_list])
    
    orig_entropy_mean = np.mean([m.entropy for m in orig_metrics_list])
    mod_entropy_mean = np.mean([m.entropy for m in mod_metrics_list])
    
    delta_std_mean = abs(orig_std_mean - mod_std_mean)
    delta_entropy_mean = abs(orig_entropy_mean - mod_entropy_mean)
    
    print(f"  STD:")
    print(f"    Original  → {orig_std_mean:.5f} ± {orig_std_std:.5f}")
    print(f"    Modified  → {mod_std_mean:.5f} ± {mod_std_std:.5f}")
    print(f"    Δ: {delta_std_mean:.5f}")
    
    print(f"  Entropy:")
    print(f"    Original  → {orig_entropy_mean:.2f}")
    print(f"    Modified  → {mod_entropy_mean:.2f}")
    print(f"    Δ: {delta_entropy_mean:.2f}")
    
    # Cohen's d
    pooled_std = np.sqrt((np.var([m.std for m in orig_metrics_list]) + 
                         np.var([m.std for m in mod_metrics_list])) / 2)
    cohens_d = abs(orig_std_mean - mod_std_mean) / pooled_std if pooled_std > 1e-10 else 0.0
    
    print(f"  Cohen's d: {cohens_d:.4f}", end="")
    if cohens_d > 0.8:
        print(" ✅ (Large)")
    elif cohens_d > 0.5:
        print(" ✅ (Medium)")
    elif cohens_d > 0.2:
        print(" ⚠️  (Small)")
    else:
        print(" ❌ (Negligible)")
    
    # t-test
    if len(orig_metrics_list) > 1 and len(mod_metrics_list) > 1:
        orig_stds = [m.std for m in orig_metrics_list]
        mod_stds = [m.std for m in mod_metrics_list]
        t_stat, p_value = scipy_stats.ttest_ind(orig_stds, mod_stds)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"  p-value: {p_value:.6f} {sig}")
    
    # Interpretazione
    if delta_std_mean > 0.01:
        print(f"  ✅ Δ STD > 0.01: Buona separazione!")
    if delta_entropy_mean > 0.5:
        print(f"  ✅ Δ Entropy > 0.5: Pattern diversi!")

print("\n" + "="*80)

# ============================================================================
# CELLA 10: Visualizzazioni Aggregate
# ============================================================================
print("\n📊 Generazione grafici aggregate...")

# Prepara dati
orig_std_means = []
mod_std_means = []
orig_entropy_means = []
mod_entropy_means = []
cohens_d_values = []

for technique in techniques:
    orig_metrics_list = aggregate_metrics[technique]['original']
    mod_metrics_list = aggregate_metrics[technique]['modified']
    
    orig_std_mean = np.mean([m.std for m in orig_metrics_list])
    mod_std_mean = np.mean([m.std for m in mod_metrics_list])
    orig_entropy_mean = np.mean([m.entropy for m in orig_metrics_list])
    mod_entropy_mean = np.mean([m.entropy for m in mod_metrics_list])
    
    orig_std_means.append(orig_std_mean)
    mod_std_means.append(mod_std_mean)
    orig_entropy_means.append(orig_entropy_mean)
    mod_entropy_means.append(mod_entropy_mean)
    
    pooled_std = np.sqrt((np.var([m.std for m in orig_metrics_list]) + 
                         np.var([m.std for m in mod_metrics_list])) / 2)
    cohens_d = abs(orig_std_mean - mod_std_mean) / pooled_std if pooled_std > 1e-10 else 0.0
    cohens_d_values.append(cohens_d)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# STD
x = np.arange(len(techniques))
width = 0.35
axes[0].bar(x - width/2, orig_std_means, width, label='Original', alpha=0.8, color='steelblue')
axes[0].bar(x + width/2, mod_std_means, width, label='Modified', alpha=0.8, color='coral')
for i, (orig, mod) in enumerate(zip(orig_std_means, mod_std_means)):
    delta = abs(orig - mod)
    axes[0].text(i, max(orig, mod) * 1.05, f'Δ={delta:.4f}', ha='center', fontsize=9, fontweight='bold')
axes[0].set_xlabel('Technique', fontsize=11)
axes[0].set_ylabel('STD (RAW)', fontsize=11)
axes[0].set_title('Standard Deviation', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(techniques, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Entropy
axes[1].bar(x - width/2, orig_entropy_means, width, label='Original', alpha=0.8, color='steelblue')
axes[1].bar(x + width/2, mod_entropy_means, width, label='Modified', alpha=0.8, color='coral')
for i, (orig, mod) in enumerate(zip(orig_entropy_means, mod_entropy_means)):
    delta = abs(orig - mod)
    axes[1].text(i, max(orig, mod) * 1.05, f'Δ={delta:.2f}', ha='center', fontsize=9, fontweight='bold')
axes[1].set_xlabel('Technique', fontsize=11)
axes[1].set_ylabel('Entropy (RAW)', fontsize=11)
axes[1].set_title('Entropy', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(techniques, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Cohen's d
colors = ['green' if d > 0.8 else 'orange' if d > 0.5 else 'red' for d in cohens_d_values]
axes[2].bar(x, cohens_d_values, color=colors, alpha=0.8)
axes[2].axhline(y=0.8, color='green', linestyle='--', label='Large (>0.8)', alpha=0.5)
axes[2].axhline(y=0.5, color='orange', linestyle='--', label='Medium (>0.5)', alpha=0.5)
axes[2].axhline(y=0.2, color='red', linestyle='--', label='Small (>0.2)', alpha=0.5)
axes[2].set_xlabel('Technique', fontsize=11)
axes[2].set_ylabel("Cohen's d", fontsize=11)
axes[2].set_title("Effect Size", fontsize=12, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(techniques, rotation=45, ha='right')
axes[2].legend(fontsize=8)
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle(f'Aggregate Metrics ({successful} pairs)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
plt.close()  # Chiudi per liberare memoria

print("✓ Grafici generati")


# ============================================================================
# CELLA 11: Interpretazione Finale
# ============================================================================
print("\n" + "="*80)
print("🎯 INTERPRETAZIONE FINALE")
print("="*80)

# Identifica tecnica migliore
best_technique = None
best_cohens_d = 0

for i, technique in enumerate(techniques):
    if cohens_d_values[i] > best_cohens_d:
        best_cohens_d = cohens_d_values[i]
        best_technique = technique

print(f"\n🏆 Tecnica migliore: {best_technique}")
print(f"   Cohen's d: {best_cohens_d:.4f}")
print(f"   Δ STD: {abs(orig_std_means[techniques.index(best_technique)] - mod_std_means[techniques.index(best_technique)]):.5f}")

if best_cohens_d > 0.8:
    print(f"   ✅ Large effect - OTTIMA per training!")
elif best_cohens_d > 0.5:
    print(f"   ✅ Medium effect - Buona per training")
elif best_cohens_d > 0.2:
    print(f"   ⚠️  Small effect - Potrebbe funzionare")
else:
    print(f"   ❌ Negligible - Non raccomandato")

print(f"\n💡 Raccomandazioni:")
print(f"  1. Testare {best_technique} su più coppie per conferma")
print(f"  2. Integrare come:")
print(f"     - Preprocessing (solo noise map)")
print(f"     - Dual-input (RGB + noise)")
print(f"     - Multi-channel (concatena)")

print(f"\n📊 Riepilogo:")
print(f"  Coppie analizzate: {successful}")
print(f"  Tecnica migliore: {best_technique}")
print(f"  Effect size: {best_cohens_d:.4f}")
print(f"  Metriche salvate: {METRICS_FILE if SAVE_METRICS_TO_FILE else 'No'}")

print("\n💾 Ottimizzazioni memoria applicate:")
print(f"  ✅ Elaborazione streaming")
print(f"  ✅ Solo metriche in RAM")
print(f"  ✅ Garbage collection esplicito")
print(f"  ✅ Salvataggio progressivo su disco")
print(f"  ✅ Esempi limitati: {NUM_EXAMPLES_TO_STORE}")
if MAX_IMAGE_SIZE:
    print(f"  ✅ Resize a {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}")

print("\n" + "="*80)
print("✓ ANALISI COMPLETATA")
print("="*80)

# Garbage collection finale
gc.collect()
print("\n✓ Memoria finale liberata")
