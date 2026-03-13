# -*- coding: utf-8 -*-
"""
# 🔬 Forensic Noise Analysis - Dataset Analysis

Analisi forense su MULTIPLE coppie dal dataset per identificare la tecnica migliore.

## 🎯 Obiettivo
Analizzare N coppie di immagini (original vs AI-edited) dal dataset per capire quale 
tecnica di noise detection funziona meglio in modo CONSISTENTE.

## 📋 Workflow
1. Carica dataset da cartelle
2. Matcha automaticamente le coppie (primi 8 caratteri filename)
3. Seleziona N coppie da analizzare
4. Applica 4 tecniche forensi
5. Visualizza risultati per ogni coppia
6. Mostra metriche aggregate su tutte le coppie

## ⚠️ PRINCIPIO FORENSE
- ✅ Residui sempre RAW
- ✅ Normalizzazioni SOLO per visualizzazione
- ✅ Gestione robusta di shape diverse
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
from dataclasses import dataclass
from scipy import stats as scipy_stats  # Rinominato per evitare conflitti
from pathlib import Path
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
print("✓ Setup completato")

# ============================================================================
# CELLA 3: Configurazione Dataset
# ============================================================================
# 🔧 CONFIGURA QUESTI PARAMETRI

# Path alle cartelle (modifica con i tuoi path)
ORIGINAL_DIR = "/content/drive/MyDrive/dataset/originali"  # Cartella immagini originali
MODIFIED_DIR = "/content/drive/MyDrive/dataset/modificate"  # Cartella immagini modificate

# Numero di coppie da analizzare
NUM_PAIRS = 10  # Cambia questo numero (es. 5, 10, 20)

# Modalità selezione coppie
SELECTION_MODE = "first"  # "first" = prime N, "random" = N casuali, "specific" = lista ID

# Se SELECTION_MODE = "specific", specifica gli ID (primi 8 caratteri)
SPECIFIC_IDS = ["20260227", "20260228"]  # Esempio

# Parametri analyzer
MEDIAN_KERNEL = 5
GAUSSIAN_KERNEL = 5
GAUSSIAN_SIGMA = 1.5
ELA_QUALITY = 90

print(f"✓ Configurazione:")
print(f"  Original dir: {ORIGINAL_DIR}")
print(f"  Modified dir: {MODIFIED_DIR}")
print(f"  Numero coppie: {NUM_PAIRS}")
print(f"  Modalità: {SELECTION_MODE}")


# ============================================================================
# CELLA 4: Funzioni di Matching Dataset
# ============================================================================
def extract_id(filename):
    """Estrae ID (primi 8 caratteri) dal filename."""
    return Path(filename).stem[:8]

def load_dataset_pairs(original_dir, modified_dir):
    """
    Carica e matcha automaticamente le coppie dal dataset.
    
    Returns:
        pairs: Lista di tuple (id, original_path, modified_path)
        stats: Dict con statistiche
    """
    original_dir = Path(original_dir)
    modified_dir = Path(modified_dir)
    
    # Trova tutti i file
    original_files = list(original_dir.glob("*.jpg")) + list(original_dir.glob("*.jpeg")) + list(original_dir.glob("*.png"))
    modified_files = list(modified_dir.glob("*.jpg")) + list(modified_dir.glob("*.jpeg")) + list(modified_dir.glob("*.png"))
    
    # Crea dizionari ID -> path
    original_dict = {extract_id(f.name): f for f in original_files}
    modified_dict = {extract_id(f.name): f for f in modified_files}
    
    # Trova coppie valide
    common_ids = set(original_dict.keys()) & set(modified_dict.keys())
    pairs = [(id, original_dict[id], modified_dict[id]) for id in sorted(common_ids)]
    
    # Statistiche
    stats = {
        'total_original': len(original_files),
        'total_modified': len(modified_files),
        'matched_pairs': len(pairs),
        'unmatched_original': len(original_dict) - len(common_ids),
        'unmatched_modified': len(modified_dict) - len(common_ids),
        'unmatched_original_ids': sorted(set(original_dict.keys()) - common_ids),
        'unmatched_modified_ids': sorted(set(modified_dict.keys()) - common_ids)
    }
    
    return pairs, stats

def select_pairs(pairs, mode="first", num_pairs=10, specific_ids=None):
    """
    Seleziona N coppie secondo la modalità specificata.
    
    Args:
        pairs: Lista di tuple (id, orig_path, mod_path)
        mode: "first", "random", "specific"
        num_pairs: Numero di coppie da selezionare
        specific_ids: Lista di ID specifici (per mode="specific")
    
    Returns:
        selected_pairs: Lista di coppie selezionate
    """
    if mode == "first":
        return pairs[:num_pairs]
    elif mode == "random":
        return random.sample(pairs, min(num_pairs, len(pairs)))
    elif mode == "specific":
        if not specific_ids:
            raise ValueError("specific_ids richiesto per mode='specific'")
        pairs_dict = {id: (id, orig, mod) for id, orig, mod in pairs}
        return [pairs_dict[id] for id in specific_ids if id in pairs_dict]
    else:
        raise ValueError(f"Modalità sconosciuta: {mode}")

print("✓ Funzioni matching definite")

# ============================================================================
# CELLA 5: Carica e Matcha Dataset
# ============================================================================
print("📂 Caricamento dataset...")
print(f"  Original: {ORIGINAL_DIR}")
print(f"  Modified: {MODIFIED_DIR}")

pairs, stats = load_dataset_pairs(ORIGINAL_DIR, MODIFIED_DIR)

print(f"\n📊 Statistiche Dataset:")
print(f"  Immagini originali trovate: {stats['total_original']}")
print(f"  Immagini modificate trovate: {stats['total_modified']}")
print(f"  ✅ Coppie valide matchate: {stats['matched_pairs']}")
print(f"  ⚠️  Originali senza match: {stats['unmatched_original']}")
print(f"  ⚠️  Modificate senza match: {stats['unmatched_modified']}")

if stats['unmatched_original'] > 0:
    print(f"\n  ID originali senza match: {stats['unmatched_original_ids'][:5]}...")
if stats['unmatched_modified'] > 0:
    print(f"  ID modificate senza match: {stats['unmatched_modified_ids'][:5]}...")

# Seleziona coppie
selected_pairs = select_pairs(pairs, mode=SELECTION_MODE, num_pairs=NUM_PAIRS, 
                              specific_ids=SPECIFIC_IDS if SELECTION_MODE == "specific" else None)

print(f"\n🎯 Coppie selezionate per analisi: {len(selected_pairs)}")
print(f"  Modalità: {SELECTION_MODE}")

# Preview coppie
print(f"\n📋 Preview coppie:")
print(f"{'ID':<12} {'Original':<40} {'Modified':<40}")
print("-" * 95)
for id, orig, mod in selected_pairs[:10]:  # Mostra prime 10
    print(f"{id:<12} {orig.name:<40} {mod.name:<40}")
if len(selected_pairs) > 10:
    print(f"... e altre {len(selected_pairs) - 10} coppie")

print("\n✓ Dataset caricato e matchato")


# ============================================================================
# CELLA 6: Classi di Analisi (con gestione shape robusta)
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
    """Analisi forense - Gestione robusta di shape diverse."""
    
    def __init__(self, median_kernel=5, gaussian_kernel=5, gaussian_sigma=1.5, ela_quality=90):
        self.median_kernel = median_kernel
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
        self.ela_quality = ela_quality
    
    def load_image(self, path):
        """Carica immagine RGB float [0,1]."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Impossibile caricare: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0
    
    def ensure_same_shape(self, img1, img2):
        """
        Assicura che due immagini abbiano la stessa shape.
        Ridimensiona img2 per matchare img1 se necessario.
        """
        if img1.shape != img2.shape:
            # Ridimensiona img2 per matchare img1
            h, w = img1.shape[:2]
            img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)
            print(f"    ⚠️  Shape mismatch: {img2.shape} -> {img2_resized.shape}")
            return img2_resized
        return img2
    
    def median_noise_residual(self, image):
        """Median Residual - RAW (può essere negativo)."""
        img_uint8 = (image * 255).astype(np.uint8)
        denoised = cv2.medianBlur(img_uint8, self.median_kernel)
        denoised = denoised.astype(np.float32) / 255.0
        return image - denoised  # RAW
    
    def gaussian_residual_noise(self, image):
        """Gaussian Residual - RAW (può essere negativo)."""
        denoised = cv2.GaussianBlur(image, (self.gaussian_kernel, self.gaussian_kernel), 
                                     self.gaussian_sigma)
        return image - denoised  # RAW
    
    def high_pass_filter(self, image):
        """High-pass Filter - RAW Laplacian."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian = laplacian.astype(np.float32) / 255.0
        
        # Converti a RGB per consistenza
        if len(image.shape) == 3:
            laplacian = np.stack([laplacian] * 3, axis=-1)
        
        return laplacian  # RAW
    
    def error_level_analysis(self, image):
        """ELA - RAW absolute difference."""
        img_uint8 = (image * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=self.ela_quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        compressed = np.array(compressed_img).astype(np.float32) / 255.0
        return np.abs(image - compressed)  # RAW
    
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
        
        # Calcola metriche
        results_with_metrics = {}
        for technique, noise in results.items():
            metrics = self.compute_metrics(noise)
            results_with_metrics[technique] = (noise, metrics)
        
        return results_with_metrics

print("✓ NoiseAnalyzer definito (con gestione shape robusta)")


# ============================================================================
# CELLA 7: Funzioni Visualizzazione (CORRETTE)
# ============================================================================
def normalize_for_display(noise, method="centered"):
    """
    Normalizza SOLO per visualizzazione.
    
    IMPORTANTE: Questa funzione NON altera i dati RAW, serve solo per rendering.
    """
    if method == "centered":
        # Per residui che possono essere negativi
        abs_max = max(abs(noise.min()), abs(noise.max()))
        if abs_max < 1e-8:
            return np.ones_like(noise) * 0.5
        normalized = (noise / (2 * abs_max)) + 0.5
        normalized = np.clip(normalized, 0, 1)
    else:  # minmax
        # Per valori sempre positivi (es. ELA)
        range_val = noise.max() - noise.min()
        if range_val < 1e-8:
            return np.zeros_like(noise)
        normalized = (noise - noise.min()) / range_val
    
    return normalized

def to_grayscale_display(noise, method="centered"):
    """Converte a grayscale per visualizzazione neutra."""
    normalized = normalize_for_display(noise, method)
    
    if len(normalized.shape) == 3:
        # Converti a grayscale
        gray = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return gray.astype(np.float32) / 255.0
    
    return normalized

def plot_single_pair_comparison(original, modified, original_results, modified_results, 
                                pair_id, save_path=None):
    """
    Visualizza confronto per una singola coppia.
    
    CORRETTO: Gestisce correttamente la normalizzazione per visualizzazione.
    """
    techniques = list(original_results.keys())
    n_cols = len(techniques) * 2 + 1
    
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    
    # Row 1: Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f'ORIGINAL\n{pair_id}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    col_idx = 1
    for technique in techniques:
        noise, metrics = original_results[technique]
        
        # Determina metodo normalizzazione
        if 'Residual' in technique or 'High-pass' in technique:
            norm_method = 'centered'
            cmap = 'RdBu_r'
        else:  # ELA
            norm_method = 'minmax'
            cmap = 'hot'
        
        # Visualizzazione neutra (grayscale)
        gray = to_grayscale_display(noise, norm_method)
        axes[0, col_idx].imshow(gray, cmap='gray', vmin=0, vmax=1)
        axes[0, col_idx].set_title(f'{technique}\n(Neutral)', fontsize=9)
        axes[0, col_idx].axis('off')
        
        # Visualizzazione enfatizzata (colormap)
        colored = normalize_for_display(noise, norm_method)
        axes[0, col_idx + 1].imshow(colored, cmap=cmap)
        axes[0, col_idx + 1].set_title(f'{technique}\n(Enhanced)\nSTD:{metrics.std:.4f}', 
                                      fontsize=9)
        axes[0, col_idx + 1].axis('off')
        
        col_idx += 2
    
    # Row 2: Modified
    axes[1, 0].imshow(modified)
    axes[1, 0].set_title(f'MODIFIED\n{pair_id}', fontsize=12, fontweight='bold', color='red')
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
        
        # Neutra
        gray = to_grayscale_display(noise, norm_method)
        axes[1, col_idx].imshow(gray, cmap='gray', vmin=0, vmax=1)
        axes[1, col_idx].set_title(f'{technique}\n(Neutral)', fontsize=9)
        axes[1, col_idx].axis('off')
        
        # Enfatizzata
        colored = normalize_for_display(noise, norm_method)
        axes[1, col_idx + 1].imshow(colored, cmap=cmap)
        axes[1, col_idx + 1].set_title(f'{technique}\n(Enhanced)\nSTD:{metrics.std:.4f}', 
                                      fontsize=9)
        axes[1, col_idx + 1].axis('off')
        
        col_idx += 2
    
    plt.suptitle(f'Forensic Noise Analysis - Pair {pair_id}', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

print("✓ Funzioni visualizzazione definite (CORRETTE)")


# ============================================================================
# CELLA 8: Analisi su Tutte le Coppie Selezionate
# ============================================================================
print("🔬 Inizio analisi su tutte le coppie selezionate...")
print(f"  Numero coppie: {len(selected_pairs)}")

# Inizializza analyzer
analyzer = NoiseAnalyzer(
    median_kernel=MEDIAN_KERNEL,
    gaussian_kernel=GAUSSIAN_KERNEL,
    gaussian_sigma=GAUSSIAN_SIGMA,
    ela_quality=ELA_QUALITY
)

# Storage risultati
all_results = []
aggregate_metrics = defaultdict(lambda: {'original': [], 'modified': []})

# Analizza ogni coppia
for idx, (pair_id, orig_path, mod_path) in enumerate(selected_pairs, 1):
    print(f"\n{'─'*80}")
    print(f"Coppia {idx}/{len(selected_pairs)}: {pair_id}")
    print(f"{'─'*80}")
    
    try:
        # Carica immagini
        print(f"  📂 Caricamento...")
        original = analyzer.load_image(orig_path)
        modified = analyzer.load_image(mod_path)
        
        # Assicura stessa shape
        modified = analyzer.ensure_same_shape(original, modified)
        
        print(f"    Original: {original.shape}")
        print(f"    Modified: {modified.shape}")
        
        # Analizza
        print(f"  🔬 Applicazione tecniche forensi...")
        original_results = analyzer.analyze_image(original)
        modified_results = analyzer.analyze_image(modified)
        
        # Salva risultati
        all_results.append({
            'pair_id': pair_id,
            'original_path': orig_path,
            'modified_path': mod_path,
            'original': original,
            'modified': modified,
            'original_results': original_results,
            'modified_results': modified_results
        })
        
        # Aggrega metriche
        for technique in original_results.keys():
            orig_metrics = original_results[technique][1]
            mod_metrics = modified_results[technique][1]
            aggregate_metrics[technique]['original'].append(orig_metrics)
            aggregate_metrics[technique]['modified'].append(mod_metrics)
        
        # Mostra metriche per questa coppia
        print(f"\n  📊 Metriche RAW:")
        for technique in original_results.keys():
            orig_metrics = original_results[technique][1]
            mod_metrics = modified_results[technique][1]
            delta_std = abs(orig_metrics.std - mod_metrics.std)
            
            print(f"    {technique}:")
            print(f"      Orig STD: {orig_metrics.std:.5f} | Mod STD: {mod_metrics.std:.5f} | Δ: {delta_std:.5f}")
            
            if delta_std > 0.01:
                print(f"      ✅ Δ STD > 0.01")
        
        print(f"\n  ✓ Coppia {pair_id} completata")
        
    except Exception as e:
        print(f"  ✗ ERRORE coppia {pair_id}: {e}")
        continue

print(f"\n{'='*80}")
print(f"✓ Analisi completata su {len(all_results)}/{len(selected_pairs)} coppie")
print(f"{'='*80}")


# ============================================================================
# CELLA 9: Visualizzazione Coppie (Prime 3)
# ============================================================================
print("\n📊 Visualizzazione prime 3 coppie...")

num_to_show = min(3, len(all_results))

for i in range(num_to_show):
    result = all_results[i]
    print(f"\nCoppia {i+1}: {result['pair_id']}")
    
    plot_single_pair_comparison(
        result['original'],
        result['modified'],
        result['original_results'],
        result['modified_results'],
        result['pair_id']
    )

print(f"\n✓ Visualizzate {num_to_show} coppie")

# ============================================================================
# CELLA 10: Metriche Aggregate su Tutte le Coppie
# ============================================================================
print("\n" + "="*80)
print("📊 METRICHE AGGREGATE SU TUTTE LE COPPIE")
print("="*80)
print(f"Numero coppie analizzate: {len(all_results)}")

for technique in aggregate_metrics.keys():
    print(f"\n{technique}:")
    print(f"{'─'*80}")
    
    orig_metrics_list = aggregate_metrics[technique]['original']
    mod_metrics_list = aggregate_metrics[technique]['modified']
    
    # Calcola statistiche
    orig_std_mean = np.mean([m.std for m in orig_metrics_list])
    orig_std_std = np.std([m.std for m in orig_metrics_list])
    mod_std_mean = np.mean([m.std for m in mod_metrics_list])
    mod_std_std = np.std([m.std for m in mod_metrics_list])
    
    orig_entropy_mean = np.mean([m.entropy for m in orig_metrics_list])
    mod_entropy_mean = np.mean([m.entropy for m in mod_metrics_list])
    
    delta_std_mean = abs(orig_std_mean - mod_std_mean)
    delta_entropy_mean = abs(orig_entropy_mean - mod_entropy_mean)
    
    print(f"  STD:")
    print(f"    Original class  → Mean: {orig_std_mean:.5f} ± {orig_std_std:.5f}")
    print(f"    Modified class  → Mean: {mod_std_mean:.5f} ± {mod_std_std:.5f}")
    print(f"    Δ Mean: {delta_std_mean:.5f}")
    
    print(f"  Entropy:")
    print(f"    Original class  → Mean: {orig_entropy_mean:.2f}")
    print(f"    Modified class  → Mean: {mod_entropy_mean:.2f}")
    print(f"    Δ Mean: {delta_entropy_mean:.2f}")
    
    # Cohen's d
    pooled_std = np.sqrt((np.var([m.std for m in orig_metrics_list]) + 
                         np.var([m.std for m in mod_metrics_list])) / 2)
    if pooled_std > 1e-10:
        cohens_d = abs(orig_std_mean - mod_std_mean) / pooled_std
    else:
        cohens_d = 0.0
    
    print(f"  Cohen's d: {cohens_d:.4f}", end="")
    if cohens_d > 0.8:
        print(" ✅ (Large effect)")
    elif cohens_d > 0.5:
        print(" ✅ (Medium effect)")
    elif cohens_d > 0.2:
        print(" ⚠️  (Small effect)")
    else:
        print(" ❌ (Negligible)")
    
    # t-test
    orig_stds = [m.std for m in orig_metrics_list]
    mod_stds = [m.std for m in mod_metrics_list]
    if len(orig_stds) > 1 and len(mod_stds) > 1:
        t_stat, p_value = scipy_stats.ttest_ind(orig_stds, mod_stds)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"  t-test p-value: {p_value:.6f} {sig}")
    
    # Interpretazione
    if delta_std_mean > 0.01:
        print(f"  ✅ Δ STD > 0.01: Buona separazione!")
    if delta_entropy_mean > 0.5:
        print(f"  ✅ Δ Entropy > 0.5: Pattern diversi!")

print("\n" + "="*80)


# ============================================================================
# CELLA 11: Visualizzazione Metriche Aggregate
# ============================================================================
print("\n📊 Visualizzazione metriche aggregate...")

techniques = list(aggregate_metrics.keys())

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
    
    # Cohen's d
    pooled_std = np.sqrt((np.var([m.std for m in orig_metrics_list]) + 
                         np.var([m.std for m in mod_metrics_list])) / 2)
    if pooled_std > 1e-10:
        cohens_d = abs(orig_std_mean - mod_std_mean) / pooled_std
    else:
        cohens_d = 0.0
    cohens_d_values.append(cohens_d)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# STD Comparison
x = np.arange(len(techniques))
width = 0.35
axes[0].bar(x - width/2, orig_std_means, width, label='Original', alpha=0.8, color='steelblue')
axes[0].bar(x + width/2, mod_std_means, width, label='Modified', alpha=0.8, color='coral')
for i, (orig, mod) in enumerate(zip(orig_std_means, mod_std_means)):
    delta = abs(orig - mod)
    axes[0].text(i, max(orig, mod) * 1.05, f'Δ={delta:.4f}', ha='center', fontsize=9, fontweight='bold')
axes[0].set_xlabel('Technique', fontsize=11)
axes[0].set_ylabel('STD (RAW)', fontsize=11)
axes[0].set_title('Standard Deviation Comparison', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(techniques, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Entropy Comparison
axes[1].bar(x - width/2, orig_entropy_means, width, label='Original', alpha=0.8, color='steelblue')
axes[1].bar(x + width/2, mod_entropy_means, width, label='Modified', alpha=0.8, color='coral')
for i, (orig, mod) in enumerate(zip(orig_entropy_means, mod_entropy_means)):
    delta = abs(orig - mod)
    axes[1].text(i, max(orig, mod) * 1.05, f'Δ={delta:.2f}', ha='center', fontsize=9, fontweight='bold')
axes[1].set_xlabel('Technique', fontsize=11)
axes[1].set_ylabel('Entropy (RAW)', fontsize=11)
axes[1].set_title('Entropy Comparison', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(techniques, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Cohen's d
colors = ['green' if d > 0.8 else 'orange' if d > 0.5 else 'red' for d in cohens_d_values]
axes[2].bar(x, cohens_d_values, color=colors, alpha=0.8)
axes[2].axhline(y=0.8, color='green', linestyle='--', label='Large effect (>0.8)', alpha=0.5)
axes[2].axhline(y=0.5, color='orange', linestyle='--', label='Medium effect (>0.5)', alpha=0.5)
axes[2].axhline(y=0.2, color='red', linestyle='--', label='Small effect (>0.2)', alpha=0.5)
axes[2].set_xlabel('Technique', fontsize=11)
axes[2].set_ylabel("Cohen's d", fontsize=11)
axes[2].set_title("Effect Size (Cohen's d)", fontsize=12, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(techniques, rotation=45, ha='right')
axes[2].legend(fontsize=8)
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle(f'Aggregate Metrics ({len(all_results)} pairs)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n✓ Visualizzazione completata")

# ============================================================================
# CELLA 12: Interpretazione Finale e Raccomandazioni
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
    print(f"   ✅ Large effect - OTTIMA per il training!")
elif best_cohens_d > 0.5:
    print(f"   ✅ Medium effect - Buona per il training")
elif best_cohens_d > 0.2:
    print(f"   ⚠️  Small effect - Potrebbe funzionare")
else:
    print(f"   ❌ Negligible effect - Non raccomandato")

print(f"\n💡 Raccomandazioni:")
print(f"  1. Testare {best_technique} su più coppie per conferma")
print(f"  2. Integrare nella pipeline come:")
print(f"     - Preprocessing (usa solo noise map)")
print(f"     - Dual-input (RGB + noise map)")
print(f"     - Multi-channel (concatena come canali)")
print(f"  3. Valutare miglioramento metriche di classificazione")

print(f"\n📊 Riepilogo analisi:")
print(f"  Coppie analizzate: {len(all_results)}")
print(f"  Tecniche testate: {len(techniques)}")
print(f"  Tecnica migliore: {best_technique}")
print(f"  Effect size: {best_cohens_d:.4f}")

print("\n" + "="*80)
print("✓ ANALISI COMPLETATA")
print("="*80)
