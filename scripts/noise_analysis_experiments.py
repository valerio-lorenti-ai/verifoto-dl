"""
Noise Analysis Experiments - Forensic Edition
==============================================
Analisi forense di tecniche di noise detection per image forensics.

PRINCIPIO GUIDA:
L'obiettivo NON è migliorare l'estetica o uniformare i dati.
L'obiettivo è PRESERVARE e RENDERE VISIBILI eventuali anomalie forensi nel rumore.

Tecniche implementate:
1. Median Noise Residual - Reverse denoising (simile a Forensically)
2. Gaussian Residual Noise - Variante smooth
3. High-pass Filter - Componenti ad alta frequenza (Laplacian)
4. Error Level Analysis (ELA) - Analisi compressione JPEG

IMPORTANTE:
- I residui RAW sono sempre preservati senza normalizzazione
- Le normalizzazioni sono SOLO per visualizzazione
- Le metriche sono calcolate sui dati RAW
- Ogni tecnica mostra visualizzazione neutra (grayscale) + enfatizzata (colormap)

Usage:
    # Analisi singola coppia
    python noise_analysis_experiments.py --original path/to/original.jpg --modified path/to/modified.jpg
    
    # Analisi batch (Verifoto dataset)
    python noise_analysis_experiments.py --verifoto-dataset images/coppie-orig-mod/gpt --output results/
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import argparse
from dataclasses import dataclass, asdict
import warnings
import json
from scipy import stats
warnings.filterwarnings('ignore')


@dataclass
class NoiseMetrics:
    """
    Metriche quantitative calcolate sui residui RAW (non normalizzati).
    
    Queste metriche riflettono il segnale forense reale senza alterazioni.
    """
    std: float  # Standard deviation - variabilità del rumore
    mean_abs: float  # Mean absolute value - intensità media
    range: float  # Max - Min - dinamica del segnale
    snr: float  # Signal-to-Noise Ratio
    entropy: float  # Entropia dell'istogramma
    percentile_95: float  # 95° percentile - robustezza agli outliers
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ClassSeparationMetrics:
    """Metriche di separazione tra classe original e modified."""
    delta_std: float  # Differenza assoluta STD
    delta_entropy: float  # Differenza assoluta Entropy
    cohens_d: float  # Effect size (Cohen's d)
    separation_score: float  # Score custom di separabilità



class NoiseAnalyzer:
    """
    Classe per l'analisi forense del rumore.
    
    IMPORTANTE: Tutti i metodi restituiscono residui RAW senza normalizzazione.
    La normalizzazione è responsabilità del visualizer, non dell'analyzer.
    """
    
    def __init__(self, median_kernel: int = 5, gaussian_kernel: int = 5, 
                 gaussian_sigma: float = 1.5, ela_quality: int = 90):
        """
        Args:
            median_kernel: Dimensione kernel per median filter (dispari)
            gaussian_kernel: Dimensione kernel per gaussian blur (dispari)
            gaussian_sigma: Sigma per gaussian blur
            ela_quality: Quality factor per ELA (default: 90, standard forense)
        """
        self.median_kernel = median_kernel
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
        self.ela_quality = ela_quality
    
    def load_image(self, path: str) -> np.ndarray:
        """
        Carica immagine in formato RGB float [0, 1].
        
        IMPORTANTE: Preserva il range dinamico originale.
        """
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Impossibile caricare l'immagine: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0
    
    def median_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Median Noise Residual - Reverse denoising (metodo Forensically).
        
        Formula: noise = image - median_filter(image)
        
        IMPORTANTE: Restituisce residuo RAW che può contenere valori negativi.
        Range tipico: [-0.5, 0.5] ma può variare.
        
        Evidenzia:
        - Airbrush artifacts
        - Warping
        - Manipolazioni AI (alterazione del rumore naturale)
        - Interpolazioni geometriche
        """
        img_uint8 = (image * 255).astype(np.uint8)
        denoised = cv2.medianBlur(img_uint8, self.median_kernel)
        denoised = denoised.astype(np.float32) / 255.0
        
        # Residuo RAW - può essere negativo
        noise = image - denoised
        return noise
    
    def gaussian_residual_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Gaussian Residual Noise - Variante smooth del median residual.
        
        Formula: noise = image - gaussian_blur(image)
        
        IMPORTANTE: Restituisce residuo RAW con valori negativi.
        Range tipico: [-0.3, 0.3]
        
        Caratteristiche:
        - Più smooth del median filter
        - Meno sensibile a outliers
        - Isola rumore ad alta frequenza
        """
        denoised = cv2.GaussianBlur(image, 
                                     (self.gaussian_kernel, self.gaussian_kernel),
                                     self.gaussian_sigma)
        noise = image - denoised
        return noise
    
    def high_pass_filter(self, image: np.ndarray) -> np.ndarray:
        """
        High-pass Filter - Evidenzia componenti ad alta frequenza (Laplacian).
        
        IMPORTANTE: Restituisce valori RAW del Laplacian (possono essere negativi).
        Range tipico: [-1.0, 1.0] ma può variare significativamente.
        
        Evidenzia:
        - Artefatti di interpolazione
        - Pattern sintetici (tipici di AI-generated)
        - Anomalie nelle texture
        - Bordi e discontinuità
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Laplacian RAW - preserva valori negativi
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian = laplacian.astype(np.float32) / 255.0  # Scala ma non normalizza
        
        # Converti a RGB per consistenza (replica il canale)
        if len(image.shape) == 3:
            laplacian = np.stack([laplacian] * 3, axis=-1)
        
        return laplacian

    
    def error_level_analysis(self, image: np.ndarray) -> np.ndarray:
        """
        Error Level Analysis (ELA) - Tecnica classica di image forensics.
        
        Formula: ela = |image - jpeg_compress(image, quality)|
        
        IMPORTANTE: Restituisce differenza assoluta RAW.
        Range: [0, 1] (sempre positivo perché è valore assoluto)
        
        Quality factor:
        - Default: 90 (standard forense)
        - Per immagini già compresse a Q basso, potrebbe servire Q più basso
        - Documentato in letteratura forense: Q=90-95 è ottimale per la maggior parte dei casi
        
        Evidenzia:
        - Differenze locali di compressione
        - Aree ricompresse multiple volte
        - Possibili manipolazioni (splicing, cloning)
        
        NOTA: ELA è più efficace per manipolazioni locali che per classificazione globale.
        """
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Comprimi e ricarica
        pil_img = Image.fromarray(img_uint8)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=self.ela_quality)
        buffer.seek(0)
        
        compressed_img = Image.open(buffer)
        compressed = np.array(compressed_img).astype(np.float32) / 255.0
        
        # Differenza assoluta RAW
        ela = np.abs(image - compressed)
        
        return ela
    
    def compute_metrics(self, noise: np.ndarray) -> NoiseMetrics:
        """
        Calcola metriche quantitative sui residui RAW (non normalizzati).
        
        IMPORTANTE: Le metriche riflettono il segnale forense reale.
        """
        # Converti a grayscale se necessario
        if len(noise.shape) == 3:
            noise_gray = cv2.cvtColor((np.clip(noise, -1, 1) * 127.5 + 127.5).astype(np.uint8), 
                                     cv2.COLOR_RGB2GRAY).astype(np.float32) / 127.5 - 1.0
        else:
            noise_gray = noise
        
        # Standard deviation - variabilità del rumore
        std = float(np.std(noise_gray))
        
        # Mean absolute value - intensità media
        mean_abs = float(np.mean(np.abs(noise_gray)))
        
        # Range - dinamica del segnale
        range_val = float(np.max(noise_gray) - np.min(noise_gray))
        
        # SNR approssimato
        signal_power = np.mean(noise_gray ** 2)
        noise_power = np.var(noise_gray)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Entropia dell'istogramma
        # Normalizza per istogramma ma calcola su dati raw
        noise_norm = (noise_gray - noise_gray.min()) / (noise_gray.max() - noise_gray.min() + 1e-10)
        hist, _ = np.histogram(noise_norm.flatten(), bins=256, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # 95° percentile - robustezza agli outliers
        percentile_95 = float(np.percentile(np.abs(noise_gray), 95))
        
        return NoiseMetrics(
            std=std,
            mean_abs=mean_abs,
            range=range_val,
            snr=float(snr),
            entropy=float(entropy),
            percentile_95=percentile_95
        )
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, NoiseMetrics]]:
        """
        Applica tutte le tecniche di noise analysis a un'immagine.
        
        Returns:
            Dict con chiave = nome tecnica, valore = (raw_noise_map, metrics)
            
        IMPORTANTE: Le noise maps sono RAW, senza normalizzazione.
        """
        results = {}
        
        results['Median Residual'] = (
            self.median_noise_residual(image),
            None  # Metrics calcolate dopo
        )
        
        results['Gaussian Residual'] = (
            self.gaussian_residual_noise(image),
            None
        )
        
        results['High-pass Filter'] = (
            self.high_pass_filter(image),
            None
        )
        
        results['ELA'] = (
            self.error_level_analysis(image),
            None
        )
        
        # Calcola metriche su raw data
        for technique in results:
            noise, _ = results[technique]
            metrics = self.compute_metrics(noise)
            results[technique] = (noise, metrics)
        
        return results



class NoiseVisualizer:
    """
    Classe per visualizzare i risultati dell'analisi forense.
    
    PRINCIPIO: Mostrare sempre visualizzazione neutra + enfatizzata.
    - Neutra: Grayscale, rappresentazione fedele
    - Enfatizzata: Colormap divergente, evidenzia anomalie
    """
    
    @staticmethod
    def normalize_for_display(noise: np.ndarray, method: str = 'centered') -> np.ndarray:
        """
        Normalizza il noise residual SOLO per visualizzazione.
        
        IMPORTANTE: Questa funzione NON deve essere usata per analisi o metriche.
        È SOLO per rendere visibili i pattern nelle immagini.
        
        Args:
            noise: Raw noise map
            method: 'centered' (per residui con negativi) o 'minmax' (per valori positivi)
        """
        if method == 'centered':
            # Centra intorno a 0.5 per visualizzare positivi/negativi
            abs_max = max(abs(noise.min()), abs(noise.max()))
            if abs_max < 1e-8:
                return np.ones_like(noise) * 0.5
            normalized = (noise / (2 * abs_max)) + 0.5
            normalized = np.clip(normalized, 0, 1)
        elif method == 'minmax':
            # MinMax per valori sempre positivi (es. ELA)
            range_val = noise.max() - noise.min()
            if range_val < 1e-8:
                return np.zeros_like(noise)
            normalized = (noise - noise.min()) / range_val
        else:
            raise ValueError(f"Metodo sconosciuto: {method}")
        
        return normalized
    
    @staticmethod
    def to_grayscale_display(noise: np.ndarray, method: str = 'centered') -> np.ndarray:
        """
        Converte noise map a grayscale per visualizzazione neutra.
        
        Returns: Array [H, W] normalizzato per display
        """
        normalized = NoiseVisualizer.normalize_for_display(noise, method)
        
        if len(normalized.shape) == 3:
            # Converti a grayscale
            gray = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            return gray.astype(np.float32) / 255.0
        
        return normalized

    
    @staticmethod
    def plot_comparison(original: np.ndarray, modified: np.ndarray,
                       original_results: Dict, modified_results: Dict,
                       save_path: Optional[str] = None):
        """
        Visualizza confronto completo: original vs modified con tutte le tecniche.
        
        Layout per ogni tecnica:
        - Colonna 1: Visualizzazione neutra (grayscale)
        - Colonna 2: Visualizzazione enfatizzata (colormap)
        
        Questo permette valutazione obiettiva senza bias della colormap.
        """
        techniques = list(original_results.keys())
        n_technique_cols = len(techniques) * 2  # 2 colonne per tecnica (neutra + enfatizzata)
        n_cols = 1 + n_technique_cols  # +1 per immagine originale
        
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
        
        # Row 1: Original
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('ORIGINAL\nImage', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        col_idx = 1
        for technique in techniques:
            noise, metrics = original_results[technique]
            
            # Determina metodo di normalizzazione
            if 'Residual' in technique or 'High-pass' in technique:
                norm_method = 'centered'
                cmap = 'RdBu_r'
            else:  # ELA
                norm_method = 'minmax'
                cmap = 'hot'
            
            # Visualizzazione neutra (grayscale)
            gray = NoiseVisualizer.to_grayscale_display(noise, norm_method)
            axes[0, col_idx].imshow(gray, cmap='gray', vmin=0, vmax=1)
            axes[0, col_idx].set_title(f'{technique}\n(Neutral)', fontsize=10)
            axes[0, col_idx].axis('off')
            
            # Visualizzazione enfatizzata (colormap)
            colored = NoiseVisualizer.normalize_for_display(noise, norm_method)
            axes[0, col_idx + 1].imshow(colored, cmap=cmap)
            axes[0, col_idx + 1].set_title(f'{technique}\n(Enhanced)\nSTD:{metrics.std:.4f}', 
                                          fontsize=10)
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
            
            # Neutra
            gray = NoiseVisualizer.to_grayscale_display(noise, norm_method)
            axes[1, col_idx].imshow(gray, cmap='gray', vmin=0, vmax=1)
            axes[1, col_idx].set_title(f'{technique}\n(Neutral)', fontsize=10)
            axes[1, col_idx].axis('off')
            
            # Enfatizzata
            colored = NoiseVisualizer.normalize_for_display(noise, norm_method)
            axes[1, col_idx + 1].imshow(colored, cmap=cmap)
            axes[1, col_idx + 1].set_title(f'{technique}\n(Enhanced)\nSTD:{metrics.std:.4f}', 
                                          fontsize=10)
            axes[1, col_idx + 1].axis('off')
            
            col_idx += 2
        
        plt.suptitle('Forensic Noise Analysis: Original vs Modified', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Comparison plot salvato: {save_path}")
        
        plt.show()

    
    @staticmethod
    def plot_difference_maps(original_results: Dict, modified_results: Dict,
                            save_path: Optional[str] = None):
        """
        Visualizza difference maps: |original_noise - modified_noise|
        
        Mostra quanto ogni tecnica discrimina tra le due classi.
        Valori alti = buona separabilità.
        """
        techniques = list(original_results.keys())
        n_cols = len(techniques)
        
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]
        
        for idx, technique in enumerate(techniques):
            orig_noise, _ = original_results[technique]
            mod_noise, _ = modified_results[technique]
            
            # Differenza assoluta RAW
            diff = np.abs(orig_noise - mod_noise)
            
            # Metriche di separazione
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            max_diff = np.max(diff)
            
            # Visualizza
            diff_display = NoiseVisualizer.normalize_for_display(diff, 'minmax')
            axes[idx].imshow(diff_display, cmap='hot')
            axes[idx].set_title(f'{technique}\nMean Δ: {mean_diff:.4f}\nMax Δ: {max_diff:.4f}',
                               fontsize=11, fontweight='bold')
            axes[idx].axis('off')
        
        plt.suptitle('Difference Maps: |Original Noise - Modified Noise|\n(Higher values = Better discrimination)', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Difference maps salvate: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(original_results: Dict, modified_results: Dict,
                               save_path: Optional[str] = None):
        """
        Visualizza confronto delle metriche RAW tra original e modified.
        
        Mostra metriche calcolate sui dati non normalizzati.
        """
        techniques = list(original_results.keys())
        metrics_names = ['std', 'mean_abs', 'entropy', 'percentile_95']
        metrics_labels = ['Standard Deviation', 'Mean Absolute', 'Entropy', '95th Percentile']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
            orig_values = [getattr(original_results[t][1], metric_name) for t in techniques]
            mod_values = [getattr(modified_results[t][1], metric_name) for t in techniques]
            
            x = np.arange(len(techniques))
            width = 0.35
            
            bars1 = axes[idx].bar(x - width/2, orig_values, width, label='Original', 
                                 alpha=0.8, color='steelblue')
            bars2 = axes[idx].bar(x + width/2, mod_values, width, label='Modified', 
                                 alpha=0.8, color='coral')
            
            # Aggiungi delta come testo
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
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Metrics comparison salvato: {save_path}")
        
        plt.show()



def compute_class_separation(original_metrics_list: List[NoiseMetrics],
                             modified_metrics_list: List[NoiseMetrics]) -> ClassSeparationMetrics:
    """
    Calcola metriche di separazione tra classe original e modified.
    
    Args:
        original_metrics_list: Lista di metriche per immagini originali
        modified_metrics_list: Lista di metriche per immagini modificate
    
    Returns:
        ClassSeparationMetrics con indicatori di separabilità
    """
    # Estrai STD
    orig_stds = np.array([m.std for m in original_metrics_list])
    mod_stds = np.array([m.std for m in modified_metrics_list])
    
    # Estrai Entropy
    orig_entropies = np.array([m.entropy for m in original_metrics_list])
    mod_entropies = np.array([m.entropy for m in modified_metrics_list])
    
    # Delta medie
    delta_std = abs(np.mean(orig_stds) - np.mean(mod_stds))
    delta_entropy = abs(np.mean(orig_entropies) - np.mean(mod_entropies))
    
    # Cohen's d (effect size) per STD
    pooled_std = np.sqrt((np.var(orig_stds) + np.var(mod_stds)) / 2)
    if pooled_std > 1e-10:
        cohens_d = abs(np.mean(orig_stds) - np.mean(mod_stds)) / pooled_std
    else:
        cohens_d = 0.0
    
    # Separation score custom (combina delta_std e delta_entropy normalizzati)
    # Normalizza per range tipici
    std_score = delta_std / (np.mean(orig_stds) + np.mean(mod_stds) + 1e-10)
    entropy_score = delta_entropy / (np.mean(orig_entropies) + np.mean(mod_entropies) + 1e-10)
    separation_score = (std_score + entropy_score) / 2
    
    return ClassSeparationMetrics(
        delta_std=delta_std,
        delta_entropy=delta_entropy,
        cohens_d=cohens_d,
        separation_score=separation_score
    )


def analyze_image_pair(original_path: str, modified_path: str, 
                       output_dir: Optional[str] = None,
                       analyzer_params: Optional[Dict] = None,
                       verbose: bool = True):
    """
    Analizza una coppia di immagini (original, modified) con tutte le tecniche forensi.
    
    Args:
        original_path: Path all'immagine originale
        modified_path: Path all'immagine modificata
        output_dir: Directory dove salvare le visualizzazioni (opzionale)
        analyzer_params: Parametri custom per NoiseAnalyzer (opzionale)
        verbose: Se True, stampa output dettagliato
    
    Returns:
        (original_results, modified_results)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"FORENSIC NOISE ANALYSIS")
        print(f"{'='*80}")
        print(f"Original:  {original_path}")
        print(f"Modified:  {modified_path}")
        print(f"{'='*80}\n")
    
    # Inizializza analyzer
    params = analyzer_params or {}
    analyzer = NoiseAnalyzer(**params)
    
    # Carica immagini
    if verbose:
        print("📂 Caricamento immagini...")
    original = analyzer.load_image(original_path)
    modified = analyzer.load_image(modified_path)
    if verbose:
        print(f"   Original shape: {original.shape}")
        print(f"   Modified shape: {modified.shape}")
    
    # Analizza
    if verbose:
        print("\n🔬 Applicazione tecniche forensi...")
    original_results = analyzer.analyze_image(original)
    modified_results = analyzer.analyze_image(modified)
    
    # Print metriche RAW
    if verbose:
        print("\n" + "="*80)
        print("METRICHE QUANTITATIVE (calcolate su RAW data)")
        print("="*80)
        for technique in original_results.keys():
            print(f"\n{technique}:")
            orig_metrics = original_results[technique][1]
            mod_metrics = modified_results[technique][1]
            
            print(f"  Original  → STD: {orig_metrics.std:.5f} | "
                  f"Entropy: {orig_metrics.entropy:.2f} | "
                  f"Range: {orig_metrics.range:.5f}")
            print(f"  Modified  → STD: {mod_metrics.std:.5f} | "
                  f"Entropy: {mod_metrics.entropy:.2f} | "
                  f"Range: {mod_metrics.range:.5f}")
            
            delta_std = abs(orig_metrics.std - mod_metrics.std)
            delta_entropy = abs(orig_metrics.entropy - mod_metrics.entropy)
            print(f"  Δ STD: {delta_std:.5f} | Δ Entropy: {delta_entropy:.2f}")
    
    # Visualizza
    if verbose:
        print("\n" + "="*80)
        print("VISUALIZZAZIONI")
        print("="*80)
    
    visualizer = NoiseVisualizer()
    
    # Setup output paths
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = output_dir / "comparison.png"
        difference_path = output_dir / "difference_maps.png"
        metrics_path = output_dir / "metrics_comparison.png"
    else:
        comparison_path = difference_path = metrics_path = None
    
    # Plot comparison
    if verbose:
        print("\n1. Comparison plot (neutral + enhanced per ogni tecnica)...")
    visualizer.plot_comparison(original, modified, original_results, modified_results,
                              save_path=comparison_path)
    
    # Plot difference maps
    if verbose:
        print("\n2. Difference maps (discriminabilità)...")
    visualizer.plot_difference_maps(original_results, modified_results,
                                   save_path=difference_path)
    
    # Plot metrics
    if verbose:
        print("\n3. Metrics comparison...")
    visualizer.plot_metrics_comparison(original_results, modified_results,
                                      save_path=metrics_path)
    
    if verbose:
        print("\n" + "="*80)
        print("✓ ANALISI COMPLETATA")
        print("="*80)
    
    return original_results, modified_results



def batch_analyze_verifoto(dataset_dir: str, output_dir: str, 
                           max_pairs: Optional[int] = None,
                           analyzer_params: Optional[Dict] = None):
    """
    Analizza batch di coppie dal dataset Verifoto.
    
    Struttura attesa:
        dataset_dir/
            originali/
                Orig_*.jpeg
            modificate/
                Mod_fromOrig_*.jpeg
    
    Args:
        dataset_dir: Directory del dataset (es. images/coppie-orig-mod/gpt)
        output_dir: Directory output per i risultati
        max_pairs: Numero massimo di coppie da analizzare (None = tutte)
        analyzer_params: Parametri custom per NoiseAnalyzer
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Trova immagini
    orig_dir = dataset_dir / "originali"
    mod_dir = dataset_dir / "modificate"
    
    if not orig_dir.exists() or not mod_dir.exists():
        raise ValueError(f"Directory non trovate: {orig_dir} o {mod_dir}")
    
    orig_files = sorted(list(orig_dir.glob("Orig_*.jpeg")))
    
    if max_pairs:
        orig_files = orig_files[:max_pairs]
    
    print(f"\n{'='*80}")
    print(f"BATCH FORENSIC ANALYSIS - VERIFOTO DATASET")
    print(f"{'='*80}")
    print(f"Dataset:       {dataset_dir}")
    print(f"Output:        {output_dir}")
    print(f"Coppie totali: {len(orig_files)}")
    print(f"{'='*80}\n")
    
    # Statistiche aggregate per classe
    class_stats = {
        'original': {technique: [] for technique in ['Median Residual', 'Gaussian Residual', 
                                                      'High-pass Filter', 'ELA']},
        'modified': {technique: [] for technique in ['Median Residual', 'Gaussian Residual', 
                                                      'High-pass Filter', 'ELA']}
    }
    
    successful_pairs = 0
    
    for pair_idx, orig_file in enumerate(orig_files, 1):
        # Trova corrispondente modificata
        timestamp = orig_file.stem.replace('Orig_', '')
        mod_file = mod_dir / f"Mod_fromOrig_{timestamp}.jpeg"
        
        if not mod_file.exists():
            print(f"⚠ SKIP pair {pair_idx}: Modified non trovata per {orig_file.name}")
            continue
        
        print(f"\n{'─'*80}")
        print(f"Processing pair {pair_idx}/{len(orig_files)}: {orig_file.name}")
        print(f"{'─'*80}")
        
        # Analizza coppia
        pair_output_dir = output_dir / f"pair_{pair_idx:03d}"
        
        try:
            orig_results, mod_results = analyze_image_pair(
                str(orig_file),
                str(mod_file),
                output_dir=str(pair_output_dir),
                analyzer_params=analyzer_params,
                verbose=False  # Meno verbose in batch
            )
            
            # Aggrega metriche per classe
            for technique in orig_results.keys():
                class_stats['original'][technique].append(orig_results[technique][1])
                class_stats['modified'][technique].append(mod_results[technique][1])
            
            successful_pairs += 1
            print(f"✓ Pair {pair_idx} completata")
        
        except Exception as e:
            print(f"✗ ERROR pair {pair_idx}: {e}")
            continue
    
    # Analisi statistiche aggregate
    print("\n\n" + "="*80)
    print("STATISTICHE AGGREGATE PER CLASSE")
    print("="*80)
    print(f"Coppie analizzate con successo: {successful_pairs}/{len(orig_files)}\n")
    
    separation_results = {}
    
    for technique in class_stats['original'].keys():
        orig_metrics = class_stats['original'][technique]
        mod_metrics = class_stats['modified'][technique]
        
        if not orig_metrics:
            continue
        
        print(f"\n{technique}:")
        print(f"{'─'*80}")
        
        # Statistiche per classe
        orig_std_mean = np.mean([m.std for m in orig_metrics])
        orig_std_std = np.std([m.std for m in orig_metrics])
        mod_std_mean = np.mean([m.std for m in mod_metrics])
        mod_std_std = np.std([m.std for m in mod_metrics])
        
        orig_entropy_mean = np.mean([m.entropy for m in orig_metrics])
        mod_entropy_mean = np.mean([m.entropy for m in mod_metrics])
        
        print(f"  STD:")
        print(f"    Original class  → Mean: {orig_std_mean:.5f} ± {orig_std_std:.5f}")
        print(f"    Modified class  → Mean: {mod_std_mean:.5f} ± {mod_std_std:.5f}")
        print(f"    Δ Mean: {abs(orig_std_mean - mod_std_mean):.5f}")
        
        print(f"  Entropy:")
        print(f"    Original class  → Mean: {orig_entropy_mean:.2f}")
        print(f"    Modified class  → Mean: {mod_entropy_mean:.2f}")
        print(f"    Δ Mean: {abs(orig_entropy_mean - mod_entropy_mean):.2f}")
        
        # Calcola separazione
        separation = compute_class_separation(orig_metrics, mod_metrics)
        separation_results[technique] = separation
        
        print(f"  Separation Metrics:")
        print(f"    Cohen's d:        {separation.cohens_d:.4f}")
        print(f"    Separation score: {separation.separation_score:.4f}")
        
        # Statistical significance (t-test)
        orig_stds = [m.std for m in orig_metrics]
        mod_stds = [m.std for m in mod_metrics]
        t_stat, p_value = stats.ttest_ind(orig_stds, mod_stds)
        print(f"    t-test p-value:   {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    # Salva risultati in JSON
    results_json = {
        'dataset': str(dataset_dir),
        'total_pairs': len(orig_files),
        'successful_pairs': successful_pairs,
        'techniques': {}
    }
    
    for technique, separation in separation_results.items():
        orig_metrics = class_stats['original'][technique]
        mod_metrics = class_stats['modified'][technique]
        
        results_json['techniques'][technique] = {
            'original_class': {
                'std_mean': float(np.mean([m.std for m in orig_metrics])),
                'std_std': float(np.std([m.std for m in orig_metrics])),
                'entropy_mean': float(np.mean([m.entropy for m in orig_metrics]))
            },
            'modified_class': {
                'std_mean': float(np.mean([m.std for m in mod_metrics])),
                'std_std': float(np.std([m.std for m in mod_metrics])),
                'entropy_mean': float(np.mean([m.entropy for m in mod_metrics]))
            },
            'separation': {
                'delta_std': float(separation.delta_std),
                'delta_entropy': float(separation.delta_entropy),
                'cohens_d': float(separation.cohens_d),
                'separation_score': float(separation.separation_score)
            }
        }
    
    json_path = output_dir / "batch_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✓ Risultati salvati in: {json_path}")
    
    # Visualizzazione aggregate
    print(f"\n{'='*80}")
    print("VISUALIZZAZIONE AGGREGATE")
    print(f"{'='*80}")
    
    plot_class_separation_summary(separation_results, output_dir)
    
    print(f"\n{'='*80}")
    print("✓ BATCH ANALYSIS COMPLETATA")
    print(f"{'='*80}")


def plot_class_separation_summary(separation_results: Dict[str, ClassSeparationMetrics],
                                  output_dir: Path):
    """Visualizza summary delle metriche di separazione tra classi."""
    techniques = list(separation_results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Cohen's d
    cohens_d_values = [separation_results[t].cohens_d for t in techniques]
    axes[0].bar(range(len(techniques)), cohens_d_values, color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(len(techniques)))
    axes[0].set_xticklabels(techniques, rotation=45, ha='right')
    axes[0].set_ylabel("Cohen's d", fontsize=12)
    axes[0].set_title("Effect Size (Cohen's d)\nHigher = Better separation", fontsize=12, fontweight='bold')
    axes[0].axhline(y=0.5, color='orange', linestyle='--', label='Medium effect')
    axes[0].axhline(y=0.8, color='red', linestyle='--', label='Large effect')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Separation score
    sep_scores = [separation_results[t].separation_score for t in techniques]
    axes[1].bar(range(len(techniques)), sep_scores, color='coral', alpha=0.8)
    axes[1].set_xticks(range(len(techniques)))
    axes[1].set_xticklabels(techniques, rotation=45, ha='right')
    axes[1].set_ylabel("Separation Score", fontsize=12)
    axes[1].set_title("Custom Separation Score\nHigher = Better discrimination", fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Delta STD
    delta_stds = [separation_results[t].delta_std for t in techniques]
    axes[2].bar(range(len(techniques)), delta_stds, color='seagreen', alpha=0.8)
    axes[2].set_xticks(range(len(techniques)))
    axes[2].set_xticklabels(techniques, rotation=45, ha='right')
    axes[2].set_ylabel("Δ STD", fontsize=12)
    axes[2].set_title("STD Difference Between Classes\nHigher = More distinct", fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Class Separation Summary (Original vs Modified)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / "class_separation_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Class separation summary salvato: {save_path}")
    plt.show()



def main():
    """Entry point per CLI."""
    parser = argparse.ArgumentParser(
        description='Forensic Noise Analysis - Verifoto Edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PRINCIPIO FORENSE:
  L'obiettivo NON è migliorare l'estetica o uniformare i dati.
  L'obiettivo è PRESERVARE e RENDERE VISIBILI eventuali anomalie forensi.

Esempi di utilizzo:

  # Analisi singola coppia
  python noise_analysis_experiments.py \\
      --original images/coppie-orig-mod/gpt/originali/Orig_2026-02-27_13.31.49.jpeg \\
      --modified images/coppie-orig-mod/gpt/modificate/Mod_fromOrig_2026-02-27_13.31.49.jpeg \\
      --output results/single_pair/
  
  # Batch analysis su dataset Verifoto
  python noise_analysis_experiments.py \\
      --verifoto-dataset images/coppie-orig-mod/gpt \\
      --output results/batch_analysis/ \\
      --max-pairs 20
  
  # Con parametri custom
  python noise_analysis_experiments.py \\
      --original img1.jpg --modified img2.jpg \\
      --median-kernel 7 --ela-quality 95 \\
      --output results/
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--original', type=str, 
                           help='Path immagine originale (single pair mode)')
    mode_group.add_argument('--verifoto-dataset', type=str, 
                           help='Path al dataset Verifoto (batch mode)')
    
    # Single mode args
    parser.add_argument('--modified', type=str, 
                       help='Path immagine modificata (required in single pair mode)')
    
    # Common args
    parser.add_argument('--output', type=str, required=True,
                       help='Directory output per visualizzazioni e risultati')
    parser.add_argument('--max-pairs', type=int, 
                       help='Numero massimo di coppie in batch mode')
    
    # Analyzer parameters
    parser.add_argument('--median-kernel', type=int, default=5, 
                       help='Dimensione kernel median filter (default: 5)')
    parser.add_argument('--gaussian-kernel', type=int, default=5,
                       help='Dimensione kernel gaussian blur (default: 5)')
    parser.add_argument('--gaussian-sigma', type=float, default=1.5,
                       help='Sigma gaussian blur (default: 1.5)')
    parser.add_argument('--ela-quality', type=int, default=90,
                       help='Quality factor per ELA 0-100 (default: 90, standard forense)')
    
    args = parser.parse_args()
    
    # Prepara parametri analyzer
    analyzer_params = {
        'median_kernel': args.median_kernel,
        'gaussian_kernel': args.gaussian_kernel,
        'gaussian_sigma': args.gaussian_sigma,
        'ela_quality': args.ela_quality
    }
    
    print("\n" + "="*80)
    print("FORENSIC NOISE ANALYSIS - VERIFOTO EDITION")
    print("="*80)
    print("\nPRINCIPIO GUIDA:")
    print("  ✓ Preservare anomalie forensi nel rumore")
    print("  ✓ Metriche calcolate su dati RAW")
    print("  ✓ Normalizzazioni SOLO per visualizzazione")
    print("="*80)
    
    # Single pair mode
    if args.original:
        if not args.modified:
            parser.error("--modified è richiesto quando si usa --original")
        
        analyze_image_pair(
            args.original,
            args.modified,
            output_dir=args.output,
            analyzer_params=analyzer_params,
            verbose=True
        )
    
    # Batch mode (Verifoto dataset)
    elif args.verifoto_dataset:
        batch_analyze_verifoto(
            args.verifoto_dataset,
            args.output,
            max_pairs=args.max_pairs,
            analyzer_params=analyzer_params
        )


if __name__ == '__main__':
    main()
