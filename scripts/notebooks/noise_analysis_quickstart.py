"""
Forensic Noise Analysis - Quick Start Examples
===============================================

Esempi rapidi per testare le tecniche di noise analysis sul dataset Verifoto.
Ottimizzato per immagini AI-edited (food delivery).

PRINCIPIO FORENSE:
- Preservare anomalie forensi nel rumore
- Metriche su dati RAW
- Normalizzazioni SOLO per visualizzazione
"""

import sys
from pathlib import Path

# Aggiungi parent directory al path
sys.path.append(str(Path(__file__).parent.parent))

from noise_analysis_experiments import (
    NoiseAnalyzer,
    NoiseVisualizer,
    analyze_image_pair,
    batch_analyze_verifoto
)
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# ESEMPIO 1: Analisi Singola Coppia (Dataset Verifoto)
# ============================================================================

def example_verifoto_single_pair():
    """Analizza una coppia dal dataset Verifoto."""
    
    print("="*80)
    print("ESEMPIO 1: Analisi Singola Coppia (Verifoto)")
    print("="*80)
    
    # Paths dal dataset Verifoto
    original_path = "images/coppie-orig-mod/gpt/originali/Orig_2026-02-27_13.31.49.jpeg"
    modified_path = "images/coppie-orig-mod/gpt/modificate/Mod_fromOrig_2026-02-27_13.31.49.jpeg"
    
    # Analizza
    analyze_image_pair(
        original_path,
        modified_path,
        output_dir="results/verifoto_single_pair"
    )
    
    print("\n✓ Risultati salvati in: results/verifoto_single_pair/")


# ============================================================================
# ESEMPIO 2: Batch Analysis su Dataset Verifoto
# ============================================================================

def example_verifoto_batch():
    """Analizza batch di coppie dal dataset Verifoto."""
    
    print("\n" + "="*80)
    print("ESEMPIO 2: Batch Analysis (Verifoto)")
    print("="*80)
    
    # Path al dataset
    dataset_path = "images/coppie-orig-mod/gpt"
    
    # Analizza prime 10 coppie
    batch_analyze_verifoto(
        dataset_path,
        output_dir="results/verifoto_batch_10",
        max_pairs=10
    )
    
    print("\n✓ Risultati salvati in: results/verifoto_batch_10/")
    print("✓ Vedi batch_results.json per statistiche aggregate")


# ============================================================================
# ESEMPIO 3: Analisi Programmatica con Accesso ai Raw Data
# ============================================================================

def example_raw_data_access():
    """Esempio di accesso diretto ai raw residuals per analisi custom."""
    
    print("\n" + "="*80)
    print("ESEMPIO 3: Accesso Raw Data")
    print("="*80)
    
    # Inizializza analyzer
    analyzer = NoiseAnalyzer()
    
    # Carica immagini
    original_path = "images/coppie-orig-mod/gpt/originali/Orig_2026-02-27_13.31.49.jpeg"
    modified_path = "images/coppie-orig-mod/gpt/modificate/Mod_fromOrig_2026-02-27_13.31.49.jpeg"
    
    original = analyzer.load_image(original_path)
    modified = analyzer.load_image(modified_path)
    
    print(f"\nImmagini caricate:")
    print(f"  Original shape: {original.shape}")
    print(f"  Modified shape: {modified.shape}")
    
    # Applica Median Residual
    print("\n🔬 Applicazione Median Residual...")
    median_noise_orig = analyzer.median_noise_residual(original)
    median_noise_mod = analyzer.median_noise_residual(modified)
    
    # Analizza raw data
    print(f"\nRAW Residual Statistics:")
    print(f"  Original:")
    print(f"    Range: [{median_noise_orig.min():.5f}, {median_noise_orig.max():.5f}]")
    print(f"    Mean:  {median_noise_orig.mean():.5f}")
    print(f"    STD:   {median_noise_orig.std():.5f}")
    
    print(f"  Modified:")
    print(f"    Range: [{median_noise_mod.min():.5f}, {median_noise_mod.max():.5f}]")
    print(f"    Mean:  {median_noise_mod.mean():.5f}")
    print(f"    STD:   {median_noise_mod.std():.5f}")
    
    # Calcola differenza
    diff = np.abs(median_noise_orig - median_noise_mod)
    print(f"\n  Difference:")
    print(f"    Mean: {diff.mean():.5f}")
    print(f"    Max:  {diff.max():.5f}")
    
    # Visualizza raw vs normalized
    print("\n📊 Creazione visualizzazione raw vs normalized...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Raw residual (centered around 0.5 for display)
    raw_display = (median_noise_orig / (2 * max(abs(median_noise_orig.min()), 
                                                 abs(median_noise_orig.max())))) + 0.5
    axes[0, 1].imshow(raw_display, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Raw Residual (Neutral)\nRange: [{median_noise_orig.min():.3f}, {median_noise_orig.max():.3f}]')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(raw_display, cmap='RdBu_r')
    axes[0, 2].set_title('Raw Residual (Enhanced)')
    axes[0, 2].axis('off')
    
    # Histogram
    axes[0, 3].hist(median_noise_orig.flatten(), bins=100, alpha=0.7, color='steelblue')
    axes[0, 3].set_title('Raw Residual Histogram')
    axes[0, 3].set_xlabel('Value')
    axes[0, 3].set_ylabel('Frequency')
    axes[0, 3].axvline(x=0, color='red', linestyle='--', label='Zero')
    axes[0, 3].legend()
    
    # Row 2: Modified
    axes[1, 0].imshow(modified)
    axes[1, 0].set_title('Modified Image (AI-edited)')
    axes[1, 0].axis('off')
    
    raw_display_mod = (median_noise_mod / (2 * max(abs(median_noise_mod.min()), 
                                                    abs(median_noise_mod.max())))) + 0.5
    axes[1, 1].imshow(raw_display_mod, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Raw Residual (Neutral)\nRange: [{median_noise_mod.min():.3f}, {median_noise_mod.max():.3f}]')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(raw_display_mod, cmap='RdBu_r')
    axes[1, 2].set_title('Raw Residual (Enhanced)')
    axes[1, 2].axis('off')
    
    axes[1, 3].hist(median_noise_mod.flatten(), bins=100, alpha=0.7, color='coral')
    axes[1, 3].set_title('Raw Residual Histogram')
    axes[1, 3].set_xlabel('Value')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].axvline(x=0, color='red', linestyle='--', label='Zero')
    axes[1, 3].legend()
    
    plt.suptitle('Raw Data Analysis: Median Noise Residual', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path("results").mkdir(exist_ok=True)
    plt.savefig('results/raw_data_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Salvato in: results/raw_data_analysis.png")
    plt.show()


# ============================================================================
# ESEMPIO 4: Confronto Tecniche con Parametri Custom
# ============================================================================

def example_custom_parameters():
    """Testa diverse configurazioni di parametri."""
    
    print("\n" + "="*80)
    print("ESEMPIO 4: Parametri Custom")
    print("="*80)
    
    original_path = "images/coppie-orig-mod/gpt/originali/Orig_2026-02-27_13.31.49.jpeg"
    modified_path = "images/coppie-orig-mod/gpt/modificate/Mod_fromOrig_2026-02-27_13.31.49.jpeg"
    
    # Test 1: Kernel più grande
    print("\n📊 Test 1: Median kernel 7x7 (più denoising)")
    analyze_image_pair(
        original_path,
        modified_path,
        output_dir="results/custom_kernel7",
        analyzer_params={'median_kernel': 7},
        verbose=False
    )
    
    # Test 2: ELA quality diverso
    print("\n📊 Test 2: ELA quality 95 (più alta qualità)")
    analyze_image_pair(
        original_path,
        modified_path,
        output_dir="results/custom_ela95",
        analyzer_params={'ela_quality': 95},
        verbose=False
    )
    
    print("\n✓ Confronta i risultati in:")
    print("  - results/custom_kernel7/")
    print("  - results/custom_ela95/")


# ============================================================================
# ESEMPIO 5: Estrazione Noise Map per Training
# ============================================================================

def example_extract_for_training():
    """Esempio di come estrarre noise maps per usarle nel training."""
    
    print("\n" + "="*80)
    print("ESEMPIO 5: Estrazione per Training")
    print("="*80)
    
    analyzer = NoiseAnalyzer()
    
    # Carica immagine
    image_path = "images/coppie-orig-mod/gpt/originali/Orig_2026-02-27_13.31.49.jpeg"
    image = analyzer.load_image(image_path)
    
    # Estrai noise map (assumiamo Median Residual sia la migliore)
    noise_map = analyzer.median_noise_residual(image)
    
    print(f"\nNoise map estratta:")
    print(f"  Shape: {noise_map.shape}")
    print(f"  Dtype: {noise_map.dtype}")
    print(f"  Range: [{noise_map.min():.5f}, {noise_map.max():.5f}]")
    
    # Opzione 1: Salva come numpy array (preserva raw data)
    output_path = Path("results/noise_maps")
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / "noise_map_raw.npy", noise_map)
    print(f"\n✓ Raw noise map salvata: {output_path / 'noise_map_raw.npy'}")
    
    # Opzione 2: Converti per PyTorch
    import torch
    
    # Converti a tensor (preserva range raw)
    noise_tensor = torch.from_numpy(noise_map).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    print(f"\nPyTorch tensor:")
    print(f"  Shape: {noise_tensor.shape}")
    print(f"  Range: [{noise_tensor.min():.5f}, {noise_tensor.max():.5f}]")
    
    # Esempio di dual-input
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    
    print(f"\nDual-input setup:")
    print(f"  RGB tensor:   {image_tensor.shape} | Range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    print(f"  Noise tensor: {noise_tensor.shape} | Range: [{noise_tensor.min():.5f}, {noise_tensor.max():.5f}]")
    
    # Opzione 3: Multi-channel input
    multi_channel = torch.cat([image_tensor, noise_tensor], dim=0)
    print(f"\nMulti-channel input:")
    print(f"  Shape: {multi_channel.shape} (6 channels: 3 RGB + 3 noise)")
    
    print("\n💡 IMPORTANTE:")
    print("  - Il noise map è in range RAW (può essere negativo)")
    print("  - NON normalizzare a [0, 1] per preservare il segnale forense")
    print("  - Il modello deve gestire questo range (es. no ReLU nel primo layer)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Crea directory results
    Path("results").mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("FORENSIC NOISE ANALYSIS - QUICK START")
    print("="*80)
    print("\nEsempi disponibili:")
    print("  1. example_verifoto_single_pair() - Analisi singola coppia")
    print("  2. example_verifoto_batch() - Batch analysis (10 coppie)")
    print("  3. example_raw_data_access() - Accesso raw data")
    print("  4. example_custom_parameters() - Test parametri custom")
    print("  5. example_extract_for_training() - Estrazione per training")
    print("\nDecommenta l'esempio che vuoi eseguire.")
    print("="*80 + "\n")
    
    # Decommenta l'esempio che vuoi eseguire:
    
    # example_verifoto_single_pair()
    # example_verifoto_batch()
    # example_raw_data_access()
    # example_custom_parameters()
    # example_extract_for_training()
    
    # Oppure esegui tutti in sequenza:
    # example_verifoto_single_pair()
    # example_raw_data_access()
    # example_extract_for_training()
