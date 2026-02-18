"""
Temperature scaling for probability calibration.

Reduces overconfidence by optimizing a single temperature parameter T
on the validation set to minimize negative log-likelihood.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


def find_optimal_temperature(logits: np.ndarray, labels: np.ndarray, 
                             initial_T: float = 1.0) -> float:
    """
    Find optimal temperature T to minimize NLL on validation set.
    
    Args:
        logits: Raw model outputs (before sigmoid) [N]
        labels: Ground truth labels [N]
        initial_T: Initial temperature guess
    
    Returns:
        optimal_T: Temperature that minimizes NLL
    """
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    def nll_loss(T):
        """Negative log-likelihood with temperature scaling."""
        T_val = float(T[0])
        if T_val <= 0:
            return 1e10  # Invalid temperature
        
        # Apply temperature scaling
        scaled_logits = logits_tensor / T_val
        probs = torch.sigmoid(scaled_logits)
        
        # Binary cross-entropy (NLL for binary classification)
        loss = F.binary_cross_entropy(probs, labels_tensor, reduction='mean')
        return loss.item()
    
    # Optimize temperature
    result = minimize(
        nll_loss,
        x0=[initial_T],
        bounds=[(0.1, 10.0)],
        method='L-BFGS-B'
    )
    
    optimal_T = float(result.x[0])
    optimal_nll = result.fun
    
    print(f"\nTemperature Calibration:")
    print(f"  Initial T: {initial_T:.4f}")
    print(f"  Optimal T: {optimal_T:.4f}")
    print(f"  NLL (before): {nll_loss([initial_T]):.4f}")
    print(f"  NLL (after):  {optimal_nll:.4f}")
    
    return optimal_T


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to logits.
    
    Args:
        logits: Raw model outputs [N]
        temperature: Temperature parameter
    
    Returns:
        calibrated_probs: Calibrated probabilities [N]
    """
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    scaled_logits = logits_tensor / temperature
    calibrated_probs = torch.sigmoid(scaled_logits).numpy()
    return calibrated_probs


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities [N]
        labels: Ground truth labels [N]
        n_bins: Number of bins for calibration
    
    Returns:
        ece: Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calibration_report(probs_before: np.ndarray, probs_after: np.ndarray, 
                       labels: np.ndarray) -> dict:
    """
    Generate calibration report comparing before/after.
    
    Args:
        probs_before: Probabilities before calibration
        probs_after: Probabilities after calibration
        labels: Ground truth labels
    
    Returns:
        report: Dictionary with calibration metrics
    """
    ece_before = compute_ece(probs_before, labels)
    ece_after = compute_ece(probs_after, labels)
    
    # Overconfidence analysis
    overconfident_before = ((probs_before > 0.95) & (labels == 0)).sum()
    overconfident_after = ((probs_after > 0.95) & (labels == 0)).sum()
    
    underconfident_before = ((probs_before < 0.05) & (labels == 1)).sum()
    underconfident_after = ((probs_after < 0.05) & (labels == 1)).sum()
    
    report = {
        'ece_before': float(ece_before),
        'ece_after': float(ece_after),
        'ece_improvement': float(ece_before - ece_after),
        'overconfident_negatives_before': int(overconfident_before),
        'overconfident_negatives_after': int(overconfident_after),
        'underconfident_positives_before': int(underconfident_before),
        'underconfident_positives_after': int(underconfident_after)
    }
    
    return report
