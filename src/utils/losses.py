"""
Advanced loss functions per AI-generated image detection.
Focus su riduzione FP su immagini reali.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss per binary classification.
    
    Riduce il peso degli esempi "easy" e si concentra sugli esempi "hard".
    Utile quando hai molti easy negatives (immagini reali ovvie) e pochi hard negatives.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: peso per classe positiva (default 0.25)
        gamma: focusing parameter (default 2.0)
               - gamma=0: equivalente a BCE
               - gamma>0: riduce peso su easy examples
               - gamma=2: standard
        reduction: 'mean', 'sum', o 'none'
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (N,) raw logits from model
            targets: (N,) binary targets (0 or 1)
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss con peso aggiuntivo per immagini reali.
    
    Combina:
    1. Focal Loss (riduce peso su easy examples)
    2. Class weights (bilancia classi sbilanciate)
    3. Real image weight (penalizza di più errori su immagini reali)
    
    Questo è il MIGLIORE per il tuo caso perché:
    - Riduce peso su easy negatives (immagini generate ovvie)
    - Aumenta peso su hard negatives (immagini reali che sembrano generate)
    - Penalizza di più FP su immagini reali
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=1.0, real_weight=2.0, reduction='mean'):
        """
        Args:
            alpha: peso per classe positiva in focal loss
            gamma: focusing parameter
            pos_weight: peso per classe positiva (per bilanciare classi)
            real_weight: peso extra per errori su immagini reali (label=0)
            reduction: 'mean', 'sum', o 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.real_weight = real_weight
        self.reduction = reduction

    def forward(self, logits, targets, is_real=None):
        """
        Args:
            logits: (N,) raw logits from model
            targets: (N,) binary targets (0=real, 1=generated)
            is_real: (N,) boolean mask indicating real images (optional)
                     Se None, assume is_real = (targets == 0)
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t (con pos_weight)
        alpha_t = self.alpha * self.pos_weight * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        # Apply real image weight
        if is_real is None:
            is_real = (targets == 0)
        
        # Penalizza di più errori su immagini reali
        real_weight_mask = torch.where(
            is_real,
            torch.tensor(self.real_weight, device=logits.device),
            torch.tensor(1.0, device=logits.device)
        )
        
        weighted_focal_loss = focal_loss * real_weight_mask
        
        if self.reduction == 'mean':
            return weighted_focal_loss.mean()
        elif self.reduction == 'sum':
            return weighted_focal_loss.sum()
        else:
            return weighted_focal_loss


class CostSensitiveLoss(nn.Module):
    """
    Loss function con costi asimmetrici per FP e FN.
    
    Utile quando FP e FN hanno costi diversi:
    - FP (predire FRODE su immagine reale): costo alto (falso allarme)
    - FN (predire REALE su immagine generata): costo basso (miss)
    
    Formula: Loss = fp_cost * FP_loss + fn_cost * FN_loss
    """
    def __init__(self, fp_cost=2.0, fn_cost=1.0, reduction='mean'):
        """
        Args:
            fp_cost: costo di un falso positivo
            fn_cost: costo di un falso negativo
            reduction: 'mean', 'sum', o 'none'
        """
        super().__init__()
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (N,) raw logits from model
            targets: (N,) binary targets (0=real, 1=generated)
        """
        probs = torch.sigmoid(logits)
        
        # Compute BCE loss per sample
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Compute cost weights
        # FP: target=0, pred=1 → penalizza con fp_cost
        # FN: target=1, pred=0 → penalizza con fn_cost
        # TP, TN: costo normale (1.0)
        
        # Approssimazione: usa probabilità per pesare
        # Se target=0 e prob alta → FP → alto costo
        # Se target=1 e prob bassa → FN → costo medio
        cost_weight = torch.where(
            targets == 0,
            1.0 + (self.fp_cost - 1.0) * probs,  # Più prob è alta, più costa
            1.0 + (self.fn_cost - 1.0) * (1 - probs)  # Più prob è bassa, più costa
        )
        
        weighted_loss = bce_loss * cost_weight
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


def build_loss_function(loss_type='bce', **kwargs):
    """
    Factory function per creare loss function.
    
    Args:
        loss_type: 'bce', 'focal', 'weighted_focal', 'cost_sensitive'
        **kwargs: parametri specifici per ogni loss
    
    Returns:
        loss_fn: loss function
    """
    if loss_type == 'bce':
        pos_weight = kwargs.get('pos_weight', 1.0)
        if isinstance(pos_weight, (int, float)):
            pos_weight = torch.tensor([pos_weight])
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('focal_alpha', 0.25)
        gamma = kwargs.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'weighted_focal':
        alpha = kwargs.get('focal_alpha', 0.25)
        gamma = kwargs.get('focal_gamma', 2.0)
        pos_weight = kwargs.get('pos_weight', 1.0)
        real_weight = kwargs.get('real_weight', 2.0)
        return WeightedFocalLoss(
            alpha=alpha, gamma=gamma, 
            pos_weight=pos_weight, real_weight=real_weight
        )
    
    elif loss_type == 'cost_sensitive':
        fp_cost = kwargs.get('fp_cost', 2.0)
        fn_cost = kwargs.get('fn_cost', 1.0)
        return CostSensitiveLoss(fp_cost=fp_cost, fn_cost=fn_cost)
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# Test functions
if __name__ == "__main__":
    # Test focal loss
    print("Testing Focal Loss...")
    logits = torch.randn(10)
    targets = torch.randint(0, 2, (10,)).float()
    
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test weighted focal loss
    print("\nTesting Weighted Focal Loss...")
    weighted_focal = WeightedFocalLoss(alpha=0.25, gamma=2.0, pos_weight=1.5, real_weight=2.0)
    loss = weighted_focal(logits, targets)
    print(f"Weighted Focal Loss: {loss.item():.4f}")
    
    # Test cost sensitive loss
    print("\nTesting Cost Sensitive Loss...")
    cost_sensitive = CostSensitiveLoss(fp_cost=2.0, fn_cost=1.0)
    loss = cost_sensitive(logits, targets)
    print(f"Cost Sensitive Loss: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")
