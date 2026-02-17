import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


@torch.no_grad()
def predict_proba(model, loader: DataLoader, device):
    """
    Predice probabilità e raccoglie metadati.
    
    Returns:
        probs: array di probabilità
        y_true: array di label vere
        metadata: lista di dict con metadati per ogni sample
    """
    model.eval()  # CRITICAL: Disabilita dropout e usa running stats per batch norm
    logits_all, y_all, metadata_all = [], [], []
    
    for x, y, meta_batch in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).squeeze(1).detach().cpu().numpy()
        logits_all.append(logits)
        y_all.append(y.numpy())
        
        # Converti batch di metadati in lista di dict
        batch_size = len(y)
        for i in range(batch_size):
            meta_dict = {k: v[i] if isinstance(v, (list, tuple)) else v for k, v in meta_batch.items()}
            metadata_all.append(meta_dict)
    
    logits_all = np.concatenate(logits_all)
    y_all = np.concatenate(y_all).astype(np.int32)
    probs = sigmoid_np(logits_all)
    
    return probs, y_all, metadata_all


def compute_metrics_from_probs(probs: np.ndarray, y_true: np.ndarray, threshold=0.5):
    y_pred = (probs >= threshold).astype(np.int32)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    roc_auc = np.nan
    pr_auc = np.nan
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "prec_macro": prec_m, "rec_macro": rec_m, "f1_macro": f1_m,
        "roc_auc": roc_auc, "pr_auc": pr_auc
    }


def find_optimal_threshold(probs: np.ndarray, y_true: np.ndarray, metric='f1', 
                           min_thresh=0.1, max_thresh=0.9, step=0.05):
    """
    Trova threshold ottimale su validation set.
    
    IMPORTANTE: Usa SOLO su validation set, MAI su test set!
    
    Args:
        probs: probabilità predette
        y_true: label vere
        metric: metrica da ottimizzare ('f1', 'precision', 'recall', 'accuracy')
        min_thresh, max_thresh, step: range di threshold da testare
    
    Returns:
        best_threshold: threshold ottimale
        best_score: score migliore ottenuto
        threshold_scores: dict con score per ogni threshold (per plotting)
    """
    thresholds = np.arange(min_thresh, max_thresh + step, step)
    scores = []
    
    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(np.int32)
        
        if metric == 'f1':
            score = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)[2]
        elif metric == 'precision':
            score = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)[0]
        elif metric == 'recall':
            score = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)[1]
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    threshold_scores = {
        'thresholds': thresholds.tolist(),
        'scores': scores,
        'best_threshold': float(best_threshold),
        'best_score': float(best_score),
        'metric': metric
    }
    
    return best_threshold, best_score, threshold_scores


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.bad = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False, True
        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.bad = 0
            return False, True
        self.bad += 1
        return self.bad >= self.patience, False



def compute_group_metrics(df: pd.DataFrame, group_col: str, threshold=0.5):
    """
    Calcola metriche per gruppi (es: per food_category, defect_type, etc.)
    
    Args:
        df: DataFrame con colonne y_true, y_prob, y_pred + metadati
        group_col: nome colonna per raggruppamento
        threshold: soglia per classificazione
    
    Returns:
        DataFrame con metriche per gruppo
    """
    results = []
    
    for group_val, group_df in df.groupby(group_col):
        if pd.isna(group_val) or len(group_df) == 0:
            continue
        
        y_true = group_df['y_true'].values
        y_prob = group_df['y_prob'].values
        y_pred = group_df['y_pred'].values
        
        n = len(y_true)
        n_pos = (y_true == 1).sum()
        n_neg = (y_true == 0).sum()
        
        # Metriche base
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        
        # AUC (solo se entrambe le classi presenti)
        roc_auc = np.nan
        pr_auc = np.nan
        if len(np.unique(y_true)) == 2:
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
                pr_auc = average_precision_score(y_true, y_prob)
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        results.append({
            group_col: group_val,
            'n_samples': n,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    return pd.DataFrame(results).sort_values('n_samples', ascending=False)


def get_top_errors(df: pd.DataFrame, error_type='fp', top_n=50):
    """
    Estrae i top errori (falsi positivi o falsi negativi).
    
    Args:
        df: DataFrame con y_true, y_pred, y_prob + metadati
        error_type: 'fp' (false positives) o 'fn' (false negatives)
        top_n: numero di errori da restituire
    
    Returns:
        DataFrame con top errori ordinati per confidenza
    """
    if error_type == 'fp':
        # Predetto 1 ma vero 0
        errors = df[(df['y_pred'] == 1) & (df['y_true'] == 0)].copy()
        errors = errors.sort_values('y_prob', ascending=False)
    elif error_type == 'fn':
        # Predetto 0 ma vero 1
        errors = df[(df['y_pred'] == 0) & (df['y_true'] == 1)].copy()
        errors = errors.sort_values('y_prob', ascending=True)
    else:
        raise ValueError("error_type deve essere 'fp' o 'fn'")
    
    return errors.head(top_n)
