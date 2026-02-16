import numpy as np
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
    model.eval()
    logits_all, y_all, paths_all = [], [], []
    for x, y, fp in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).squeeze(1).detach().cpu().numpy()
        logits_all.append(logits)
        y_all.append(y.numpy())
        paths_all.extend(fp)
    logits_all = np.concatenate(logits_all)
    y_all = np.concatenate(y_all).astype(np.int32)
    probs = sigmoid_np(logits_all)
    return probs, y_all, paths_all


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
