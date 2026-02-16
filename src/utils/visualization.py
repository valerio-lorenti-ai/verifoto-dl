import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


def plot_prob_distributions(probs, y_true, bins=18, title="Predicted probability distributions", save_path=None):
    p0 = probs[y_true == 0]
    p1 = probs[y_true == 1]
    plt.figure(figsize=(7, 4))
    plt.hist(p0, bins=bins, alpha=0.7, label="NON_FRODE (0)")
    plt.hist(p1, bins=bins, alpha=0.7, label="FRODE (1)")
    plt.xlabel("Predicted P(FRODE)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_pr(probs, y_true, save_path_roc=None, save_path_pr=None):
    if len(np.unique(y_true)) < 2:
        print("ROC/PR curve non disponibili: una sola classe nel set.")
        return

    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC curve (AUC={roc_auc:.3f})")
    plt.grid(True, alpha=0.3)
    if save_path_roc:
        plt.savefig(save_path_roc, dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (AP/PR-AUC={pr_auc:.3f})")
    plt.grid(True, alpha=0.3)
    if save_path_pr:
        plt.savefig(save_path_pr, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['NON_FRODE', 'FRODE'])
    plt.yticks(tick_marks, ['NON_FRODE', 'FRODE'])

    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
