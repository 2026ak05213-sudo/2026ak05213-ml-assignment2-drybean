"""Shared evaluation helpers (used by train_models.py and app.py)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Return the six required metrics as a dict.

    Binary problem      -> precision/recall/F1 use average='binary'
    Multi-class problem -> precision/recall/F1 use average='weighted',
                           AUC uses one-vs-rest.
    """
    avg = "binary" if n_classes == 2 else "weighted"

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_true, y_proba, multi_class="ovr",
                average="weighted", labels=list(range(n_classes)),
            )
    except ValueError:
        # e.g. the uploaded test file contains only one class
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
