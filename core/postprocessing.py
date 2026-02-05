import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------
# Overall metrics (used in Week 3 & Week 4)
# ---------------------------------------------------------------------
def compute_overall_metrics(task_type, y_true, y_pred, y_prob=None):
    """
    Compute out-of-sample metrics.

    Parameters
    ----------
    task_type : str
        binary_classification | regression
    y_true : array-like
    y_pred : array-like
    y_prob : array-like or None
        Required for binary classification ROC-AUC

    Returns
    -------
    dict
    """
    if task_type == "binary_classification":
        if y_prob is None:
            raise ValueError("y_prob must be provided for binary classification")

        return {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "f1": float(f1_score(y_true, y_pred)),
        }

    elif task_type == "regression":
        return {
            "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


# ---------------------------------------------------------------------
# Risk segmentation helpers (classification)
# ---------------------------------------------------------------------
def make_risk_segments(y_prob, n_bins=10):
    """
    Assign risk segments based on predicted probabilities.
    Higher segment = higher risk.
    """
    return pd.qcut(
        y_prob,
        q=n_bins,
        labels=False,
        duplicates="drop",
    ).astype(int)


def segment_summary_classification(y_true, y_prob, n_bins=10):
    """
    Create a lift-style table by risk segment.
    """
    seg = make_risk_segments(y_prob, n_bins=n_bins)

    df = pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob,
        "segment": seg,
    })

    summary = []
    for s in sorted(df["segment"].unique()):
        d = df[df["segment"] == s]
        summary.append({
            "segment": int(s),
            "count": int(len(d)),
            "avg_pred_prob": float(d["y_prob"].mean()),
            "positive_rate": float(d["y_true"].mean()),
        })

    # Sort highest risk first
    summary = sorted(summary, key=lambda x: x["avg_pred_prob"], reverse=True)
    return summary
