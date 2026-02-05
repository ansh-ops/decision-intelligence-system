import numpy as np
from sklearn.metrics import f1_score

def optimize_threshold(y_true, y_prob, metric="f1"):
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds)
        scores.append((t, score))

    best_t, best_score = max(scores, key=lambda x: x[1])
    return {
        "best_threshold": float(best_t),
        "best_score": float(best_score),
        "all_scores": scores
    }
