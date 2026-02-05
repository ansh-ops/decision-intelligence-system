import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class ErrorAnalysisAgent:
    def __init__(self, max_points=800, n_clusters=4):
        self.max_points = max_points
        self.n_clusters = n_clusters

    def run(self, pipeline, X, y, task_type: str):
        prep = pipeline.named_steps["prep"]

        # Predict
        y_pred = pipeline.predict(X)

        if task_type == "binary_classification":
            y_prob = pipeline.predict_proba(X)[:, 1]
            # focus on confidently wrong cases
            wrong = (y_pred != y)
            # score wrongness: high confidence but wrong
            confidence = np.abs(y_prob - 0.5) * 2
            score = confidence * wrong.astype(float)
        else:
            # focus on large residuals
            resid = np.abs(y - y_pred)
            score = resid.values if hasattr(resid, "values") else resid

        df = pd.DataFrame({"score": score}, index=X.index).sort_values("score", ascending=False)
        df = df.head(min(len(df), self.max_points))
        X_hard = X.loc[df.index]

        Xt = prep.transform(X_hard)

        # Reduce dimension for clustering (works for sparse too)
        pca = PCA(n_components=10, random_state=42)
        Xt_red = pca.fit_transform(Xt.toarray() if hasattr(Xt, "toarray") else Xt)

        k = min(self.n_clusters, len(X_hard))
        if k < 2:
            return {"clusters": [], "note": "Not enough hard examples to cluster."}

        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(Xt_red)

        clusters = []
        for c in range(k):
            members = X_hard.index[labels == c].tolist()
            clusters.append({
                "cluster_id": int(c),
                "count": int(len(members)),
                "example_indices": members[:10]
            })

        return {
            "hard_example_count": int(len(X_hard)),
            "clusters": clusters,
            "note": "Clusters are formed in transformed feature space; use them to inspect common error patterns."
        }
