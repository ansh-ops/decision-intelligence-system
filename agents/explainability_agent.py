import numpy as np
import shap


class ExplainabilityAgent:
    """
    Robust SHAP-based explainability.
    Guarantees non-empty output by falling back to
    transformed feature indices if names cannot be resolved.
    """

    def __init__(self, max_background_samples: int = 100):
        self.max_background_samples = max_background_samples

    def run(self, pipeline, X, task_type: str):
        # --------------------------------------------------
        # Validate pipeline
        # --------------------------------------------------
        if not hasattr(pipeline, "named_steps"):
            raise ValueError("ExplainabilityAgent expects a sklearn Pipeline.")

        step_names = list(pipeline.named_steps.keys())

        if len(step_names) < 2:
            raise ValueError("Pipeline must include preprocessing and model.")

        model = pipeline.named_steps[step_names[-1]]
        preprocessor = pipeline.named_steps[step_names[-2]]

        # --------------------------------------------------
        # Background sample
        # --------------------------------------------------
        X_bg = X.sample(
            n=min(len(X), self.max_background_samples),
            random_state=42
        )

        # --------------------------------------------------
        # Transform
        # --------------------------------------------------
        X_bg_t = preprocessor.transform(X_bg)

        if hasattr(X_bg_t, "toarray"):
            X_bg_t = X_bg_t.toarray()

        # --------------------------------------------------
        # SHAP explainer
        # --------------------------------------------------
        try:
            explainer = shap.Explainer(model, X_bg_t)
        except Exception:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer(X_bg_t)

        # Binary classification → positive class
        if (
            task_type == "binary_classification"
            and shap_values.values.ndim == 3
        ):
            shap_matrix = shap_values.values[:, :, 1]
        else:
            shap_matrix = shap_values.values

        # --------------------------------------------------
        # Aggregate importance
        # --------------------------------------------------
        mean_abs_shap = np.mean(
            np.abs(shap_matrix), axis=0
        )

        n_features = mean_abs_shap.shape[0]

        # --------------------------------------------------
        # Resolve feature names (best effort)
        # --------------------------------------------------
        feature_names = self._resolve_feature_names(
            preprocessor, X, n_features
        )

        # --------------------------------------------------
        # Build final output (NEVER empty)
        # --------------------------------------------------
        output = [
            {
                "feature": feature_names[i],
                "importance": float(mean_abs_shap[i]),
            }
            for i in range(n_features)
        ]

        output.sort(
            key=lambda x: x["importance"],
            reverse=True
        )

        return output

    # --------------------------------------------------
    # Feature name resolution with guaranteed fallback
    # --------------------------------------------------
    def _resolve_feature_names(self, preprocessor, X, n_features):
        """
        Attempts to extract feature names from ColumnTransformer.
        If it fails, falls back to generic feature indices.
        """

        names = []

        # Try to unwrap ColumnTransformer
        ct = None

        if hasattr(preprocessor, "transformers_"):
            ct = preprocessor
        elif hasattr(preprocessor, "steps"):
            for _, step in preprocessor.steps:
                if hasattr(step, "transformers_"):
                    ct = step
                    break

        if ct is not None:
            try:
                for _, transformer, cols in ct.transformers_:
                    if transformer == "drop":
                        continue
                    if hasattr(transformer, "get_feature_names_out"):
                        names.extend(
                            transformer.get_feature_names_out(cols)
                        )
                    else:
                        names.extend(cols)
            except Exception:
                names = []

        # Hard fallback (guaranteed)
        if len(names) != n_features:
            names = [f"feature_{i}" for i in range(n_features)]

        return names
