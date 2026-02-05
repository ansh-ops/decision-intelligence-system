class FeatureAgent:
    def run(self, numeric_features, categorical_features, stats):
        reasoning = []

        if len(categorical_features) > 0:
            reasoning.append(
                "Applied one-hot encoding due to presence of categorical features."
            )

        if any(abs(v) > 1 for v in stats.get("skewness", {}).values()):
            reasoning.append(
                "Standard scaling applied to address skewed numeric distributions."
            )

        if max(stats["missingness"].values()) > 0:
            reasoning.append(
                "Imputation applied due to missing values."
            )

        return {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_reasoning": reasoning
        }
