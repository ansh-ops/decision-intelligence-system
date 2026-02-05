class DecisionAgent:
    """
    Synthesizes modeling results, explainability, segmentation,
    and error analysis into a decision-focused report.

    This agent does NOT perform modeling.
    It translates results into actionable insights.
    """

    def __init__(self, llm=None):
        self.llm = llm  # kept for future extension, unused for now

    def run(
        self,
        task_type: str,
        modeling_best: dict,
        overall_metrics: dict,
        segment_summary: list | None,
        explainability: list | None,
        error_analysis: dict,
    ) -> dict:

        # --------------------------------------------------
        # Model summary
        # --------------------------------------------------
        model_selected = modeling_best.get("model_name")

        cv_perf = modeling_best.get("cv_performance", {})
        fit_metrics = modeling_best.get("fit_metrics", {})

        # --------------------------------------------------
        # Explainability (TOP drivers)
        # --------------------------------------------------
        top_features = []

        if explainability and isinstance(explainability, list):
            top_features = explainability[:5]  # top 5 global drivers

        top_explanations = {
            "top_features": top_features
        }

        # --------------------------------------------------
        # Risk segmentation (classification only)
        # --------------------------------------------------
        risk_summary = segment_summary if segment_summary else []

        # --------------------------------------------------
        # Limitations (honest + aligned with pipeline)
        # --------------------------------------------------
        limitations = [
            "Explainability reflects global feature importance, not individual predictions.",
            "One-hot encoding can expand feature space for high-cardinality categoricals.",
            "Metrics are evaluated on a held-out test set but assume stable data distribution."
        ]

        # --------------------------------------------------
        # Recommended next steps
        # --------------------------------------------------
        next_steps = [
            "Incorporate cost-sensitive or profit-based threshold optimization.",
            "Evaluate PR-AUC and calibration for imbalanced classification.",
            "Monitor feature drift and retrain periodically."
        ]

        # --------------------------------------------------
        # Final decision report
        # --------------------------------------------------
        return {
            "model_selected": model_selected,
            "cv_performance": cv_perf,
            "fit_metrics": fit_metrics,
            "overall_metrics": overall_metrics,
            "risk_segmentation_summary": risk_summary,
            "top_explanations": top_explanations,
            "error_analysis": error_analysis,
            "limitations": limitations,
            "recommended_next_steps": next_steps,
        }
