class DecisionAgent:
    def __init__(self, llm=None):
        self.llm = llm

    def run(
        self,
        task_type: str,
        modeling_best: dict,
        overall_metrics: dict,
        segment_summary,
        explainability,
        error_analysis: dict,
    ) -> dict:

        # ✅ Your run_modeling stores best model name under "model"
        model_selected = (
            modeling_best.get("model")
            or modeling_best.get("model_name")
            or modeling_best.get("selected_model")
            or "Unknown"
        )

        cv_perf = {
            "primary_metric": modeling_best.get("primary_metric"),
            "cv_mean": modeling_best.get("cv_mean"),
            "cv_std": modeling_best.get("cv_std"),
        }
        cv_perf = {k: v for k, v in cv_perf.items() if v is not None}

        fit_metrics = modeling_best.get("fit_metrics", {})

        top_features = (explainability or [])[:5] if isinstance(explainability, list) else []

        return {
            "model_selected": model_selected,
            "cv_performance": cv_perf,
            "fit_metrics": fit_metrics,
            "overall_metrics": overall_metrics,
            "risk_segmentation_summary": segment_summary or [],
            "top_explanations": {"top_features": top_features},
            "error_analysis": error_analysis or {},
        }
