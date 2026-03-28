import json
import os

from services.llm_client import GeminiLLMClient


def build_adaptive_policy(dataset_profile: dict, task_type: str, target: str) -> dict:
    heuristic_policy = _heuristic_policy(dataset_profile, task_type, target)

    if not os.getenv("GEMINI_API_KEY"):
        return heuristic_policy

    try:
        client = GeminiLLMClient()
        prompt = f"""
You are selecting an adaptive modeling policy for a tabular ML workflow.

Return ONLY valid JSON with this exact schema:
{{
  "policy_name": "short policy name",
  "primary_metric": "roc_auc or f1 or r2",
  "threshold_metric": "f1 or balanced_accuracy",
  "model_candidates": ["logistic", "rf"] ,
  "reasoning": "2-3 short sentences"
}}

Rules:
- For binary classification, primary_metric must be either "roc_auc" or "f1".
- For regression, primary_metric must be "r2".
- threshold_metric only matters for binary classification.
- model_candidates may only contain "logistic", "rf", or "linear".

Context:
{json.dumps({
    "task_type": task_type,
    "target": target,
    "dataset_profile": dataset_profile,
}, default=str)}
"""
        raw = client.generate(prompt)
        parsed = json.loads(raw)
        normalized = _normalize_policy(parsed, task_type)
        if normalized:
            normalized["source"] = "gemini"
            return normalized
    except Exception:
        pass

    return heuristic_policy


def _heuristic_policy(dataset_profile: dict, task_type: str, target: str) -> dict:
    rows = int(dataset_profile.get("rows") or 0)
    imbalance_ratio = dataset_profile.get("imbalance_ratio") or 1.0
    missing_columns = len(dataset_profile.get("missing_summary") or {})

    if task_type == "binary_classification":
        if imbalance_ratio > 2.0:
            return {
                "source": "heuristic",
                "policy_name": "imbalance_aware_policy",
                "primary_metric": "f1",
                "threshold_metric": "f1",
                "model_candidates": ["logistic", "rf"],
                "reasoning": (
                    f"The target '{target}' is materially imbalanced, so the policy optimizes for F1 "
                    "and keeps both a linear and non-linear classifier in the candidate set."
                ),
            }

        if rows < 2000 and missing_columns < 3:
            return {
                "source": "heuristic",
                "policy_name": "interpretable_small_data_policy",
                "primary_metric": "roc_auc",
                "threshold_metric": "balanced_accuracy",
                "model_candidates": ["logistic", "rf"],
                "reasoning": (
                    f"The dataset for '{target}' is moderate in size, so the policy starts with ROC-AUC "
                    "and favors interpretable baselines while still checking a random forest."
                ),
            }

        return {
            "source": "heuristic",
            "policy_name": "balanced_classification_policy",
            "primary_metric": "roc_auc",
            "threshold_metric": "balanced_accuracy",
            "model_candidates": ["logistic", "rf"],
            "reasoning": (
                "The dataset appears reasonably stable for a broad classification search, so ROC-AUC "
                "is used for model ranking and balanced accuracy guides thresholding."
            ),
        }

    return {
        "source": "heuristic",
        "policy_name": "regression_performance_policy",
        "primary_metric": "r2",
        "threshold_metric": None,
        "model_candidates": ["linear", "rf"],
        "reasoning": (
            f"The target '{target}' is treated as regression, so the policy compares linear and "
            "random-forest regressors using R2."
        ),
    }


def _normalize_policy(policy: dict, task_type: str) -> dict | None:
    allowed_models = {"logistic", "rf", "linear"}
    model_candidates = [
        str(model) for model in policy.get("model_candidates", []) if str(model) in allowed_models
    ]
    if not model_candidates:
        return None

    primary_metric = str(policy.get("primary_metric") or "").lower()
    threshold_metric = policy.get("threshold_metric")
    threshold_metric = str(threshold_metric).lower() if threshold_metric is not None else None

    if task_type == "binary_classification":
        if primary_metric not in {"roc_auc", "f1"}:
            return None
        if threshold_metric not in {"f1", "balanced_accuracy"}:
            threshold_metric = "f1"
    else:
        primary_metric = "r2"
        threshold_metric = None

    return {
        "policy_name": policy.get("policy_name") or "adaptive_policy",
        "primary_metric": primary_metric,
        "threshold_metric": threshold_metric,
        "model_candidates": model_candidates,
        "reasoning": policy.get("reasoning") or "Adaptive policy generated for this dataset.",
    }
