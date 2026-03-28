import json
import os

from services.llm_client import GeminiLLMClient


def build_ai_enrichment(result: dict) -> dict:
    summary = _deterministic_summary(result)

    if not os.getenv("GEMINI_API_KEY"):
        return {
            "provider": "deterministic",
            "enabled": False,
            "summary": summary,
        }

    try:
        client = GeminiLLMClient()
        prompt = f"""
You are helping summarize a completed tabular ML analysis run.

Return ONLY valid JSON with this exact schema:
{{
  "executive_summary": "2-4 sentences",
  "recommended_next_step": "1 sentence",
  "watchouts": ["short item", "short item"]
}}

Use this analysis result:
{json.dumps(result, default=str)}
"""
        raw = client.generate(prompt)
        parsed = json.loads(raw)
        return {
            "provider": "gemini",
            "enabled": True,
            "summary": parsed,
        }
    except Exception as exc:
        return {
            "provider": "deterministic",
            "enabled": False,
            "summary": summary,
            "error": str(exc),
        }


def _deterministic_summary(result: dict) -> dict:
    modeling = result.get("modeling", {})
    recommended = modeling.get("recommended_model", {})
    evaluation = result.get("evaluation", {}).get("overall_metrics", {})
    target = result.get("target", "unknown")
    task_type = result.get("task_type", "unknown")
    model_name = recommended.get("name") or "best available model"

    primary_metric = "N/A"
    for key in ["f1", "roc_auc", "accuracy", "r2", "rmse", "mae"]:
        value = evaluation.get(key)
        if value is not None:
            primary_metric = f"{key}={value}"
            break

    return {
        "executive_summary": (
            f"The run completed for target '{target}' as a {task_type} task. "
            f"The recommended model is {model_name} with primary evaluation signal {primary_metric}."
        ),
        "recommended_next_step": (
            "Validate the recommended threshold and review segment-level errors before production use."
        ),
        "watchouts": [
            "Cross-validation and holdout performance should be checked together.",
            "Monitor production drift and missing-value patterns after deployment.",
        ],
    }
