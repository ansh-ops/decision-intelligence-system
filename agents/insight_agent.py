import json


class InsightAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_summary(
        self,
        filename: str,
        task_type: str,
        modeling_results: dict,
        metrics: dict,
        explainability: dict,
        prior_context: list,
    ) -> dict:
        prompt = f"""
You are generating a business-facing analytical summary.

Dataset: {filename}
Task type: {task_type}

Modeling results:
{json.dumps(modeling_results, indent=2, default=str)}

Metrics:
{json.dumps(metrics, indent=2, default=str)}

Explainability:
{json.dumps(explainability, indent=2, default=str)}

Retrieved prior analyses:
{json.dumps(prior_context, indent=2, default=str)}

Return strict JSON:
{{
  "executive_summary": "...",
  "key_risks": ["..."],
  "recommended_actions": ["..."]
}}
"""
        raw = self.llm_client.generate(prompt)
        return json.loads(raw)