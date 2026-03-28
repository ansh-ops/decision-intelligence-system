import json
from typing import Dict, Any
from agents.planner_schema import ToolAction


ALLOWED_TOOLS = {
    "inspect_schema_tool",
    "profile_dataset_tool",
    "train_baseline_models_tool",
}


class PlannerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan_next_action(self, state: Dict[str, Any]) -> ToolAction:
        prompt = prompt = f"""
You are coordinating a tabular ML analysis workflow.

Current state:
{json.dumps(state, indent=2, default=str)}

Choose exactly one next tool from:
{sorted(ALLOWED_TOOLS)}

Return ONLY valid JSON with this exact schema:
{{
  "tool_name": "one of the allowed tools",
  "arguments": {{}},
  "reasoning": "brief explanation"
}}

Do not include markdown fences.
Do not include any text before or after the JSON.
"""
        raw = self.llm_client.generate(prompt)
        parsed = json.loads(raw)
        action = ToolAction(**parsed)

        if action.tool_name not in ALLOWED_TOOLS:
            raise ValueError(f"Unsupported tool requested: {action.tool_name}")

        return action