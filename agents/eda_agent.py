import json

class EDAAgent:
    def __init__(self, llm):
        self.llm = llm

    def build_prompt(self, schema, stats, target, task_type):
        return f"""
You are a senior data scientist performing exploratory data analysis.

Rules:
- Base ALL claims strictly on provided statistics.
- Do NOT speculate.
- Explicitly mention risks, assumptions, and limitations.

Dataset schema:
{json.dumps(schema, indent=2)}

Statistical summary:
{json.dumps(stats, indent=2)}

Target column: {target}
Task type: {task_type}

Generate:
1. Key data quality issues
2. Important statistical observations
3. Potential modeling risks
4. Hypotheses worth testing later
"""

    def run(self, schema, stats, target, task_type):
        prompt = self.build_prompt(schema, stats, target, task_type)
        narrative = self.llm(prompt)

        return {
            "eda_narrative": narrative,
            "stats": stats
        }
