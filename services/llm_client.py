import os
import json
import re
from google import genai


class MockLLMClient:
    def generate(self, prompt: str) -> str:
        state = self._extract_state(prompt)
        artifacts = state.get("artifacts_available", [])

        if not artifacts:
            return json.dumps({
                "tool_name": "inspect_schema_tool",
                "arguments": {},
                "reasoning": "Start by inspecting the dataset schema."
            })

        if "profile_dataset_tool" not in artifacts:
            return json.dumps({
                "tool_name": "profile_dataset_tool",
                "arguments": {},
                "reasoning": "Profile the dataset using the inferred target column."
            })

        return json.dumps({
            "tool_name": "train_baseline_models_tool",
            "arguments": {},
            "reasoning": "Train baseline models."
        })

    def _extract_state(self, prompt: str) -> dict:
        match = re.search(r"Current state:\s*(\{.*?\})\s*Choose exactly one next tool from:", prompt, re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(1))
        except Exception:
            return {}


class GeminiLLMClient:
    def __init__(self, model: str = "gemini-3-flash-preview"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini returned an empty response")
        return text.strip()


def get_llm_client():
    if os.getenv("GEMINI_API_KEY"):
        return GeminiLLMClient()
    return MockLLMClient()
