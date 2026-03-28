from tools.dataset_tools import inspect_schema_tool, profile_dataset_tool
from tools.modeling_tools import train_baseline_models_tool


TOOL_REGISTRY = {
    "inspect_schema_tool": inspect_schema_tool,
    "profile_dataset_tool": profile_dataset_tool,
    "train_baseline_models_tool": train_baseline_models_tool,
}


def dispatch_tool(tool_name: str, arguments: dict):
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool not found: {tool_name}")
    return TOOL_REGISTRY[tool_name](**arguments)