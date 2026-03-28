def validate_tool_arguments(tool_name: str, arguments: dict):
    required = {
        "inspect_schema_tool": ["file_path"],
        "profile_dataset_tool": ["file_path", "target"],
        "train_baseline_models_tool": ["file_path", "target", "task_type", "stats"],
    }

    required_args = required.get(tool_name, [])
    missing = [arg for arg in required_args if arg not in arguments]
    if missing:
        raise ValueError(f"Missing required args for {tool_name}: {missing}")
