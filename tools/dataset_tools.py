import pandas as pd
from core.schema_inference import infer_schema
from core.statistics import compute_statistics
from core.task_detection import infer_candidate_targets, infer_task_type


def load_dataset(file_path: str):
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.read_excel(file_path)


def inspect_schema_tool(file_path: str):
    df = load_dataset(file_path)
    return {
        "columns": list(df.columns),
        "shape": df.shape,
        "schema": infer_schema(df),
        "candidate_targets": infer_candidate_targets(df),
    }


def profile_dataset_tool(file_path: str, target: str):
    df = load_dataset(file_path)

    normalized = {str(c).strip().lower(): c for c in df.columns}
    requested = target.strip().lower()

    if requested not in normalized:
        raise ValueError(
            f"Target '{target}' not found. Available columns: {list(df.columns)}"
        )

    actual_target = normalized[requested]

    task_type = infer_task_type(df[actual_target])
    stats = compute_statistics(df, actual_target, task_type)
    return {
        "target": actual_target,
        "task_type": task_type,
        "statistics": stats,
    }