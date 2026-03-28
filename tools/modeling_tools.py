import pandas as pd
from pipelines.run_modeling import run_modeling


def _normalize_binary_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        return

    series = df[target]
    if series.dtype != object:
        return

    cleaned = series.astype(str).str.strip().str.lower()
    unique_values = set(cleaned.unique())
    if unique_values <= {"yes", "no"}:
        df[target] = cleaned.map({"yes": 1, "no": 0})


def train_baseline_models_tool(
    file_path: str,
    target: str,
    task_type: str,
    stats: dict,
    policy: dict | None = None,
):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        df = pd.read_excel(file_path)

    _normalize_binary_target(df, target)

    result = run_modeling(df, target, task_type, stats, policy=policy)

    if "best_model" in result and "pipeline" in result["best_model"]:
        result["best_model"] = {
            k: v for k, v in result["best_model"].items() if k != "pipeline"
        }

    return result
