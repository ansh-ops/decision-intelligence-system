from pathlib import Path
import pandas as pd

from core.schema_inference import infer_schema
from core.task_detection import infer_candidate_targets, infer_task_type
from core.statistics import compute_statistics
from core.postprocessing import (
    compute_overall_metrics,
    segment_summary_classification,
)
from core.splitting import split_data
from core.thresholding import optimize_threshold

from agents.eda_agent import EDAAgent
from pipelines.run_modeling import run_modeling
from agents.explainability_agent import ExplainabilityAgent
from agents.error_analysis_agent import ErrorAnalysisAgent
from agents.decision_agent import DecisionAgent

print(">>> NEW run_analysis LOADED <<<")

# --------------------------------------------------
# Mock LLM (kept for future extensibility)
# --------------------------------------------------
def mock_llm(prompt: str) -> str:
    return "Narrative disabled (deterministic mode)."


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def run_pipeline(csv_path: str, target_override: str | None = None):
    # --------------------------------------------------
    # Robust file ingestion (CSV / Excel)
    # --------------------------------------------------
    path = Path(csv_path)

    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_excel(path)
        except Exception as e:
            raise ValueError(
                f"Could not read file {path}. "
                "Only CSV and Excel files are supported."
            ) from e

    # --------------------------------------------------
    # Target validation
    # --------------------------------------------------
    candidates = infer_candidate_targets(df)

    if target_override:
        if target_override not in df.columns:
            raise ValueError(
                f"Target '{target_override}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        target = target_override
    else:
        if not candidates:
            raise ValueError("No valid target candidates found.")
        target = candidates[0]

    # --------------------------------------------------
    # HARD normalize binary labels (CRITICAL)
    # --------------------------------------------------
    if df[target].dtype == object:
        cleaned = (
            df[target]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if set(cleaned.unique()) <= {"yes", "no"}:
            df[target] = cleaned.map({"yes": 1, "no": 0})

    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(
            f"Target '{target}' must be numeric after encoding. "
            f"Found dtype={df[target].dtype}"
        )

    # --------------------------------------------------
    # Task inference
    # --------------------------------------------------
    task_type = infer_task_type(df[target])

    # --------------------------------------------------
    # Statistics (Week 1 grounding)
    # --------------------------------------------------
    stats = compute_statistics(df, target, task_type)

    # --------------------------------------------------
    # Week 1 — EDA
    # --------------------------------------------------
    eda_agent = EDAAgent(llm=mock_llm)
    eda_results = eda_agent.run(
        infer_schema(df),
        stats,
        target,
        task_type,
    )

    # --------------------------------------------------
    # Train / Val / Test split (Week 4)
    # --------------------------------------------------
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, task_type
    )

    # --------------------------------------------------
    # Week 2 — Modeling (on TRAIN only)
    # --------------------------------------------------
    modeling_results = run_modeling(
        pd.concat([X_train, y_train], axis=1),
        target,
        task_type,
        stats,
    )

    best_pipeline = modeling_results["best_model"]["pipeline"]
    best_pipeline.fit(X_train, y_train)

    # --------------------------------------------------
    # Validation predictions (threshold tuning)
    # --------------------------------------------------
    decision_threshold = 0.5
    threshold_info = None

    if task_type == "binary_classification":
        y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
        threshold_info = optimize_threshold(y_val, y_val_prob)
        decision_threshold = threshold_info["best_threshold"]

    # --------------------------------------------------
    # Test predictions (true OOS)
    # --------------------------------------------------
    if task_type == "binary_classification":
        y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= decision_threshold).astype(int)
    else:
        y_test_pred = best_pipeline.predict(X_test)
        y_test_prob = None

    # --------------------------------------------------
    # Evaluation metrics (Week 4)
    # --------------------------------------------------
    overall_metrics = compute_overall_metrics(
        task_type,
        y_test,
        y_test_pred,
        y_prob=y_test_prob,
    )

    # --------------------------------------------------
    # Week 3 — Explainability (TRAIN set)
    # --------------------------------------------------
    explain_agent = ExplainabilityAgent()
    explainability = explain_agent.run(
        best_pipeline,
        X_train,
        task_type,
    )

    # --------------------------------------------------
    # Risk segmentation (classification only)
    # --------------------------------------------------
    segment_summary = None
    if task_type == "binary_classification":
        segment_summary = segment_summary_classification(
            y_test,
            y_test_prob,
            n_bins=10,
        )

    # --------------------------------------------------
    # Error analysis (TEST set)
    # --------------------------------------------------
    error_agent = ErrorAnalysisAgent()
    error_analysis = error_agent.run(
        best_pipeline,
        X_test,
        y_test,
        task_type,
    )

    # --------------------------------------------------
    # Decision intelligence
    # --------------------------------------------------
    decision_agent = DecisionAgent(llm=None)
    decision_report = decision_agent.run(
        task_type=task_type,
        modeling_best=modeling_results["best_model"],
        overall_metrics=overall_metrics,
        segment_summary=segment_summary,
        explainability=explainability,
        error_analysis=error_analysis,
    )

    # --------------------------------------------------
    # Remove non-serializable objects
    # --------------------------------------------------
    modeling_results["best_model"] = {
        k: v
        for k, v in modeling_results["best_model"].items()
        if k != "pipeline"
    }

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------
    return {
        "dataset": str(path),
        "target": target,
        "task_type": task_type,
        "splits": {
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
        },
        "eda": eda_results,
        "modeling": modeling_results,
        "evaluation": {
            "overall_metrics": overall_metrics,
            "threshold": decision_threshold,
            "threshold_info": threshold_info,
        },
        "week3": {
            "segmentation": segment_summary,
            "explainability": explainability,
            "error_analysis": error_analysis,
            "decision_report": decision_report,
        },
    }


# --------------------------------------------------
# CLI entry point
# --------------------------------------------------
if __name__ == "__main__":
    res = run_pipeline(
        "datasets/telco_churn.csv",
        target_override="Churn",
    )
    print(res["week3"]["decision_report"])
