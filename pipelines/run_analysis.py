from pathlib import Path
import math
from typing import Any, Callable
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


def mock_llm(prompt: str) -> str:
    return "Narrative disabled (deterministic mode)."


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    stage: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    if not progress_callback:
        return

    progress_callback(
        {
            "stage": stage,
            "message": message,
            "details": details or {},
        }
    )


def _safe_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return None
            return float(value)
        return float(value)
    except Exception:
        return None


def _round_metrics(metrics: dict) -> dict:
    cleaned = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            cleaned[k] = _round_metrics(v)
        else:
            num = _safe_float(v)
            cleaned[k] = round(num, 4) if num is not None else v
    return cleaned


def _normalize_target(df: pd.DataFrame, target_override: str | None, candidates: list[str]) -> str:
    if target_override:
        normalized = {str(c).strip().lower(): c for c in df.columns}
        requested = target_override.strip().lower()

        if requested not in normalized:
            raise ValueError(
                f"Target '{target_override}' not found. Available columns: {list(df.columns)}"
            )
        return normalized[requested]

    if not candidates:
        raise ValueError("No valid target candidates found.")
    return candidates[0]


def _dataset_profile(df: pd.DataFrame, target: str, task_type: str) -> dict:
    missing = df.isna().sum()
    missing_summary = {
        col: int(cnt)
        for col, cnt in missing[missing > 0].sort_values(ascending=False).head(10).items()
    }

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    class_distribution = None
    imbalance_ratio = None
    if task_type == "binary_classification":
        vc = df[target].value_counts(dropna=False).to_dict()
        class_distribution = {str(k): int(v) for k, v in vc.items()}
        if len(vc) >= 2:
            counts = sorted(vc.values(), reverse=True)
            imbalance_ratio = round(counts[0] / max(counts[-1], 1), 4)

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_columns": int(len(numeric_cols)),
        "categorical_columns": int(len(categorical_cols)),
        "missing_summary": missing_summary,
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
    }


def _extract_feature_importance(best_pipeline, X_train: pd.DataFrame) -> list[dict]:
    try:
        model = best_pipeline
        feature_names = list(X_train.columns)

        if hasattr(best_pipeline, "named_steps"):
            steps = list(best_pipeline.named_steps.values())
            model = steps[-1]

            preprocessor = best_pipeline.named_steps.get("prep")
            if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
                feature_names = list(preprocessor.get_feature_names_out())

        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
            pairs = [
                {"feature": str(name), "importance": round(float(val), 4)}
                for name, val in zip(feature_names, values)
            ]
            pairs.sort(key=lambda x: x["importance"], reverse=True)
            return pairs[:10]

        if hasattr(model, "coef_"):
            coef = model.coef_
            if hasattr(coef, "ndim") and coef.ndim > 1:
                coef = coef[0]
            pairs = [
                {"feature": str(name), "importance": round(abs(float(val)), 4)}
                for name, val in zip(feature_names, coef)
            ]
            pairs.sort(key=lambda x: x["importance"], reverse=True)
            return pairs[:10]

        if hasattr(model, "feature_name_") and hasattr(model, "feature_importances_"):
            pairs = [
                {"feature": str(name), "importance": round(float(val), 4)}
                for name, val in zip(model.feature_name_, model.feature_importances_)
            ]
            pairs.sort(key=lambda x: x["importance"], reverse=True)
            return pairs[:10]

    except Exception as e:
        print(f"Feature importance extraction failed: {e}")

    return []


def _build_leaderboard(modeling_results: dict, task_type: str) -> list[dict]:
    leaderboard = modeling_results.get("leaderboard", [])
    cleaned = []

    for row in leaderboard:
        model_name = row.get("model") or row.get("name") or "Unknown Model"
        metrics = row.get("metrics") or row.get("fit_metrics") or {}

        cleaned_row = {
            "model": model_name,
            "metrics": _round_metrics(metrics),
            "fit_time": _safe_float(row.get("fit_time")),
            "cv_mean": _safe_float(row.get("cv_mean")),
            "cv_std": _safe_float(row.get("cv_std")),
        }

        if task_type == "binary_classification":
            cleaned_row["primary_score"] = (
                _safe_float(metrics.get("f1"))
                or _safe_float(metrics.get("roc_auc"))
                or _safe_float(metrics.get("accuracy"))
                or _safe_float(row.get("cv_mean"))
            )
        else:
            cleaned_row["primary_score"] = (
                _safe_float(metrics.get("r2"))
                or (_safe_float(metrics.get("rmse")) * -1 if _safe_float(metrics.get("rmse")) is not None else None)
                or _safe_float(row.get("cv_mean"))
            )

        cleaned.append(cleaned_row)

    return cleaned


def _choose_recommended_model(leaderboard: list[dict], task_type: str, imbalance_ratio: float | None) -> dict:
    if not leaderboard:
        return {
            "name": None,
            "reason": "No models available for recommendation.",
            "cv_mean": None,
            "cv_std": None,
            "policy": "N/A",
            "metrics": {},
        }

    def get_metric(row, metric_name):
        return _safe_float(row.get("metrics", {}).get(metric_name))

    scored = []

    for row in leaderboard:
        score = -1e9
        reason = "Selected based on strongest overall performance."
        policy = "Overall performance"

        if task_type == "binary_classification":
            if imbalance_ratio and imbalance_ratio > 1.5:
                f1 = get_metric(row, "f1") or 0
                roc_auc = get_metric(row, "roc_auc") or 0
                recall = get_metric(row, "recall") or 0
                score = 0.5 * f1 + 0.3 * roc_auc + 0.2 * recall
                reason = "Chosen because the dataset is imbalanced, so F1 and ROC-AUC were prioritized."
                policy = "Imbalanced classification"
            else:
                roc_auc = get_metric(row, "roc_auc") or 0
                accuracy = get_metric(row, "accuracy") or 0
                f1 = get_metric(row, "f1") or 0
                score = 0.45 * roc_auc + 0.35 * accuracy + 0.2 * f1
                reason = "Chosen for the best balance of ROC-AUC, accuracy, and F1."
                policy = "Balanced classification"
        else:
            r2 = get_metric(row, "r2") or 0
            rmse = get_metric(row, "rmse")
            mae = get_metric(row, "mae")
            rmse_penalty = rmse if rmse is not None else 0
            mae_penalty = mae if mae is not None else 0
            score = 1.0 * r2 - 0.3 * rmse_penalty - 0.2 * mae_penalty
            reason = "Chosen for the best tradeoff between higher R² and lower error."
            policy = "Regression performance"

        scored.append((score, row, reason, policy))

    scored.sort(key=lambda x: x[0], reverse=True)
    _, best_row, reason, policy = scored[0]

    return {
        "name": best_row["model"],
        "reason": reason,
        "primary_score": best_row.get("primary_score"),
        "cv_mean": best_row.get("cv_mean"),
        "cv_std": best_row.get("cv_std"),
        "policy": policy,
        "metrics": best_row.get("metrics", {}),
    }


def _choose_recommended_model_from_policy(
    leaderboard: list[dict],
    task_type: str,
    imbalance_ratio: float | None,
    policy: dict | None,
) -> dict:
    if not policy:
        return _choose_recommended_model(leaderboard, task_type, imbalance_ratio)

    primary_metric = policy.get("primary_metric")
    if not primary_metric:
        return _choose_recommended_model(leaderboard, task_type, imbalance_ratio)

    def get_metric(row, metric_name):
        return _safe_float(row.get("metrics", {}).get(metric_name))

    scored = []
    for row in leaderboard:
        score = get_metric(row, primary_metric)
        if score is None:
            score = _safe_float(row.get("cv_mean"))
        scored.append((score if score is not None else -1e9, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    _, best_row = scored[0]

    return {
        "name": best_row["model"],
        "reason": policy.get("reasoning") or f"Chosen using adaptive policy metric '{primary_metric}'.",
        "primary_score": best_row.get("primary_score"),
        "cv_mean": best_row.get("cv_mean"),
        "cv_std": best_row.get("cv_std"),
        "policy": policy.get("policy_name") or "Adaptive policy",
        "metrics": best_row.get("metrics", {}),
    }


def _format_decision_report(
    task_type: str,
    recommended_model: dict,
    overall_metrics: dict,
    feature_importance: list[dict],
    data_profile: dict,
) -> dict:
    model_name = (
        recommended_model.get("name")
        or recommended_model.get("model")
        or "Best available model"
    )

    primary_metric = None
    for key in ["f1", "roc_auc", "accuracy", "r2", "rmse", "mae"]:
        if key in overall_metrics:
            primary_metric = f"{key.upper() if key == 'roc_auc' else key}: {overall_metrics[key]}"
            break

    top_features = [item["feature"] for item in feature_importance[:3]] if feature_importance else []

    summary = (
        f"The recommended model is {model_name} for this {task_type} task. "
        f"It was selected based on comparative model performance and robustness. "
        f"Primary evaluation signal: {primary_metric or 'N/A'}."
    )

    risks = []
    if data_profile.get("imbalance_ratio") and data_profile["imbalance_ratio"] > 1.5:
        risks.append(
            "Class imbalance is present, so threshold choice and recall/F1 tradeoffs should be monitored carefully."
        )
    if data_profile.get("missing_summary"):
        risks.append(
            "Missing values are present in the dataset, which may reduce model stability if production data quality changes."
        )

    actions = [
        f"Use {model_name} as the default production candidate for further validation.",
        "Monitor performance over time using the selected primary metrics and threshold sensitivity.",
        "Review the decision threshold based on the business cost of false positives versus false negatives.",
    ]

    if top_features:
        actions.append(
            f"Prioritize monitoring for key drivers such as {', '.join(top_features)}."
        )

    return {
        "executive_summary": summary,
        "key_drivers": top_features or ["Top drivers were not available from feature importance extraction."],
        "risks": risks or ["No major data risks were automatically flagged."],
        "recommended_actions": actions,
    }


def run_pipeline(
    csv_path: str,
    target_override: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    policy: dict | None = None,
):
    path = Path(csv_path)
    _emit_progress(
        progress_callback,
        "loading_dataset",
        "Loading uploaded dataset.",
        {"dataset": str(path)},
    )

    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_excel(path)
        except Exception as e:
            raise ValueError(
                f"Could not read file {path}. Only CSV and Excel files are supported."
            ) from e

    candidates = infer_candidate_targets(df)
    target = _normalize_target(df, target_override, candidates)
    _emit_progress(
        progress_callback,
        "target_resolution",
        "Resolved target column and inferred candidate targets.",
        {
            "target": target,
            "candidate_targets": candidates[:10],
        },
    )

    if df[target].dtype == object:
        cleaned = df[target].astype(str).str.strip().str.lower()
        if set(cleaned.unique()) <= {"yes", "no"}:
            df[target] = cleaned.map({"yes": 1, "no": 0})

    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(
            f"Target '{target}' must be numeric after encoding. Found dtype={df[target].dtype}"
        )

    task_type = infer_task_type(df[target])
    stats = compute_statistics(df, target, task_type)
    data_profile = _dataset_profile(df, target, task_type)
    _emit_progress(
        progress_callback,
        "profiling",
        "Computed dataset statistics and profile.",
        {
            "task_type": task_type,
            "rows": data_profile.get("rows"),
            "columns": data_profile.get("columns"),
        },
    )

    eda_agent = EDAAgent(llm=mock_llm)
    eda_results = eda_agent.run(
        infer_schema(df),
        stats,
        target,
        task_type,
    )
    _emit_progress(
        progress_callback,
        "eda",
        "Completed exploratory analysis.",
    )

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, task_type)
    _emit_progress(
        progress_callback,
        "split",
        "Created train, validation, and test splits.",
        {
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
        },
    )

    modeling_results = run_modeling(
        pd.concat([X_train, y_train], axis=1),
        target,
        task_type,
        stats,
        policy=policy,
    )
    _emit_progress(
        progress_callback,
        "modeling",
        "Finished baseline model training.",
        {
            "models": [row.get("model") for row in modeling_results.get("leaderboard", [])],
        },
    )
    for row in modeling_results.get("leaderboard", []):
        _emit_progress(
            progress_callback,
            "model_evaluated",
            f"Evaluated {row.get('model', 'unknown')} pipeline.",
            {
                "model": row.get("model"),
                "cv_mean": row.get("cv_mean"),
                "cv_std": row.get("cv_std"),
                "fit_metrics": row.get("fit_metrics", {}),
            },
        )

    best_pipeline = modeling_results["best_model"]["pipeline"]
    best_pipeline.fit(X_train, y_train)

    decision_threshold = 0.5
    threshold_info = None

    if task_type == "binary_classification":
        y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
        threshold_metric = (policy or {}).get("threshold_metric") or "f1"
        threshold_info = optimize_threshold(y_val, y_val_prob, metric=threshold_metric)
        decision_threshold = threshold_info["best_threshold"]
        _emit_progress(
            progress_callback,
            "thresholding",
            "Optimized classification threshold on validation data.",
            {
                "best_threshold": threshold_info.get("best_threshold"),
                "best_score": threshold_info.get("best_score"),
                "metric": threshold_info.get("metric"),
            },
        )

    if task_type == "binary_classification":
        y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= decision_threshold).astype(int)
    else:
        y_test_pred = best_pipeline.predict(X_test)
        y_test_prob = None

    overall_metrics = compute_overall_metrics(
        task_type,
        y_test,
        y_test_pred,
        y_prob=y_test_prob,
    )
    overall_metrics = _round_metrics(overall_metrics)

    explain_agent = ExplainabilityAgent()
    explainability = explain_agent.run(
        best_pipeline,
        X_train,
        task_type,
    )
    _emit_progress(
        progress_callback,
        "explainability",
        "Generated explainability outputs.",
    )

    segment_summary = None
    if task_type == "binary_classification":
        segment_summary = segment_summary_classification(
            y_test,
            y_test_prob,
            n_bins=10,
        )

    error_agent = ErrorAnalysisAgent()
    error_analysis = error_agent.run(
        best_pipeline,
        X_test,
        y_test,
        task_type,
    )
    _emit_progress(
        progress_callback,
        "error_analysis",
        "Completed error analysis on holdout data.",
    )

    leaderboard = _build_leaderboard(modeling_results, task_type)
    recommended_model = _choose_recommended_model_from_policy(
        leaderboard,
        task_type,
        data_profile.get("imbalance_ratio"),
        policy,
    )
    feature_importance = _extract_feature_importance(best_pipeline, X_train)
    formatted_decision_report = _format_decision_report(
        task_type=task_type,
        recommended_model=recommended_model,
        overall_metrics=overall_metrics,
        feature_importance=feature_importance,
        data_profile=data_profile,
    )

    best_model_serializable = {
        k: v for k, v in modeling_results["best_model"].items() if k != "pipeline"
    }

    modeling_results["best_model"] = best_model_serializable
    modeling_results["leaderboard"] = leaderboard
    modeling_results["recommended_model"] = recommended_model
    modeling_results["adaptive_policy"] = policy or {}
    _emit_progress(
        progress_callback,
        "decision_report",
        "Assembled recommendation and decision report.",
        {
            "recommended_model": recommended_model.get("name"),
        },
    )

    return {
        "dataset": str(path),
        "target": target,
        "task_type": task_type,
        "data_profile": data_profile,
        "splits": {
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
        },
        "eda": eda_results,
        "modeling": modeling_results,
        "evaluation": {
            "overall_metrics": overall_metrics,
            "threshold": round(float(decision_threshold), 4),
            "threshold_info": threshold_info,
        },
        "explainability": {
            "feature_importance": feature_importance,
            "raw": explainability,
        },
        "week3": {
            "segmentation": segment_summary,
            "error_analysis": error_analysis,
            "decision_report": formatted_decision_report,
        },
    }
