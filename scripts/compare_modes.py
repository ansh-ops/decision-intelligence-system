import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.state import PipelineState
from services.run_service import execute_run


DEFAULT_MODES = ["deterministic", "agentic", "adaptive"]


def build_state(file_path: str, target_override: str | None, mode: str) -> PipelineState:
    run_id = f"benchmark-{mode}-{int(time.time() * 1000)}"
    return PipelineState(
        run_id=run_id,
        file_path=file_path,
        filename=Path(file_path).name,
        target_override=target_override,
        mode=mode,
    )


def summarize_run(state: PipelineState, elapsed_seconds: float) -> dict:
    result = state.result or {}
    modeling = result.get("modeling", {})
    recommended = modeling.get("recommended_model", {})
    evaluation = result.get("evaluation", {})
    overall_metrics = evaluation.get("overall_metrics", {})
    explainability = result.get("explainability", {})

    top_features = []
    for item in explainability.get("feature_importance", [])[:5]:
        feature = item.get("feature")
        if feature:
            top_features.append(str(feature))

    ai_summary = (
        result.get("ai", {})
        .get("summary", {})
        .get("executive_summary")
    )

    return {
        "mode": state.mode,
        "status": state.status,
        "runtime_seconds": round(elapsed_seconds, 3),
        "target": result.get("target"),
        "task_type": result.get("task_type"),
        "recommended_model": recommended.get("name"),
        "selection_policy": recommended.get("policy"),
        "recommended_reason": recommended.get("reason"),
        "threshold": evaluation.get("threshold"),
        "threshold_metric": (
            evaluation.get("threshold_info", {}) or {}
        ).get("metric"),
        "overall_metrics": overall_metrics,
        "top_features": top_features,
        "event_count": len(state.events or []),
        "error_count": len(state.errors or []),
        "ai_summary": ai_summary,
        "agentic": result.get("agentic"),
        "adaptive": result.get("adaptive"),
    }


def print_summary_table(summaries: list[dict]) -> None:
    headers = [
        "mode",
        "runtime_seconds",
        "recommended_model",
        "selection_policy",
        "threshold",
        "status",
    ]
    rows = []
    for summary in summaries:
        rows.append([
            str(summary.get("mode", "")),
            str(summary.get("runtime_seconds", "")),
            str(summary.get("recommended_model", "")),
            str(summary.get("selection_policy", "")),
            str(summary.get("threshold", "")),
            str(summary.get("status", "")),
        ])

    widths = []
    for idx, header in enumerate(headers):
        cell_width = max(len(header), *(len(row[idx]) for row in rows))
        widths.append(cell_width)

    def fmt_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(fmt_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deterministic, agentic, and adaptive run modes.")
    parser.add_argument("file_path", help="Path to the dataset file")
    parser.add_argument("--target", dest="target_override", default=None, help="Optional target column override")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=DEFAULT_MODES,
        help="Modes to compare",
    )
    parser.add_argument(
        "--output",
        default="artifacts/mode_comparison.json",
        help="Where to save the JSON comparison report",
    )
    args = parser.parse_args()

    summaries = []

    for mode in args.modes:
        print(f"\nRunning mode: {mode}")
        state = build_state(args.file_path, args.target_override, mode)
        start = time.perf_counter()
        final_state = execute_run(state)
        elapsed = time.perf_counter() - start
        summary = summarize_run(final_state, elapsed)
        summaries.append(summary)

        print(
            f"Completed {mode}: status={summary['status']}, "
            f"runtime={summary['runtime_seconds']}s, "
            f"model={summary.get('recommended_model')}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summaries": summaries}, f, indent=2, default=str)

    print("\nComparison Summary")
    print_summary_table(summaries)
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
