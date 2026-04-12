import uuid
from pathlib import Path
from typing import Any

from orchestrator.state import PipelineState
from services.artifact_store import load_run_state, save_run_state
from dataclasses import asdict


def create_run(file_path: str, target_override: str | None = None) -> PipelineState:
    run_id = str(uuid.uuid4())
    state = PipelineState(
        run_id=run_id,
        file_path=file_path,
        filename=Path(file_path).name,
        target_override=target_override,
    )
    return state


def _persist_state(state: PipelineState) -> None:
    save_run_state(state.run_id, asdict(state))


def _append_event(
    state: PipelineState,
    stage: str,
    message: str,
    details: dict | None = None,
) -> None:
    state.current_stage = stage
    state.events.append(
        {
            "stage": stage,
            "message": message,
            "details": details or {},
        }
    )
    _persist_state(state)


def _analysis_progress_event(state: PipelineState, event: dict[str, Any]) -> None:
    _append_event(
        state,
        event["stage"],
        event["message"],
        event.get("details"),
    )


def _finalize_completed_run(state: PipelineState, output: dict[str, Any]) -> None:
    from services.ai_enrichment import build_ai_enrichment

    output["ai"] = build_ai_enrichment(output)
    state.result = output
    state.artifacts["run_pipeline_output"] = output
    state.status = "completed"
    state.current_stage = "done"
    _append_event(
        state,
        "completed",
        "Analysis completed successfully.",
    )


def _resolve_agentic_target(state: PipelineState) -> str | None:
    if state.target_override:
        return state.target_override

    profile = state.artifacts.get("profile_dataset_tool", {})
    if profile.get("target"):
        return profile["target"]

    schema = state.artifacts.get("inspect_schema_tool", {})
    candidates = schema.get("candidate_targets", [])
    if candidates:
        return candidates[0]

    return None


def _build_planner_state(state: PipelineState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "file_path": state.file_path,
        "filename": state.filename,
        "target_override": state.target_override,
        "mode": state.mode,
        "current_stage": state.current_stage,
        "artifacts_available": list(state.artifacts.keys()),
        "errors": state.errors,
        "retrieved_context": state.retrieved_context,
    }


def _enrich_tool_arguments(state: PipelineState, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(arguments)
    enriched["file_path"] = state.file_path

    profile = state.artifacts.get("profile_dataset_tool", {})
    schema = state.artifacts.get("inspect_schema_tool", {})
    candidate_targets = schema.get("candidate_targets", [])

    if tool_name in {"profile_dataset_tool", "train_baseline_models_tool"} and "target" not in enriched:
        if state.target_override:
            enriched["target"] = state.target_override
        elif profile.get("target"):
            enriched["target"] = profile["target"]
        elif candidate_targets:
            enriched["target"] = candidate_targets[0]

    if tool_name == "train_baseline_models_tool":
        if "task_type" not in enriched and profile.get("task_type"):
            enriched["task_type"] = profile["task_type"]
        if "stats" not in enriched and profile.get("statistics"):
            enriched["stats"] = profile["statistics"]

    return enriched


def _execute_deterministic_run(state: PipelineState) -> PipelineState:
    from pipelines.run_analysis import run_pipeline

    _append_event(
        state,
        "queued",
        "Run accepted and queued for analysis.",
        {"filename": state.filename, "mode": state.mode},
    )

    output = run_pipeline(
        state.file_path,
        state.target_override,
        progress_callback=lambda event: _analysis_progress_event(state, event),
    )
    _finalize_completed_run(state, output)
    return state


def _execute_agentic_run(state: PipelineState) -> PipelineState:
    from agents.planner import PlannerAgent
    from core.validation import validate_tool_arguments
    from orchestrator.tool_dispatcher import dispatch_tool
    from pipelines.run_analysis import run_pipeline
    from services.llm_client import get_llm_client

    _append_event(
        state,
        "agentic_boot",
        "Agentic AI mode started. Planning tool sequence.",
        {"filename": state.filename, "mode": state.mode},
    )

    planner = PlannerAgent(get_llm_client())

    for step_number in range(3):
        planner_state = _build_planner_state(state)
        action = planner.plan_next_action(planner_state)

        _append_event(
            state,
            "agent_plan",
            f"Planner selected {action.tool_name}.",
            {
                "step": step_number + 1,
                "tool_name": action.tool_name,
                "reasoning": action.reasoning or "",
            },
        )

        arguments = _enrich_tool_arguments(state, action.tool_name, action.arguments)
        validate_tool_arguments(action.tool_name, arguments)
        tool_result = dispatch_tool(action.tool_name, arguments)
        state.artifacts[action.tool_name] = tool_result

        _append_event(
            state,
            "agent_tool_complete",
            f"Completed {action.tool_name}.",
            {
                "tool_name": action.tool_name,
                "artifact_keys": sorted(tool_result.keys()) if isinstance(tool_result, dict) else [],
            },
        )

    selected_target = _resolve_agentic_target(state)
    _append_event(
        state,
        "agent_handoff",
        "Planner completed. Running full decision analysis with the selected target.",
        {"target": selected_target},
    )

    output = run_pipeline(
        state.file_path,
        selected_target,
        progress_callback=lambda event: _analysis_progress_event(state, event),
    )
    output["agentic"] = {
        "mode": "agentic",
        "selected_target": selected_target,
        "planner_artifacts": {
            key: value for key, value in state.artifacts.items() if key != "run_pipeline_output"
        },
    }
    _finalize_completed_run(state, output)
    return state


def _execute_adaptive_run(state: PipelineState) -> PipelineState:
    from pipelines.run_analysis import run_pipeline
    from services.adaptive_policy import build_adaptive_policy

    _append_event(
        state,
        "adaptive_boot",
        "Adaptive policy engine started. Profiling dataset before selecting a policy.",
        {"filename": state.filename, "mode": state.mode},
    )

    preview_output = run_pipeline(
        state.file_path,
        state.target_override,
        progress_callback=lambda event: _analysis_progress_event(state, event),
    )
    adaptive_policy = build_adaptive_policy(
        preview_output.get("data_profile", {}),
        preview_output.get("task_type", "unknown"),
        preview_output.get("target", "unknown"),
    )

    _append_event(
        state,
        "adaptive_policy",
        "Adaptive policy selected for this dataset.",
        adaptive_policy,
    )

    output = run_pipeline(
        state.file_path,
        preview_output.get("target"),
        progress_callback=lambda event: _analysis_progress_event(state, event),
        policy=adaptive_policy,
    )
    output["adaptive"] = {
        "mode": "adaptive",
        "policy": adaptive_policy,
        "baseline_preview": {
            "recommended_model": (
                preview_output.get("modeling", {}).get("recommended_model", {})
            ),
            "threshold": preview_output.get("evaluation", {}).get("threshold"),
        },
    }
    _finalize_completed_run(state, output)
    return state


def execute_run(state: PipelineState) -> PipelineState:
    try:
        state.status = "running"
        state.current_stage = "running_pipeline"
        if state.mode == "agentic":
            _execute_agentic_run(state)
        elif state.mode == "adaptive":
            _execute_adaptive_run(state)
        else:
            _execute_deterministic_run(state)
    except Exception as e:
        state.status = "failed"
        state.current_stage = "failed"
        state.errors.append(str(e))
        _append_event(
            state,
            "failed",
            "Analysis failed.",
            {"error": str(e)},
        )

    _persist_state(state)
    return state


def execute_run_by_id(run_id: str, file_path: str, target_override: str | None = None) -> PipelineState:
    payload = load_run_state(run_id)
    state = PipelineState(**payload)
    state.file_path = file_path
    state.filename = Path(file_path).name
    state.target_override = target_override
    _persist_state(state)
    return execute_run(state)
