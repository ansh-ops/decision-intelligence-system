from dataclasses import asdict
from typing import Any, Dict, List

from services.artifact_store import save_run_state
from orchestrator.tool_dispatcher import dispatch_tool
from agents.planner import PlannerAgent
from services.llm_client import get_llm_client
from rag.retrieval import retrieve_similar_analyses, store_analysis_summary


class PipelineController:
    """
    Multi-step controller for the Autonomous Decision Intelligence workflow.

    Responsibilities:
    - maintain run state
    - retrieve prior analytical context from FAISS memory
    - call the LLM planner to choose the next tool
    - dispatch deterministic backend tools
    - persist run state after every update
    - store a completed run summary back into memory
    """

    def __init__(self, state):
        self.state = state
        self.planner = PlannerAgent(get_llm_client())

    def update(self, status: str = None, stage: str = None) -> None:
        """
        Persist current state after each major step.
        """
        if status:
            self.state.status = status
        if stage:
            self.state.current_stage = stage

        save_run_state(self.state.run_id, asdict(self.state))

    def retrieve_memory_context(self) -> List[Dict[str, Any]]:
        query_parts = [
            self.state.filename or "",
            self.state.target_override or "",
        ]

        inspect_artifact = self.state.artifacts.get("inspect_schema_tool", {})
        cols = inspect_artifact.get("columns", [])
        if cols:
            query_parts.append(" ".join(cols[:10]))

        query_text = " ".join(part for part in query_parts if part).strip()

        if not query_text:
            return []

        retrieved = retrieve_similar_analyses(query_text=query_text, top_k=3)
        self.state.retrieved_context = retrieved
        return retrieved

    def build_planner_state(self) -> Dict[str, Any]:
        """
        Build the structured state passed into the LLM planner.
        """
        retrieved = self.retrieve_memory_context()

        return {
            "run_id": self.state.run_id,
            "file_path": self.state.file_path,
            "filename": self.state.filename,
            "target_override": self.state.target_override,
            "current_stage": self.state.current_stage,
            "artifacts_available": list(self.state.artifacts.keys()),
            "errors": self.state.errors,
            "retrieved_context": retrieved,
        }

    def _extract_task_info_from_artifacts(self) -> None:
        """
        Populate task_info from tool outputs if available.
        """
        profile = self.state.artifacts.get("profile_dataset_tool", {})
        schema_info = self.state.artifacts.get("inspect_schema_tool", {})

        target = profile.get("target")
        task_type = profile.get("task_type")

        if not target and self.state.target_override:
            target = self.state.target_override

        self.state.task_info = {
            "target": target,
            "task_type": task_type,
            "candidate_targets": schema_info.get("candidate_targets", []),
        }

        if "schema" in schema_info:
            self.state.schema = schema_info["schema"]

    def _store_completed_run_in_memory(self) -> None:
        """
        Store a compact analytical summary after a successful run.
        """
        self._extract_task_info_from_artifacts()

        modeling_result = self.state.artifacts.get("train_baseline_models_tool", {})
        best_model = modeling_result.get("best_model", {})
        metrics = modeling_result.get("metrics", {})
        retrieved_count = len(self.state.retrieved_context or [])

        summary_text = f"""
        Dataset: {self.state.filename}
        Target: {self.state.task_info.get('target', 'unknown')}
        Task Type: {self.state.task_info.get('task_type', 'unknown')}
        Best Model: {best_model.get('name', 'unknown')}
        Metrics: {metrics}
        Retrieved Prior Context Count: {retrieved_count}
        """.strip()

        metadata = {
            "run_id": self.state.run_id,
            "filename": self.state.filename,
            "target": self.state.task_info.get("target", "unknown"),
            "task_type": self.state.task_info.get("task_type", "unknown"),
            "best_model": best_model.get("name", "unknown"),
            "metrics": metrics,
        }

        store_analysis_summary(
            summary_text=summary_text,
            metadata=metadata,
        )

    def run(self):
        """
        Main multi-step execution loop.
        """
        try:
            self.update(status="running", stage="planning")

            # Step 1: retrieve similar past analyses before planning
            self.retrieve_memory_context()
            self.update(stage="memory_retrieved")

            # Step 2: iterative planning + tool execution
            # You can increase this loop later if you add more tools/stages
            for _ in range(3):
                planner_state = self.build_planner_state()
                action = self.planner.plan_next_action(planner_state)

                self.state.current_stage = f"tool:{action.tool_name}"
                self.update()

                if action.tool_name == "inspect_schema_tool":
                    action.arguments["file_path"] = self.state.file_path

                if action.tool_name == "profile_dataset_tool":
                    action.arguments["file_path"] = self.state.file_path

                    profile = self.state.artifacts.get("profile_dataset_tool", {})
                    schema_info = self.state.artifacts.get("inspect_schema_tool", {})
                    candidate_targets = schema_info.get("candidate_targets", [])

                    if "target" not in action.arguments:
                        if self.state.target_override:
                            action.arguments["target"] = self.state.target_override
                        elif profile.get("target"):
                            action.arguments["target"] = profile["target"]
                        elif candidate_targets:
                            action.arguments["target"] = candidate_targets[0]
                        else:
                            raise ValueError("No target column available for profile_dataset_tool")

                if action.tool_name == "train_baseline_models_tool":
                    action.arguments["file_path"] = self.state.file_path

                    profile = self.state.artifacts.get("profile_dataset_tool", {})
                    schema_info = self.state.artifacts.get("inspect_schema_tool", {})
                    candidate_targets = schema_info.get("candidate_targets", [])

                    if "target" not in action.arguments:
                        if self.state.target_override:
                            action.arguments["target"] = self.state.target_override
                        elif profile.get("target"):
                            action.arguments["target"] = profile["target"]
                        elif candidate_targets:
                            action.arguments["target"] = candidate_targets[0]
                        else:
                            raise ValueError("No target column available for train_baseline_models_tool")

                    if "task_type" not in action.arguments:
                        if profile.get("task_type"):
                            action.arguments["task_type"] = profile["task_type"]
                        else:
                            raise ValueError("Task type unavailable for train_baseline_models_tool")

                    if "stats" not in action.arguments:
                        if profile.get("statistics"):
                            action.arguments["stats"] = profile["statistics"]
                        else:
                            raise ValueError("Statistics unavailable for train_baseline_models_tool")

                tool_result = dispatch_tool(action.tool_name, action.arguments)
                self.state.artifacts[action.tool_name] = tool_result

                # Keep task metadata fresh as artifacts arrive
                self._extract_task_info_from_artifacts()
                self.update(stage=f"completed:{action.tool_name}")

            # Optional: produce a compact result object for frontend/API use
            evaluation = {}
            if "run_pipeline_output" in self.state.artifacts:
                evaluation = self.state.artifacts["run_pipeline_output"].get("evaluation", {})

            self.state.result = {
                "task_info": self.state.task_info,
                "artifacts": self.state.artifacts,
                "retrieved_context": self.state.retrieved_context,
                "evaluation": evaluation,
}

            # Step 3: store the finished run in FAISS memory
            self._store_completed_run_in_memory()

            self.update(status="completed", stage="done")

        except Exception as e:
            self.state.errors.append(str(e))
            self.update(status="failed", stage="failed")

        return self.state