from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineState:
    run_id: str
    file_path: str
    filename: str
    mode: str = "deterministic"
    status: str = "created"
    current_stage: Optional[str] = None
    target_override: Optional[str] = None
    dataset_profile: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, Any] = field(default_factory=dict)
    task_info: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
