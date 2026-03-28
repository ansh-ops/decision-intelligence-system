from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class RunCreateResponse(BaseModel):
    run_id: str
    filename: str
    status: str
    mode: str = "deterministic"


class RunStartRequest(BaseModel):
    file_path: str
    target_override: Optional[str] = None


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    filename: Optional[str] = None
    mode: str = "deterministic"
    current_stage: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)
