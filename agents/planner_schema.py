from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class ToolAction(BaseModel):
    tool_name: str = Field(..., description="Name of the backend tool to invoke")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None