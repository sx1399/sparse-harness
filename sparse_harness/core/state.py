"""
State Envelope: Enforces the Information Bottleneck Principle.
Isolates typed variables between nodes to prevent context re-inflation and quadratic token burn.
"""

from typing import Dict, Any, Optional
import json
from pydantic import BaseModel, Field

class StateEnvelope(BaseModel):
    """
    Lightweight, typed state container passed between DAG nodes.
    Discards discursive intermediate conversation tokens.
    """
    task_id: str = Field(default="task_001", description="Unique task identifier")
    intent: Optional[str] = Field(default=None, description="Classified semantic intent")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters & variables")
    verified_result: Optional[Any] = Field(default=None, description="Deterministic tool output")
    status: str = Field(default="PENDING", description="Execution lifecycle status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Trace telemetry (tokens, latency)")

    def to_compact_json(self) -> str:
        """Serializes the state to minimal JSON without redundant whitespaces."""
        return json.dumps(self.model_dump(exclude_none=True), separators=(',', ':'))

    @classmethod
    def from_compact_json(cls, json_str: str) -> "StateEnvelope":
        """Reconstructs state from compact JSON."""
        data = json.loads(json_str)
        return cls(**data)
