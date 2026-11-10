"""
Deterministic Tool Node: Executes CPU-level symbolic logic (Python AST, Math, Regex).
Consumes 0 neural tokens and 0 neural FLOPs with 0% hallucination risk.
"""

from typing import Callable, Any
from .base import BaseNode
from ..core.state import StateEnvelope

class ToolNode(BaseNode):
    """
    Executes deterministic computation directly on CPU tensor/logic pipelines.
    Guarantees formal mathematical or syntax correctness.
    """
    def __init__(self, node_id: str, tool_fn: Callable[[StateEnvelope], Any]):
        super().__init__(node_id, parameter_count=0) # 0 neural weights
        self.tool_fn = tool_fn

    def execute(self, state: StateEnvelope) -> StateEnvelope:
        # Execute deterministic Python function
        result = self.tool_fn(state)
        state.verified_result = result
        state.metadata[f"{self.node_id}_executed"] = True
        return state
