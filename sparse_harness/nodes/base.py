"""
Base Node Interface.
All micro-models, deterministic tools, and external adapters inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from ..core.state import StateEnvelope

class BaseNode(ABC):
    """Abstract base class representing a functional vertex in the sparse harness DAG."""
    
    def __init__(self, node_id: str, parameter_count: int = 0):
        self.node_id = node_id
        self.parameter_count = parameter_count
        self.total_tokens_consumed = 0

    @abstractmethod
    def execute(self, state: StateEnvelope) -> StateEnvelope:
        """
        Executes the node logic on the incoming state envelope.
        Returns the modified state envelope.
        """
        pass

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns runtime execution metrics for this node."""
        return {
            "node_id": self.node_id,
            "parameters": self.parameter_count,
            "tokens_consumed": self.total_tokens_consumed
        }
