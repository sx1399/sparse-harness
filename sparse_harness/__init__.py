"""
Sparse Node Harness (SNH)
A brain-inspired dynamic routing framework for compute-efficient, zero-bloat AI execution.
"""

__version__ = "0.1.0"
__author__ = "Independent Research"

from .core.state import StateEnvelope
from .core.router import SemanticRouter
from .core.engine import SparseEngine
from .nodes.base import BaseNode
from .nodes.slm_node import SLMNode
from .nodes.tool_node import ToolNode

__all__ = [
    "StateEnvelope",
    "SemanticRouter",
    "SparseEngine",
    "BaseNode",
    "SLMNode",
    "ToolNode",
]
