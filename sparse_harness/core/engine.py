"""
Sparse Execution Engine: Coordinates shortest-path DAG traversal,
manages active vs. dormant nodes, and executes early-exit validation gates.
"""

from typing import Dict, List, Any, Optional
import time
from .state import StateEnvelope
from .router import SemanticRouter
from ..nodes.base import BaseNode

class SparseEngine:
    """
    Core runtime orchestrator for the Sparse Node Harness.
    Dynamically routes tasks along shortest paths, firing only the requisite micro-nodes.
    """
    def __init__(self, router: Optional[SemanticRouter] = None):
        self.router = router or SemanticRouter()
        self.nodes: Dict[str, BaseNode] = {}

    def register_node(self, node: BaseNode) -> None:
        """Registers a functional node into the available graph topology."""
        self.nodes[node.node_id] = node

    def execute(self, user_query: str, task_id: str = "task_001") -> Dict[str, Any]:
        """
        Executes a user query through the dynamically routed subnetwork.
        Returns the final state envelope along with comprehensive systems telemetry.
        """
        start_time = time.perf_counter()
        
        # 1. Semantic Shortest-Path Routing Pass
        category, active_path, confidence = self.router.route(user_query)
        
        # Initialize isolated state envelope
        state = StateEnvelope(
            task_id=task_id,
            intent=category,
            payload={"query": user_query},
            metadata={"router_confidence": confidence, "active_path": active_path}
        )

        active_nodes_fired = []
        dormant_nodes = [nid for nid in self.nodes.keys() if nid not in active_path]
        total_tokens = 0
        total_flops = 0

        # Router compute footprint (~10M parameters)
        total_flops += 2 * 10_000_000 * 30

        # 2. Sequential / Branching Node Execution
        for node_id in active_path:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                active_nodes_fired.append(node_id)
                state = node.execute(state)
                total_tokens += node.total_tokens_consumed
                total_flops += 2 * node.parameter_count * max(1, node.total_tokens_consumed)

        latency = time.perf_counter() - start_time
        state.status = "COMPLETED"

        return {
            "task_id": task_id,
            "category": category,
            "result": state.verified_result or state.metadata.get("final_output", "Success"),
            "active_nodes": active_nodes_fired,
            "dormant_nodes": dormant_nodes,
            "total_tokens_processed": total_tokens,
            "estimated_flops": total_flops,
            "latency_seconds": round(latency, 4),
            "state_envelope": state.model_dump()
        }
