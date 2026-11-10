"""
Unit Tests for Sparse Node Harness Engine & Nodes
"""

import pytest
from sparse_harness import SparseEngine, SLMNode, ToolNode, StateEnvelope

def test_state_envelope_serialization():
    envelope = StateEnvelope(task_id="test_01", intent="math", payload={"a": 10, "b": 20})
    compact_json = envelope.to_compact_json()
    assert '"task_id":"test_01"' in compact_json
    
    restored = StateEnvelope.from_compact_json(compact_json)
    assert restored.task_id == "test_01"
    assert restored.payload["a"] == 10

def test_sparse_engine_routing_and_execution():
    engine = SparseEngine()
    engine.register_node(SLMNode("intent_parser", parameter_count=135_000_000))
    engine.register_node(ToolNode("symbolic_calculator", tool_fn=lambda s: 42.0))
    engine.register_node(SLMNode("response_synthesizer", parameter_count=500_000_000))
    engine.register_node(SLMNode("dormant_auditor_7b", parameter_count=7_000_000_000))

    query = "Calculate 15% discount on $200 and return net price."
    output = engine.execute(query)

    assert output["result"] == 42.0
    assert "symbolic_calculator" in output["active_nodes"]
    assert "dormant_auditor_7b" in output["dormant_nodes"]
    assert output["total_tokens_processed"] > 0
    assert output["estimated_flops"] > 0
