"""
Sparse Node Harness: Quickstart Example
Demonstrates how to build a 3-node sparse execution pipeline in 15 lines of code.
"""

from sparse_harness import SparseEngine, SLMNode, ToolNode

# 1. Initialize the Sparse Engine
engine = SparseEngine()

# 2. Register Micro-Nodes
engine.register_node(SLMNode("intent_parser", parameter_count=135_000_000))
engine.register_node(ToolNode("symbolic_calculator", tool_fn=lambda s: 1580.0))
engine.register_node(SLMNode("response_synthesizer", parameter_count=500_000_000))

# 3. Register Large Specialized Nodes (Remain Dormant)
engine.register_node(SLMNode("code_auditor_7b", parameter_count=7_000_000_000))

# 4. Execute Query Along the Shortest Path
query = "A warehouse has 1450 crates, ships 320, and gets 450. What is current stock?"
result = engine.execute(query)

print("Task Result:", result["result"])
print("Active Path:", result["active_nodes"])
print("Dormant Nodes (0 FLOPs):", result["dormant_nodes"])
print(f"Total Tokens: {result['total_tokens_processed']} (vs ~1,200 in dense ReAct)")
