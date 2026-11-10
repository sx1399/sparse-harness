"""
Semantic Router: Determines the shortest computational path across the node DAG.
Operates with an ultra-lightweight footprint (<10M params or cosine similarity).
"""

from typing import List, Dict, Any, Tuple
import re

class SemanticRouter:
    """
    Evaluates incoming tasks against registered node signatures and returns
    the optimal directed execution sub-path.
    """
    def __init__(self, routing_threshold: float = 0.65):
        self.routing_threshold = routing_threshold
        self.routes: Dict[str, List[str]] = {
            "quantitative": ["intent_parser", "symbolic_calculator", "response_synthesizer"],
            "security": ["ast_security_auditor", "patch_generator"],
            "policy": ["policy_clause_extractor", "logic_truth_gate", "response_synthesizer"],
            "api_workflow": ["itinerary_planner", "relational_constraint_filter", "response_synthesizer"],
            "general_fallback": ["intent_parser", "response_synthesizer"]
        }
        self.signatures = {
            "quantitative": [r"\b(calculate|compute|discount|mrr|cost|revenue|kw|kwh|pue|tons|crates|price|yield)\b"],
            "security": [r"\b(vulnerability|sql\s*injection|cwe|tainted|execute|sqlite|flask|endpoint)\b"],
            "policy": [r"\b(policy|reimbursement|claim|lodging|cap|addendum|overrides|approved)\b"],
            "api_workflow": [r"\b(schedule|flight|hotel|itinerary|arrival|booking|venue|ord)\b"]
        }

    def route(self, task_input: str) -> Tuple[str, List[str], float]:
        """
        Determines the shortest active sub-path for a task.
        Returns: (matched_category, list_of_node_ids, confidence_score)
        """
        text_lower = task_input.lower()
        best_category = "general_fallback"
        max_score = 0.0

        for category, patterns in self.signatures.items():
            matches = sum(len(re.findall(p, text_lower)) for p in patterns)
            if matches > 0:
                score = min(1.0, 0.5 + (matches * 0.15))
                if score > max_score:
                    max_score = score
                    best_category = category

        confidence = max_score if max_score > 0 else 0.5
        active_path = self.routes.get(best_category, self.routes["general_fallback"])
        return best_category, active_path, confidence
