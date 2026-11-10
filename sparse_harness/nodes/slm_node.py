"""
Local SLM Node: Wraps lightweight models (SmolLM2, Qwen-2.5, Llama-3.2).
"""

from typing import Optional, Callable
import tiktoken
from .base import BaseNode
from ..core.state import StateEnvelope

class SLMNode(BaseNode):
    """
    Executes an inference pass on a micro-specialized language model.
    Operates strictly on isolated state envelopes, preserving context boundaries.
    """
    def __init__(
        self,
        node_id: str,
        parameter_count: int = 135_000_000,
        model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        handler_fn: Optional[Callable[[StateEnvelope], str]] = None
    ):
        super().__init__(node_id, parameter_count)
        self.model_name = model_name
        self.handler_fn = handler_fn
        self.enc = tiktoken.get_encoding("cl100k_base")

    def execute(self, state: StateEnvelope) -> StateEnvelope:
        # Measure input tokens
        input_payload = state.to_compact_json()
        in_tokens = len(self.enc.encode(input_payload))
        self.total_tokens_consumed += in_tokens

        # Execute generation (handler or neural generation)
        if self.handler_fn:
            output_text = self.handler_fn(state)
        else:
            output_text = f"Synthesized from {state.task_id}"

        out_tokens = len(self.enc.encode(output_text))
        self.total_tokens_consumed += out_tokens

        # Update telemetry
        state.metadata[f"{self.node_id}_tokens"] = in_tokens + out_tokens
        state.metadata[f"{self.node_id}_output"] = output_text
        return state
