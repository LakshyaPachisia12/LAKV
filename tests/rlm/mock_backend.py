"""Deterministic mock LLM backend for RLM tests — no GPU/download needed
(spec section 34: "Use a deterministic mock provider for most tests")."""

from typing import Callable, List, Union

from lakv.rlm.llm_backend import InferenceRequest, InferenceResult

# root prompts are built from prompts/root_system.txt, which always contains
# this line — sub-call prompts (engine._SUB_SYSTEM_PROMPT) never do, so this
# is a reliable way to route a canned response without special test hooks.
_ROOT_PROMPT_MARKER = "Available actions:"


class MockBackend:
    def __init__(self, root_actions: List[str], sub_answer: Union[str, Callable[[], str]] = "mock sub answer"):
        self._root_actions = list(root_actions)
        self._sub_answer = sub_answer
        self.calls = []

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def dtype(self) -> str:
        return "mock"

    @property
    def device(self) -> str:
        return "cpu"

    def generate(self, request: InferenceRequest) -> InferenceResult:
        self.calls.append(request)
        if _ROOT_PROMPT_MARKER in request.system_prompt:
            if self._root_actions:
                out = self._root_actions.pop(0)
            else:
                out = '{"action": "final", "answer": "[mock actions exhausted]"}'
        else:
            out = self._sub_answer() if callable(self._sub_answer) else self._sub_answer

        return InferenceResult(
            text=out,
            input_token_count=len(request.system_prompt.split()) + len(request.user_content.split()),
            output_token_count=len(out.split()),
            latency_s=0.001,
        )
