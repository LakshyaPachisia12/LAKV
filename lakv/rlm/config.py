"""RLM configuration — every knob a benchmark run needs to be reproducible
from its saved config (spec sections 10, 28-29).

Phase 1 scope: this is the full budget/generation surface, but NOT yet a
REPL/sandbox config (no code-execution knobs) and not yet a KV-backend
selector (that seam is documented, not wired, in docs/research/
RLM_LAKV_INTERFACE.md).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RLMBudget:
    """Global, cross-recursion-tree limits. A single RLMBudget instance is
    shared (by reference) across the root call and every recursive child —
    it is NOT reset per child, per spec section 10's explicit requirement.
    """

    max_root_iterations: int = 20
    max_total_llm_calls: int = 40
    max_recursive_calls: int = 8
    max_depth: int = 1
    max_generated_tokens: int = 8192
    max_input_tokens: int = 200_000
    max_wall_clock_s: float = 300.0
    max_context_accesses: int = 200
    max_repeated_accesses: int = 50

    used_root_iterations: int = field(default=0, repr=False)
    used_total_llm_calls: int = field(default=0, repr=False)
    used_recursive_calls: int = field(default=0, repr=False)
    used_generated_tokens: int = field(default=0, repr=False)
    used_input_tokens: int = field(default=0, repr=False)
    used_context_accesses: int = field(default=0, repr=False)

    def remaining_calls(self) -> int:
        return max(0, self.max_total_llm_calls - self.used_total_llm_calls)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RLMConfig:
    # generation
    sub_max_new_tokens: int = 96
    root_max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    seed: Optional[int] = 0

    # chunking
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 0

    # model/runtime (mirrors run.py's load_model conventions)
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    max_seq_len: Optional[int] = None

    # concurrency (spec section 16) — Phase 1 runs sequentially; this caps a
    # future parallel-dispatch implementation, kept here so configs stay
    # forward-compatible without a schema change later.
    concurrency: int = 1

    budget: RLMBudget = field(default_factory=RLMBudget)

    # cycle detection (spec section 11/27): abort after this many consecutive
    # identical (action_type, args) pairs at the same node.
    max_repeated_identical_actions: int = 3

    def as_dict(self) -> dict:
        d = asdict(self)
        return d
