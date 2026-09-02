"""LAKV RLM module — Phase 1 (see docs/research/RLM_LAKV_INTERFACE.md).

Public API:
    from lakv.rlm import RLMEngine, RLMConfig, RLMResult
    from lakv.rlm.llm_backend import HFBackend, InferenceRequest, InferenceResult

    engine = RLMEngine(backend, config)
    result = engine.run(context_text, query, tokenizer=tokenizer)
    result.answer / result.trace / result.metrics / result.termination_reason
"""

from lakv.rlm.config import RLMBudget, RLMConfig
from lakv.rlm.engine import RLMEngine, RLMResult
from lakv.rlm.tracing import RLMTrace

__all__ = ["RLMEngine", "RLMResult", "RLMConfig", "RLMBudget", "RLMTrace"]
