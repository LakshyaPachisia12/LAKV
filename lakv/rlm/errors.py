"""Exception hierarchy for the RLM engine (lakv/rlm/).

Every failure that terminates a run (rather than a single step) surfaces as
one of these so run() can always return a best-effort result with a clear
termination reason instead of an unhandled traceback (spec section 11/27:
budget exhaustion or a failed branch must not crash the run).
"""


class RLMError(Exception):
    """Base class for all RLM-specific errors."""


class BudgetExceededError(RLMError):
    """A global execution budget (calls, tokens, time, depth) was hit."""

    def __init__(self, budget_name: str, limit, used):
        self.budget_name = budget_name
        self.limit = limit
        self.used = used
        super().__init__(f"budget '{budget_name}' exceeded: used={used} limit={limit}")


class InvalidActionError(RLMError):
    """The model produced an action that failed schema/semantic validation."""


class RecursionCycleError(RLMError):
    """The same (action_type, args) pair repeated beyond the cycle-detection
    threshold — the model is stuck, not making progress."""


class InvalidChunkError(RLMError):
    """A referenced chunk_id or token range does not exist in the context."""
