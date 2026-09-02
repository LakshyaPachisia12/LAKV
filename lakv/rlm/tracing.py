"""First-class RLM trace (spec section 19) plus per-invocation KV/latency
instrumentation (spec section 18).

Phase 1 scope: everything that can be recorded WITHOUT a real KV-cache
integration point is implemented (call metadata, timing, token counts,
context-access events with revisit counts). Fields that need a wired KV
manager (kv_cache_size, kv_cache_reuse, kv_cache_eviction, peak GPU memory
beyond a cheap torch.cuda call) are present in the schema as None/0 placeholders
so downstream trace consumers don't have to special-case a missing key, and
are documented as not-yet-populated in docs/research/RLM_LAKV_INTERFACE.md.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def new_invocation_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class ContextAccessEvent:
    event: str = "context_access"
    run_id: str = ""
    invocation_id: str = ""
    parent_invocation_id: Optional[str] = None
    depth: int = 0
    chunk_id: Optional[str] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    access_type: str = ""  # e.g. "llm_query", "search_context", "get_chunk"
    query: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    revisit_count: int = 0
    led_to_recursion: bool = False


@dataclass
class InvocationEvent:
    event: str = "invocation"
    run_id: str = ""
    invocation_id: str = ""
    parent_invocation_id: Optional[str] = None
    depth: int = 0
    role: str = ""  # "root" | "sub" | "recursive_root"
    chunk_ids: List[str] = field(default_factory=list)
    input_token_count: int = 0
    output_token_count: int = 0
    model_name: str = ""
    dtype: str = ""
    device: str = ""
    timestamp: float = field(default_factory=time.time)
    latency_s: float = 0.0
    # KV-integration placeholders — populated once RLM is wired to LAKV's KV
    # manager (see docs/research/RLM_LAKV_INTERFACE.md); left explicit here
    # rather than omitted so trace consumers have a stable schema.
    kv_cache_size_bytes: Optional[int] = None
    kv_cache_reused: Optional[bool] = None
    kv_cache_recompute_tokens: Optional[int] = None
    peak_memory_mb: Optional[float] = None


@dataclass
class TerminationEvent:
    event: str = "termination"
    run_id: str = ""
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    detail: Optional[str] = None


class RLMTrace:
    """Accumulates events for one RLM run (root + every recursive child
    share the SAME trace instance, distinguished by invocation_id/depth)."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or new_run_id()
        self.events: List[dict] = []
        self._chunk_access_counts: Dict[str, int] = {}

    def record_context_access(
        self,
        invocation_id: str,
        parent_invocation_id: Optional[str],
        depth: int,
        chunk_id: Optional[str],
        token_start: Optional[int],
        token_end: Optional[int],
        access_type: str,
        query: Optional[str] = None,
        led_to_recursion: bool = False,
    ) -> ContextAccessEvent:
        revisit_count = 0
        if chunk_id is not None:
            revisit_count = self._chunk_access_counts.get(chunk_id, 0)
            self._chunk_access_counts[chunk_id] = revisit_count + 1

        ev = ContextAccessEvent(
            run_id=self.run_id,
            invocation_id=invocation_id,
            parent_invocation_id=parent_invocation_id,
            depth=depth,
            chunk_id=chunk_id,
            token_start=token_start,
            token_end=token_end,
            access_type=access_type,
            query=query,
            revisit_count=revisit_count,
            led_to_recursion=led_to_recursion,
        )
        self.events.append(asdict(ev))
        return ev

    def record_invocation(self, **kwargs) -> InvocationEvent:
        ev = InvocationEvent(run_id=self.run_id, **kwargs)
        self.events.append(asdict(ev))
        return ev

    def record_termination(self, reason: str, detail: Optional[str] = None) -> TerminationEvent:
        ev = TerminationEvent(run_id=self.run_id, reason=reason, detail=detail)
        self.events.append(asdict(ev))
        return ev

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps(ev) + "\n")

    def context_access_events(self) -> List[dict]:
        return [e for e in self.events if e["event"] == "context_access"]

    def invocation_events(self) -> List[dict]:
        return [e for e in self.events if e["event"] == "invocation"]
