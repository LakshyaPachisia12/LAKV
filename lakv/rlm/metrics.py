"""Metrics computed from an RLMTrace (spec sections 20, 31).

Phase 1 scope: the metrics that are well-defined from a single trace without
extra design work (frequency, revisit rate, coverage, depth/fan-out,
call counts). Deferred to a later phase, once there's a trace worth analyzing:
reuse-distance histograms, temporal-reuse time series, spatial-locality
scoring, and working-set-over-time windowing — each needs a windowing/
binning design decision that should be informed by real trace data, not
guessed at up front (see docs/research/RLM_LAKV_INTERFACE.md).
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from lakv.rlm.tracing import RLMTrace


def compute_metrics(trace: RLMTrace, total_context_tokens: Optional[int] = None) -> dict:
    access_events = trace.context_access_events()
    invocation_events = trace.invocation_events()

    chunk_access_counter: Counter = Counter(
        e["chunk_id"] for e in access_events if e["chunk_id"] is not None
    )
    total_accesses = sum(chunk_access_counter.values())
    unique_chunks = len(chunk_access_counter)
    repeated_accesses = sum(c - 1 for c in chunk_access_counter.values() if c > 1)
    revisit_rate = (repeated_accesses / total_accesses) if total_accesses else 0.0

    accessed_tokens = 0
    seen_ranges = set()
    for e in access_events:
        ts, te = e.get("token_start"), e.get("token_end")
        if ts is not None and te is not None and (ts, te) not in seen_ranges:
            seen_ranges.add((ts, te))
            accessed_tokens += max(0, te - ts)
    coverage = (accessed_tokens / total_context_tokens) if total_context_tokens else None

    depth_counts: Counter = Counter(e["depth"] for e in access_events)
    max_depth_accessed = max(depth_counts) if depth_counts else 0

    # fan-out: for each parent invocation, how many child invocations it spawned
    children_by_parent: Dict[str, int] = defaultdict(int)
    for e in invocation_events:
        parent = e.get("parent_invocation_id")
        if parent:
            children_by_parent[parent] += 1
    fan_outs = list(children_by_parent.values())
    avg_fan_out = (sum(fan_outs) / len(fan_outs)) if fan_outs else 0.0

    n_calls = len(invocation_events)
    n_recursive_calls = sum(1 for e in invocation_events if e.get("role") == "recursive_root")
    max_depth_invocations = max((e["depth"] for e in invocation_events), default=0)

    total_input_tokens = sum(e.get("input_token_count", 0) for e in invocation_events)
    total_output_tokens = sum(e.get("output_token_count", 0) for e in invocation_events)
    total_latency_s = sum(e.get("latency_s", 0.0) for e in invocation_events)

    return {
        "calls": n_calls,
        "recursive_calls": n_recursive_calls,
        "max_depth": max_depth_invocations,
        "unique_chunks_accessed": unique_chunks,
        "total_chunk_accesses": total_accesses,
        "repeated_accesses": repeated_accesses,
        "revisit_rate": round(revisit_rate, 4),
        "coverage": round(coverage, 4) if coverage is not None else None,
        "accessed_tokens": accessed_tokens,
        "depth_distribution": dict(depth_counts),
        "max_depth_context_access": max_depth_accessed,
        "avg_fan_out": round(avg_fan_out, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_latency_s": round(total_latency_s, 4),
        "chunk_access_frequency": dict(chunk_access_counter),
        "_deferred_metrics": [
            "reuse_distance", "temporal_reuse", "spatial_locality",
            "working_set_over_time", "recursion_local_working_set",
        ],
    }
