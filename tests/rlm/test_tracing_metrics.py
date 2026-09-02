import json
import tempfile
from pathlib import Path

from lakv.rlm.metrics import compute_metrics
from lakv.rlm.tracing import RLMTrace


def test_revisit_count_increments_per_chunk():
    trace = RLMTrace()
    e1 = trace.record_context_access("inv1", None, 0, "chunk_A", 0, 10, "get_chunk")
    e2 = trace.record_context_access("inv1", None, 0, "chunk_A", 0, 10, "get_chunk")
    e3 = trace.record_context_access("inv1", None, 0, "chunk_B", 10, 20, "get_chunk")
    assert e1.revisit_count == 0
    assert e2.revisit_count == 1
    assert e3.revisit_count == 0


def test_to_jsonl_roundtrip():
    trace = RLMTrace()
    trace.record_context_access("inv1", None, 0, "chunk_A", 0, 10, "get_chunk")
    trace.record_invocation(invocation_id="inv1", parent_invocation_id=None, depth=0,
                             role="root", chunk_ids=[], input_token_count=10,
                             output_token_count=5, model_name="m", dtype="d",
                             device="cpu", latency_s=0.1)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "trace.jsonl")
        trace.to_jsonl(path)
        lines = Path(path).read_text().strip().splitlines()
        assert len(lines) == 2
        parsed = [json.loads(l) for l in lines]
        assert parsed[0]["event"] == "context_access"
        assert parsed[1]["event"] == "invocation"


def test_compute_metrics_coverage_and_fanout():
    trace = RLMTrace()
    trace.record_context_access("root", None, 0, "chunk_A", 0, 100, "llm_query")
    trace.record_context_access("root", None, 0, "chunk_B", 100, 200, "llm_query")
    trace.record_invocation(invocation_id="root", parent_invocation_id=None, depth=0,
                             role="root", chunk_ids=[], input_token_count=50,
                             output_token_count=10, model_name="m", dtype="d",
                             device="cpu", latency_s=0.5)
    trace.record_invocation(invocation_id="child1", parent_invocation_id="root", depth=1,
                             role="recursive_root", chunk_ids=["chunk_A"], input_token_count=20,
                             output_token_count=5, model_name="m", dtype="d",
                             device="cpu", latency_s=0.2)
    m = compute_metrics(trace, total_context_tokens=1000)
    assert m["unique_chunks_accessed"] == 2
    assert m["accessed_tokens"] == 200
    assert m["coverage"] == 0.2
    assert m["avg_fan_out"] == 1.0
    assert m["calls"] == 2
