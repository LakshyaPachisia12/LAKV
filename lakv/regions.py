"""Region boundaries of a relayed chain cache, for the opt-in region-ablation arms.

Hidden setting only (receivers do not see the passages). Hop-k receivers get the cache
    [hop0 prompt | hop0 generated | hop1 prompt | hop1 generated | ... | hop(k-1) generated]
and `LAKVPipeline._run_trace[h]` records, per hop, the prompt ids and `cache_gen_len`, so every
segment boundary is known exactly. Regions:

  passages  : the tokens of the hop-0 prompt strictly inside the "Context:\n" ... "\n\nQuestion: "
              block (a token that straddles a boundary is NOT counted, so it stays visible)
  reasoning : every generated segment of every hop (hop 0 and hop 1 generations at the Finalizer)
  other     : system prompt, question and template tokens of every hop

Masking is done through the existing hidden-position mechanism (2D attention mask = 0, RoPE
positions unchanged), i.e. the same as deleting those keys.
"""
from typing import Dict, List, Sequence, Tuple

CONTEXT_MARKER = "Context:\n"
QUESTION_MARKER = "\n\nQuestion: "
REGION_MASKS = ("passages", "reasoning")  # which region a region-ablation arm HIDES


def passage_token_span(tokenizer, prompt_ids: Sequence[int]) -> Tuple[int, int]:
    """[start, end) token range of the passage text inside the hop-0 prompt.

    Re-tokenises the decoded prompt with offsets and requires the ids to round-trip exactly
    (raises otherwise: a silent mismatch would hide the wrong tokens)."""
    ids = [int(i) for i in prompt_ids]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    c0 = text.find(CONTEXT_MARKER)
    c1 = text.rfind(QUESTION_MARKER)
    if c0 < 0 or c1 < 0 or c1 <= c0:
        raise ValueError("prompt has no 'Context:\n ... \n\nQuestion: ' block; region ablation needs HotpotQA-style prompts")
    lo, hi = c0 + len(CONTEXT_MARKER), c1
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    enc_ids = [int(i) for i in enc["input_ids"]]
    if enc_ids != ids:
        raise ValueError("re-tokenised prompt does not reproduce the recorded prompt ids")
    offs = enc["offset_mapping"]
    inside = [i for i, (a, b) in enumerate(offs) if a >= lo and b <= hi and b > a]
    if not inside:
        raise ValueError("no tokens inside the passage block")
    s, e = inside[0], inside[-1] + 1
    if inside != list(range(s, e)):
        raise ValueError("passage tokens are not contiguous")
    return s, e


def cache_segments(trace: Dict[int, dict], upto_hop: int) -> List[dict]:
    """Segments of the cache a receiver at hop `upto_hop` gets (hops 0..upto_hop-1 relayed)."""
    segs, off = [], 0
    for h in range(upto_hop):
        t = trace[h]
        p, g = len(t["prompt_ids"]), int(t["cache_gen_len"])
        segs.append({"hop": h, "kind": "prompt", "start": off, "end": off + p})
        segs.append({"hop": h, "kind": "gen", "start": off + p, "end": off + p + g})
        off += p + g
    return segs


def region_ranges(mask: str, trace: Dict[int, dict], upto_hop: int, tokenizer,
                  cache_len: int) -> List[Tuple[int, int]]:
    """[start, end) ranges to HIDE for the arm that masks `mask` ('passages' or 'reasoning')."""
    if mask not in REGION_MASKS:
        raise ValueError(f"unknown region {mask!r}; expected one of {REGION_MASKS}")
    segs = cache_segments(trace, upto_hop)
    total = segs[-1]["end"] if segs else 0
    if total != cache_len:
        raise ValueError(f"region layout covers {total} positions but the injected cache has {cache_len}")
    if mask == "reasoning":
        return [(s["start"], s["end"]) for s in segs if s["kind"] == "gen" and s["end"] > s["start"]]
    a, b = passage_token_span(tokenizer, trace[0]["prompt_ids"])
    return [(a, b)]
