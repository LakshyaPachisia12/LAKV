"""Cache corruption operators for the dose-response experiment.

Design: LAKV_runs/malformed_cache_design.md (UNTESTED ON GPU; CPU unit tests only).

Four operators, each at a strength p given in percent, applied to a relayed
KV cache of L positions (every layer's K and V together):

  trunc     remove the LAST m positions                  m = round(p/100 * L)      length L - m
  pad       append m all-zero positions after the real   m = round(p/(100-p) * L)  length L + m
            ones; the zero positions are a fraction p of the FINAL cache
  fill      set the LAST m positions to zero, same length m = round(p/100 * L)     length L
  fillhead  set the FIRST m positions to zero            m = round(p/100 * L)      length L

The operators are deterministic (no randomness). Config / mode names:

  <target>_<op><pct>      e.g. real_pad25, lm_trunc50, real_fillhead25
  target "real": the operator is applied to the real relayed cache.
  target "lm":   it is applied to a clean length-matched wrong cache (the
                 length_matched donor, injected unmodified before the operator).

Per-sample logging: every substitution records the operator, strength, the
cache length before/after, the modified position range, the max |K|/|V| of
that range before and after, a content checksum of the resulting cache, and a
crc32-derived per-sample seed (stable across processes; used for the
determinism audit and, in the length-matched fan-in / delegation modes, to
draw the donor).
"""
from __future__ import annotations

import re
import zlib
from typing import Optional, Tuple

import torch

OPS = ("trunc", "pad", "fill", "fillhead")
TARGETS = ("real", "lm")
MODE_RE = re.compile(r"^(real|lm)_(fillhead|trunc|pad|fill)(\d{1,2})$")
# config-name tail: "..._audit_<mode>"
CONFIG_SUFFIX_RE = re.compile(r"_audit_((?:real|lm)_(?:fillhead|trunc|pad|fill)\d{1,2})$")

KVTuple = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


def stable_seed(*parts) -> int:
    """Process-independent seed (zlib.crc32); Python's hash() is randomized per process."""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0x7FFFFFFF


def is_corruption_mode(mode: str) -> bool:
    m = MODE_RE.match(mode or "")
    return bool(m) and 1 <= int(m.group(3)) <= 99


def parse_corruption_mode(mode: str) -> Tuple[str, str, int]:
    m = MODE_RE.match(mode or "")
    if not m:
        raise ValueError(f"not a corruption mode: {mode!r}")
    target, op, pct = m.group(1), m.group(2), int(m.group(3))
    if not (1 <= pct <= 99):
        raise ValueError(f"corruption strength must be 1..99 percent, got {pct}")
    return target, op, pct


def uses_length_matched_donor(mode: str) -> bool:
    return is_corruption_mode(mode) and parse_corruption_mode(mode)[0] == "lm"


def _len(kv: KVTuple) -> int:
    return int(kv[0][0].shape[2])


def modified_positions(op: str, length: int, pct: int) -> Tuple[int, int, int]:
    """(m, start, end) of the modified range in the OUTPUT cache's positions.
    For trunc, the range is the removed range of the INPUT cache."""
    if op == "trunc":
        m = min(round(pct / 100.0 * length), length - 1)  # always keep >= 1 position
        return m, length - m, length
    if op == "pad":
        m = round(pct / (100.0 - pct) * length)
        return m, length, length + m
    if op == "fill":
        m = min(round(pct / 100.0 * length), length)
        return m, length - m, length
    if op == "fillhead":
        m = min(round(pct / 100.0 * length), length)
        return m, 0, m
    raise ValueError(f"unknown operator {op!r}")


def cache_checksum(kv: KVTuple) -> int:
    """crc32 over the raw bytes of layer 0's K and the last layer's V (a cheap
    but content-sensitive fingerprint used to audit determinism across runs)."""
    k0 = kv[0][0].detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    vl = kv[-1][1].detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return zlib.crc32(vl, zlib.crc32(k0)) & 0xFFFFFFFF


def _max_abs(kv: KVTuple, start: int, end: int) -> Tuple[float, float]:
    if end <= start:
        return 0.0, 0.0
    mk = max(float(k[:, :, start:end, :].abs().max()) for k, _ in kv)
    mv = max(float(v[:, :, start:end, :].abs().max()) for _, v in kv)
    return mk, mv


def apply_corruption(kv: KVTuple, op: str, pct: int) -> Tuple[KVTuple, dict]:
    """Apply one operator to a cache; returns (new_kv, info). Never mutates `kv`."""
    L = _len(kv)
    m, start, end = modified_positions(op, L, pct)
    if op == "trunc":
        out = tuple((k[:, :, : L - m, :].contiguous(), v[:, :, : L - m, :].contiguous()) for k, v in kv)
        before = _max_abs(kv, start, end)
        after = (0.0, 0.0)
    elif op == "pad":
        out = tuple((torch.cat([k, k.new_zeros(k.shape[0], k.shape[1], m, k.shape[3])], dim=2),
                     torch.cat([v, v.new_zeros(v.shape[0], v.shape[1], m, v.shape[3])], dim=2)) for k, v in kv)
        before, after = (0.0, 0.0), (0.0, 0.0)
    else:  # fill / fillhead
        outl = []
        for k, v in kv:
            k2, v2 = k.clone(), v.clone()
            k2[:, :, start:end, :] = 0
            v2[:, :, start:end, :] = 0
            outl.append((k2, v2))
        out = tuple(outl)
        before = _max_abs(kv, start, end)
        after = _max_abs(out, start, end)
    info = {
        "op": op, "strength_pct": pct, "n_modified": int(m),
        "length_before": L, "length_after": _len(out),
        "modified_start": int(start), "modified_end": int(end),
        "modified_max_abs_K_before": before[0], "modified_max_abs_V_before": before[1],
        "modified_max_abs_K_after": after[0], "modified_max_abs_V_after": after[1],
    }
    return out, info


def apply_corruption_mode(mode: str, kv: KVTuple, question_key: str, agent_idx: int,
                          audit_seed: int = 0) -> Tuple[KVTuple, dict]:
    """Apply the operator named by `mode` (real_/lm_ prefix already resolved by the
    caller, which supplies the real or the donor cache as `kv`)."""
    target, op, pct = parse_corruption_mode(mode)
    out, info = apply_corruption(kv, op, pct)
    info.update({
        "target": target,
        "sample_seed": stable_seed("corrupt", audit_seed, question_key, mode, agent_idx),
        "cache_checksum": cache_checksum(out),
    })
    return out, info
