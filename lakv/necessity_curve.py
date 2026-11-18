"""Passage hiding for the relay-necessity curve.   *** UNTESTED ON GPU (CPU unit tests only) ***

Design: docs/necessity_curve_design.md. Two hiding axes, both config suffixes on base A:

    _hide20 / _hide50 / _hide80 / _hide100   fractional axis (secondary): hide PCT% of each
                                             item's passages, stratified so gold passages are
                                             hidden at the expected rate; _hide100 is an alias
                                             of _nopass (the receiver sees only the question)
    _hidegold1 / _hidegold2                  gold axis (PRIMARY): hide 1 or 2 of the 2 gold
                                             passages; every distractor stays visible

Level 0 is the plain config (no suffix).

The hidden set is a PURE function of (hide_seed, question blob, gold titles, spec): the arm
(real vs dropped) never enters it, so both arms hide exactly the same passages. All randomness
is zlib.crc32 (never Python's per-process-randomized hash()). Hidden sets are nested across
levels (hide20 within hide50 within hide80 within hide100; gold1 within gold2).

Only the RECEIVERS (agents 1 and 2) get the reduced view; the sender (agent 0, the Reasoner)
always sees every passage. The evaluator computes the view and passes it to
LAKVPipeline.run(question, receiver_question=view).
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from lakv.anchor_table import question_key

FRAC_LEVELS = (20, 50, 80, 100)
GOLD_LEVELS = (1, 2)
CONTEXT_PREFIX = "Context:\n"
QUESTION_MARKER = "\n\nQuestion: "
_HIDE_RE = re.compile(r"_hide(gold)?(\d+)$")


@dataclass(frozen=True)
class HideSpec:
    axis: str     # "frac" or "gold"
    level: int    # percent hidden (frac) or number of gold passages hidden (gold)

    @property
    def suffix(self) -> str:
        return f"_hidegold{self.level}" if self.axis == "gold" else f"_hide{self.level}"


def stable_hash(*parts) -> int:
    """Process-independent 31-bit hash (zlib.crc32). Never use built-in hash()."""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0x7FFFFFFF


# ─── config-name parsing ────────────────────────────────────────────────────

def split_hide_suffix(cfg_name: str, audit_suffixes: Iterable[str]) -> Tuple[str, Optional[HideSpec]]:
    """Remove a hide suffix from a config name.

    "<base>[hide suffix][_audit_<mode>]" -> ("<base>[_audit_<mode>]", HideSpec) ; names with no hide
    suffix are returned unchanged with None. `_hide100` is rewritten to the existing `_nopass`
    config (returned core name contains `_nopass`, spec is None), so the two are the same code path
    by construction. Level 0 has no suffix (`_hide0` / `_hidegold0` raise: use the plain config).
    """
    audit = ""
    base = cfg_name
    for suf in audit_suffixes:
        if cfg_name.endswith(suf):
            base, audit = cfg_name[: -len(suf)], suf
            break
    m = _HIDE_RE.search(base)
    if not m:
        return cfg_name, None
    axis = "gold" if m.group(1) else "frac"
    level = int(m.group(2))
    allowed = GOLD_LEVELS if axis == "gold" else FRAC_LEVELS
    if level not in allowed:
        hint = " (level 0 is the plain config, without a suffix)" if level == 0 else ""
        raise ValueError(f"{cfg_name!r}: unsupported hide level {level} on the {axis} axis; "
                         f"allowed {tuple(allowed)}{hint}")
    stem = base[: m.start()]
    if stem.endswith("_nopass"):
        raise ValueError(f"{cfg_name!r}: _nopass cannot be combined with a hide suffix")
    if axis == "frac" and level == 100:
        return stem + "_nopass" + audit, None
    return stem + audit, HideSpec(axis, level)


# ─── blob handling ──────────────────────────────────────────────────────────

def split_blob(blob: str) -> Tuple[List[str], str]:
    """Split a HotpotQA question blob into (passages, "Question: ...").

    Blob format (run.py::load_hotpotqa): "Context:\\n" + "\\n\\n".join("[Title] sentences") +
    "\\n\\nQuestion: <q>". The question part is cut at the LAST "\\n\\nQuestion: ", exactly like
    LAKVPipeline._question_only, so the all-hidden view equals _question_only(blob).
    """
    if not blob.startswith(CONTEXT_PREFIX):
        raise ValueError("not a HotpotQA-style blob (missing 'Context:' prefix)")
    idx = blob.rfind(QUESTION_MARKER)
    if idx == -1:
        raise ValueError("not a HotpotQA-style blob (missing 'Question:' marker)")
    body = blob[len(CONTEXT_PREFIX): idx]
    question_part = blob[idx + 2:]
    parts = body.split("\n\n[")
    passages = [parts[0]] + ["[" + p for p in parts[1:]]
    if not passages[0].startswith("[") or "\n\n".join(passages) != body:
        raise ValueError("could not split the context into '[Title] ...' passages")
    return passages, question_part


def gold_indices(passages: Sequence[str], gold_titles: Sequence[str]) -> List[int]:
    """Indices of the passages whose '[Title] ' prefix is a gold title. Each gold title must match
    exactly one passage; anything else fails loudly (a mis-split blob must never silently change
    which passages are gold)."""
    out = []
    for t in gold_titles:
        hits = [j for j, p in enumerate(passages) if p.startswith(f"[{t}] ")]
        if len(hits) != 1:
            raise ValueError(f"gold title {t!r} matches {len(hits)} passages (expected exactly 1)")
        out.append(hits[0])
    if len(set(out)) != len(out):
        raise ValueError("two gold titles map to the same passage")
    return sorted(out)


def _title_of(passage: str) -> str:
    end = passage.find("] ")
    return passage[1:end] if passage.startswith("[") and end > 0 else passage[:40]


# ─── hidden-set selection ───────────────────────────────────────────────────

def select_hidden(passages: Sequence[str], gold_idx: Sequence[int], spec: HideSpec,
                  key: str, hide_seed: int = 0) -> List[int]:
    """Indices (sorted) of the passages hidden from the receivers. Pure and deterministic."""
    n = len(passages)
    gold = list(gold_idx)
    dist = [j for j in range(n) if j not in set(gold)]
    if spec.axis == "gold":
        g = min(spec.level, len(gold))
        t = g
    else:
        u = stable_hash(hide_seed, key, "u") % 100
        t = (spec.level * n + u) // 100
        g = (spec.level * len(gold) + u) // 100
        g = min(g, len(gold), t)
        if t - g > len(dist):
            g = t - len(dist)
    gold_order = sorted(gold, key=lambda j: stable_hash(hide_seed, key, "g", passages[j]))
    dist_order = sorted(dist, key=lambda j: stable_hash(hide_seed, key, "d", passages[j]))
    return sorted(gold_order[:g] + dist_order[: t - g])


def render_view(blob: str, passages: Sequence[str], question_part: str, hidden: Sequence[int]) -> str:
    """The receivers' user content. Nothing hidden -> the original blob (byte-identical);
    everything hidden -> "Question: <q>" (identical to LAKVPipeline._question_only);
    otherwise visible passages in original order, original format."""
    hid = set(hidden)
    if not hid:
        return blob
    if len(hid) >= len(passages):
        return question_part
    visible = [p for j, p in enumerate(passages) if j not in hid]
    return CONTEXT_PREFIX + "\n\n".join(visible) + QUESTION_MARKER + question_part[len("Question: "):]


def make_receiver_question(blob: str, gold_titles: Optional[Sequence[str]], spec: HideSpec,
                           hide_seed: int = 0) -> Tuple[str, Dict]:
    """(receiver view, info) for one question. `info` is JSON-serialisable and is logged per sample."""
    if gold_titles is None:
        raise ValueError("hide configs need gold_titles on every dataset item; the loader in run.py "
                         "adds them for HotpotQA only")
    passages, question_part = split_blob(blob)
    gidx = gold_indices(passages, gold_titles)
    hidden = select_hidden(passages, gidx, spec, question_key(blob), hide_seed)
    view = render_view(blob, passages, question_part, hidden)
    gset = set(gidx)
    info = {
        "axis": spec.axis, "level": spec.level, "hide_seed": hide_seed,
        "n_passages": len(passages), "n_gold": len(gidx),
        "n_hidden": len(hidden), "n_gold_hidden": sum(1 for j in hidden if j in gset),
        "hidden_idx": list(hidden), "hidden_titles": [_title_of(passages[j]) for j in hidden],
    }
    return view, info


def summarize_hiding(samples: Sequence[dict], spec: HideSpec, hide_seed: int = 0) -> Dict:
    """Realized hiding over a list of dataset items (each with 'question' and 'gold_titles').
    Raises (ValueError) on any item that cannot be processed, so a bad dataset fails on the CPU
    before any model is loaded."""
    infos = [make_receiver_question(s["question"], s.get("gold_titles"), spec, hide_seed)[1] for s in samples]
    n = len(infos)
    return {"spec": spec.suffix, "n": n,
            "mean_passages_hidden": sum(i["n_hidden"] for i in infos) / n if n else 0.0,
            "mean_gold_hidden": sum(i["n_gold_hidden"] for i in infos) / n if n else 0.0,
            "frac_questions_with_gold_hidden": sum(1 for i in infos if i["n_gold_hidden"] > 0) / n if n else 0.0}
