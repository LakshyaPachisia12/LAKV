# LAKV — Future Work Reference

> Purpose: a durable reference for ideas discussed and deliberately **not**
> pursued before the current NAACL/COLING (ARR, Oct 12, 2026) submission —
> not because they're bad ideas, but because they're new scope competing
> against a fixed deadline. Each one has a real, scoped plan below so a
> future session doesn't have to re-derive the reasoning from scratch.
> Nothing here is started. Revisit after the current submission is in.

---

## 1. Heterogeneous-architecture KV relay (Mistral ↔ Qwen3-8B)

**The idea:** relay a KV cache between two *different* model architectures
within one live pipeline (e.g. a Qwen3-8B Reasoner handing its cache to a
Mistral Finalizer), instead of the same model filling every agent role.
KVCOMM itself names this as unexplored ("agents with identical
architectures but different weights," "different attention formulations").
Our own non-exchangeability framing predicts this should fail — a genuine
fourth axis (architecture identity) beyond depth/position/content-identity.

**Why this specific pair:** checked real model configs — Mistral-7B-v0.3
and Qwen3-8B happen to share identical KV tensor shapes (8 KV heads × 128
head_dim, same hidden size) and identical RoPE parameterization (plain
RoPE, theta=1,000,000, no scaling — ruled out as a confound). Shape
compatibility is a coincidence, not something engineered; it's what makes
this pair uniquely testable without building a learned bridge network.

**Directional constraint:** only Qwen3-8B (36 layers) → Mistral (32
layers) works cleanly — inject Qwen3-8B's first 32 layers' KV into
Mistral's full-depth forward pass. The reverse needs a mixed-depth cache
(32 real layers + 4 empty ones) that a standard HF forward pass can't
consume uniformly.

**Plan (standalone diagnostic script, not a `LAKVPipeline`/`evaluator.py`
change):**
1. **VRAM note:** the two models don't fit in 24GB simultaneously (14.5 +
   16.4 GB). Sequential design required: load Qwen3-8B, run the Reasoner
   role on ~15–20 held-out HotpotQA questions, capture each KV cache
   (truncated to 32 layers) to CPU, unload. Then load Mistral, inject each
   saved cache into the Finalizer role, generate, unload.
2. Reuse `LAKVPipeline._to_tuple()` for cache conversion; replicate the
   exact `position_ids`/`attention_mask` construction `_generate()`
   already uses (get this wrong and you've introduced a second bug on top
   of the one you're testing).
3. Baselines on the same questions: Mistral-alone, Qwen3-8B-alone.
4. Analysis is qualitative first (n=15–20, not a statistical claim) —
   classify against vocabulary already established: garbled-but-English
   (`A_audit_zeroed`-like), pure noise (`A_audit_random`-like),
   coherent-but-wrong (`A_audit_mismatched`-like), or something new.

**Known limitations, going in:**
- Directional only (see above) — can't claim "cross-architecture relay"
  broadly, only this one direction for this one pair.
- Truncation confound: dropping Qwen3-8B's last 4 layers by *position*,
  not calibrated importance. Deeper layers are usually considered *more*
  semantically important, not less — a failure could be "architecture-
  incompatible" or partly "wrong end truncated." Can't fully separate
  those without also running a calibration-informed truncation as a
  control.
- This is the single riskiest kind of code in this project's history —
  manual KV injection with hand-built position IDs. `E`'s offset
  correction needed four separate bug fixes within *one* architecture;
  cross-architecture injection is a genuinely novel path with zero
  precedent here. Budget real debugging time, not just the write time.
- Small-n qualitative result only, even in the best case — a real
  statistical version would be a second, larger follow-up.

**Estimated cost:** ~1–2 hours to write carefully, then well under an
hour of GPU time per attempt — but debugging this kind of code rarely
takes one attempt.

---

## 2. Topology variation (fan-in / LatentMAS-style concatenation)

**The idea:** every result in this paper is on a sequential, one-hop-at-a-
time chain (Reasoner → Verifier → Finalizer). Two other topologies exist
in the literature we already cite and could test our causal-audit
methodology against:
- **Fan-in** ("When Latent Agents Lie"): multiple agents feed one
  receiver, adversarially in their setting, non-adversarially in a version
  we could build.
- **Layer-wise concatenation** (LatentMAS): agents' KV caches get
  literally prepended onto each other rather than handed off one at a
  time.

**Why it's tractable:** single model throughout (no cross-architecture
shape/representation risk at all), reuses the compression/audit/stats
machinery entirely. The new code is a pipeline *wiring* variant — e.g. two
Reasoners on the same question, both caches concatenated, fed to one
Finalizer — not new infrastructure.

**What it would test:** does the three-tier causal ordering (real >
mismatched > zeroed/random) still hold when one of *two* concatenated
input caches is substituted, instead of the single relayed cache in our
current sequential design? A clean replication would strengthen the
non-exchangeability claim considerably (three topologies, not one); a
different ordering would be its own interesting finding.

**Limitations:** still real new pipeline code, still needs its own
statistical run once built, doesn't reuse the *exact* causal-audit
substitution logic without adaptation (need to decide which of the two
concatenated caches gets substituted, and whether that changes the
methodology's assumptions).

**Estimated cost:** lower risk than the heterogeneous-relay idea (no
cross-architecture surface area), but comparable build time — a real
pipeline variant, not a quick script.

---

## 3. Adversarial / worst-case donor selection (extends Finding 8)

**The idea:** our causal audit's `_audit_mismatched` condition currently
substitutes a *randomly*-sampled held-out question's KV cache. A stronger
version: substitute the *most topically similar* wrong question instead
(via embedding or lexical similarity), to see if a "near-miss" donor
causes more bleed-through / worse accuracy than a random one.

**Important honesty note, carried over from the original discussion:**
this is **not** the same thing as the adversarial attacks in "When Latent
Agents Lie" (which involves an agent with gradient access deliberately
optimizing its own hidden state to deceive a coordinator). Calling
similarity-based donor selection "adversarial" would overclaim — it's
better framed as "a stronger, designed interference condition," a direct
extension of Finding 8 (bleed-through), not a true adversarial-robustness
result. A *real* adversarial version (gradient-optimized KV perturbation)
is a substantially bigger, separate project — not scoped here.

**Why it's the cheapest of the three:** reuses `causal_audit.py`'s
existing pool/substitution machinery almost entirely — the only new piece
is the donor-selection strategy (similarity-ranked instead of random),
not new pipeline infrastructure.

**Estimated cost:** lowest of the three ideas here — mostly a new
selection function plus a rerun of the existing `_audit_mismatched`
machinery with it swapped in.

---

## 4. RLM (Recursive Language Models) — clarified, not a fit, don't re-litigate

Came up repeatedly this session; settling it here so it doesn't need
re-deriving. RLM (Zhang & Kraska, MIT CSAIL, arXiv:2512.24601) is a root
LM that writes code to call sub-LMs over *text*, keeping large inputs in a
Python REPL's memory specifically to avoid putting them in any model's
attention window. **It never relays a KV cache at all** — there is no
KV-relay mechanism inside it to adapt or compare against ours. What
*does* match "RLM but through KV cache instead of text" is **LatentMAS**,
which we already cite — see Idea #2 above for the actual on-thesis
version of this line of thinking.

---

## 5. AAMAS reframing — what it would actually take (not just "add RLM")

Explored directly: would any of the ideas above, especially RLM-flavored
ones, make this work a good fit for AAMAS (A\*, the actual multi-agent-
systems venue)? **No, and it's a fit problem, not a timeline problem.**
AAMAS reviewers are calibrated for agent-level properties this work
doesn't have: coordination protocols, negotiation, strategic/self-
interested behavior, mechanism design, autonomy, scalability to many
heterogeneous agents. Our pipeline — even extended per ideas #1–3 above —
stays a fixed, cooperative, 3-step relay with identical objectives at
every step. That reads as an NLP efficiency paper, not a multi-agent-
systems paper, regardless of how many LLM calls are involved.

**What would actually close the gap, in increasing order of cost:**

1. **Reframing only** (no new experiments): position the paper around
   "communication protocol robustness between autonomous agents" instead
   of "KV-cache compression efficiency." The causal audit already
   supports this almost as-is. Requires rewriting the introduction/related
   work to engage AAMAS's own LLM-multi-agent-communication subliterature
   (agent negotiation with LLMs, multi-agent debate protocols, LLM agent
   communication languages) instead of leading with the NLP efficiency
   literature (KVCOMM/KIVI/TurboQuant) we currently do.
2. **Structural change** (real new work): agent-level properties the
   system doesn't have now — more than 3 agents in a non-linear topology
   (idea #2 above is a step toward this), heterogeneous roles/objectives
   rather than interchangeable pipeline stages, and ideally a strategic or
   adversarial angle (idea #3, pushed to its full "real adversarial"
   version, is the piece that points most directly at AAMAS territory).

**Read this as a real AAMAS 2028 candidate, not a retrofit of the current
paper.** AAMAS 2027's deadline (Oct 8, 2026) was essentially concurrent
with the current NAACL/COLING push anyway — not reachable this cycle
regardless of fit.

---

## 6. Conference venue reference (researched 2026-09-10/11, CORE2023 + live dates)

Verified directly against the CORE portal (http://portal.core.edu.au/conf-ranks/)
and each venue's own site — not secondhand summaries. Re-verify dates
closer to any actual submission, especially the "unconfirmed" ones below.

| Venue | CORE rank | 2027 dates | Location | Deadline |
|---|---|---|---|---|
| **NAACL** | A | Jun 1–5, 2027 | San Francisco, USA | ARR Oct 12, 2026 (current cycle) |
| **COLING** | B | May 9–14, 2027 | Macau, China | **Same ARR Oct 12, 2026 cycle as NAACL** — commit to one or the other after reviews (~Dec 23, 2026), not before |
| **EMNLP** | A\* | Nov 2027 | Mexico / Central America | Direct Feb 5, 2027; ARR commit Mar 12, 2027 |
| **ACL** | A\* | Aug 17–22, 2027 | TBA | Direct Apr 25, 2027 |
| **EACL** | A | Mar 9–14, 2027 | Athens, Greece | Closed (was Aug 3, 2026) |
| **AAMAS** | A\* | May 3–7, 2027 | Hanoi, Vietnam | Oct 8, 2026 — see §5 above |
| **ICLR** | A\* | Apr 26–30, 2027 | Moscone Center, San Francisco | Sept 25, 2026 |
| **ICML** | A\* | ~Jul 11–17, 2027 (unconfirmed) | Not yet announced | Jan 22, 2027 |
| **NeurIPS** | A\* | Dec 2027 | Europe (city TBA) | May 21, 2027 |
| **AAAI** | A\* | Feb 16–23, 2027 | Montréal, Canada | Passed for this cycle |
| **IJCAI** | A\* | Aug 7–17, 2027 | Kyoto, Japan + Hengqin, China | Check current cycle |
| **AISTATS** | A | Apr 26–28, 2027 | Paris, France | Check current cycle |
| **UAI** | A | Jul 2027 (predicted) | TBA | Predicted, unconfirmed |
| **CoNLL** | B | Usually co-located with EMNLP | — | — |
| **COLM** *(unranked — too new for CORE)* | — | Oct 6–9, 2027 | TBA | ~Mar 2027 (predicted from pattern, unconfirmed) |
| **MLSys** *(unranked)* | — | ~May 17–22, 2027 (tentative) | Bellevue, WA | ~Oct 30, 2026 (unconfirmed) |

**The recommendation as it stood 2026-09-10:** NAACL/COLING now (same
submission, decide later). EMNLP as the real next step if rejected — a
prestige *upgrade*, not a fallback in the usual sense. ACL as the
longer-horizon aim once there's real reviewer feedback to revise against.
COLM worth a serious parallel look at that same point. The general-ML
A\* venues (NeurIPS/ICML/ICLR/AAAI/IJCAI) and AAMAS: hold off — not a
good match for this paper's current shape without the kind of reframing
or restructuring described in §5.

---

## How to use this file

Nothing above is scoped for before the Oct 12, 2026 submission. When
picking this back up: re-check whether the model/library landscape has
moved (new transformers versions, new model releases that might make
heterogeneous relay or topology variants easier or harder), re-verify any
"unconfirmed" dates in §6, and re-read `CLAUDE.md`'s "Research-extensions
findings" section for the current state of what's actually confirmed
before building on top of it.
