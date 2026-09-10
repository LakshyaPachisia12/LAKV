# LAKV — Future Work Reference

> Purpose: a durable reference for ideas discussed and deliberately **not**
> pursued before the current NAACL/COLING (ARR, Oct 12, 2026) submission —
> not because they're bad ideas, but because they're new scope competing
> against a fixed deadline. Each one has a real, scoped plan below so a
> future session doesn't have to re-derive the reasoning from scratch.
> Nothing here is started. Revisit after the current submission is in.

---

## 1. Heterogeneous-architecture KV relay — REVISED 2026-09-11 after a deeper
## literature dive changed the picture substantially. Read this whole
## section before starting anything here; the plan below supersedes an
## earlier, now-outdated version of this idea.

**Original framing (now corrected):** the first pass at this idea was "pick
any two models with matching KV shape (we found Mistral-7B-v0.3 ↔
Qwen3-8B), naively inject one's raw KV into the other, see what happens."
That framing assumed this was a low-crowding, largely unexplored gap — a
deeper search found that's no longer true.

**What a deeper search found (2026-09-11) — this is now an actively
contested area, not an open gap:**
- **["Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear
  Mapping for Prefill Reuse"](https://arxiv.org/abs/2608.03893)** (Heo,
  Shafipour, Zhao, Golub, Kamani, Borkar, Chandran, Zardoshti, Darvish
  Rouhani) — **this is the exact "same family, different size" idea,
  already done.** Requires source/target to share KV head count and
  per-head dimension (the same shape constraint we independently found).
  Method: per-head closed-form ridge regression, top-k predictive source
  layers per target layer, RoPE stripped from keys before fitting so the
  map is position-free, fit on 500 calibration sequences. Tested six pairs
  across three families, including **Qwen3-14B→32B specifically**.
  Results: 73–98% of standalone-prefill accuracy retained on four pairs,
  **sharp degradation on two** — and a nonlinear MLP variant recovered up
  to +37pp on the failures. Runs 2.7–25x faster than re-prefill.
- **"A Universal Context-Reuse Layer for Cross-Model KV Sharing"** and
  **"CacheBridge: Efficient Cross-Model KV Cache Transfer"** — two more
  papers on the same problem, same few weeks.
- **Q-KVComm** — same problem, framed explicitly as multi-agent
  communication (our exact framing) with a "heterogeneous model
  calibration system" for cross-size KV translation.
- **[Latent Briefing](https://labs.ramp.com/research/latent-briefing-kv-cache/)**
  (Ramp Labs) — the closest real analog to "RLM but exchanging KV cache
  instead of text": Claude Sonnet 4 orchestrator + Qwen-14B worker,
  real accuracy gains and real token savings. **Important precision, not a
  minor detail:** this is NOT raw cross-model KV tensor injection (Claude
  and Qwen don't share a tokenizer/embedding space, so that's not even
  possible) — it uses the orchestrator's own attention patterns to select
  *which tokens* matter, then passes that selected content across, still
  fundamentally token-based at the model boundary. It's "RLM with a much
  smarter context filter," not literal shared KV state. Worth being
  precise about this distinction if citing it.

**What this means for us, concretely:**
1. **Don't run the naive version as originally planned.** Raw injection
   without a learned mapper is implicitly already shown not to work well
   in 2608.03893 (that's the entire reason a closed-form mapper exists) —
   running it now would reproduce an already-published result with less
   rigor than the people who already published it.
2. **The real, still-open angle:** none of the four papers above apply
   anything like our causal-audit methodology. They all report *downstream
   accuracy retention* — does the receiver perform well — not *causal
   content-dependence* — does the receiver's output actually depend on the
   specific transferred content, the way our `causal_audit.py` methodology
   tests for same-model relay. **Does a "working" cross-model bridge (73–98%
   retention, per 2608.03893) pass the same three-tier causal ordering
   (real > mismatched > zeroed/random) our same-model relay does, or does
   the learned mapper just produce plausible-sounding output regardless of
   whether the transferred content was real?** That's a genuinely different
   question nobody in this set of papers is asking, and it builds directly
   on both bodies of work — extends 2608.03893's transfer method, extends
   our own causal-audit methodology — rather than either.

**A concrete, immediately-testable pairing, found by re-checking our own
models' shapes:** Qwen3-8B (already in this study, already validated
end-to-end after the LongRoPE fix), Qwen3-4B, and Qwen3-1.7B **all share
identical KV shape** (8 KV heads × 128 head_dim) — only layer count
differs (36/36/28). This is the real, RLM-motivated pairing (same family,
different capability tier, matching what an actual cost-efficient
delegation system would want), immediately testable with our existing
infrastructure with zero shape-compatibility engineering needed for the
*naive* pass, before any learned mapper is built.

**Revised step-by-step plan:**
1. **Naive same-shape, same-family injection first** (Qwen3-8B →
   Qwen3-4B or Qwen3-1.7B, no learned mapper), but run it through our
   causal-audit lens, not just accuracy: does the failure look like
   `A_audit_zeroed` (garbled-but-English), `A_audit_random` (pure noise,
   worse than zeroed), or something distinct from either — same-family
   same-shape naive injection failing differently than cross-family
   naive injection would itself be a new, reportable data point, since
   nobody's characterized *that* failure mode specifically.
2. **If step 1 collapses cleanly** (expected, per 2608.03893's implicit
   finding): build a lightweight version of their ridge-regression mapper
   scoped just to this one pair — real additional engineering (calibration
   data, RoPE-stripping, per-layer top-k source selection), a bigger lift
   than the original "diagnostic script" framing assumed.
3. **Once a working bridge exists (naive or mapped), run the full causal
   audit through it** — zeroed/random/mismatched substitution *after* the
   bridge, not instead of it. This is the actual contribution: validating
   (or debunking) whether cross-model KV transfer preserves genuine
   content-dependence, using a methodology the existing transfer-paper
   literature doesn't apply.

**Known limitations, still true:**
- Directional constraint unchanged (bigger model's early layers → smaller
  model, not the reverse — a standard forward pass can't consume a
  mixed-depth cache).
- This is still the riskiest kind of code in this project's history
  (manual KV injection, hand-built position IDs) — now compounded by
  needing a real learned-mapping component if step 1 fails as expected,
  not just a diagnostic script.
- Small-n qualitative result in the naive-injection phase; the
  causal-audit-of-a-bridge phase would need its own proper n and stats,
  a genuinely bigger project than originally scoped.

**Estimated cost:** the naive pass is still ~1–2 hours to write + well
under an hour of GPU time (unchanged). The full plan (through step 3) is
meaningfully bigger than the original one-diagnostic-script estimate —
budget for it as a real follow-up project, not an afternoon.

---

## 2. Topology variation (fan-in / LatentMAS-style concatenation)

**Relationship to RLM and to Idea #1, clarified:** RLM's literal
architecture (root LM calling sub-LMs over *text*, no KV cache exchanged
at all) is not a form of this — see §4. But an "RLM-shaped" system with
KV cache substituted for RLM's text-based delegation — one orchestrator,
several workers, hierarchical rather than linear — collapses exactly into
this section, specifically the fan-in case. And if that RLM-shaped system
is meant to capture *why* RLM's structure is actually useful (cheap
worker models under an expensive orchestrator, not same-model-everywhere),
then it needs Idea #1's heterogeneous-size relay as a prerequisite, not as
an independent, optional extra — same-model fan-in (this section, cheap,
safe) and RLM-motivated fan-in (needs Idea #1 working first) are two
different-cost versions of the same topology idea. Sequence accordingly:
test heterogeneous compatibility (Idea #1) in isolation before building
any hierarchical wiring on top of it — if the communication channel
between differently-sized workers doesn't carry real information, the
topology built on top of it is moot regardless of how well the wiring
itself works.

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
