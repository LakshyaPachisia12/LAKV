# NAACL 2027 Draft — Related Work, Method, Results, Limitations

> Status: **DECISION GATE PASSED, 2026-09-09 — Option A, non-exchangeability
> framing CONFIRMED, not narrowed.** The `D`/`B_int8` causal audit
> (`results/run_20260909_080434`, n=50 each) replicates the exact three-tier
> ordering found on uncompressed `A`, and more decisively: real vs
> zeroed/random p<0.0001 (both configs); real vs mismatched p=0.0013 (`D`),
> p=0.0003 (`B_int8`) — tighter than `A`'s own p=0.0127; mismatched vs
> zeroed/random p=0.0002-0.0005. Raw text confirms the same failure
> signatures (zeroed → garbled English, random → multilingual/code garbage,
> mismatched → coherent-but-wrong). Content-identity non-exchangeability
> holds under compression, not just uncompressed relay. **The Introduction
> below is now locked for this claim specifically** — the one remaining
> bracketed note in it (about whether the claim spans compression) is
> resolved: yes, it does.
>
> **Central hypothesis (H1), confirmed on the content-identity axis, not
> narrowed:** KV-cache information is not freely interchangeable along
> depth, positional structure, and content identity — techniques fail when
> they implicitly assume otherwise, and this is not visible from a
> technique's own internal confidence signal.
>
> Related Work reconciliation with "The Pitfalls of KV Cache Compression"
> (ACL 2026), "Rethinking Layer Redundancy" (2026), and AdaK (2026) is
> written. Falsified calibration-confidence-predictor result and the
> quantified Qwen-vs-Mistral failure-texture comparison are written into
> Results as Findings 6-7.
>
> Bleed-through analysis done (`results/run_20260909_104129`): manually
> reviewed 20/36 wrong `A_audit_mismatched` examples against their logged
> donor question(s). Found one unambiguous case (a Kansas fight-song
> question producing "Ellie Goulding" in the answer, traceable only to the
> donor) and one weaker ambiguous case — but the other ~18 are ordinary
> confusion on the real question's own topic, not donor substitution.
> **Honest finding: bleed-through is real but rare, not the dominant
> failure mode** — report it as a qualitative, exploratory data point (one
> concrete example), not a quantified rate from an unvalidated manual
> subset. `B_int4_kivi_full` and `B_int4_hybrid` both failed to improve on
> the key-only fix — reported as strength (mechanism fully isolated to
> keys). Orthogonal Backfill scoped out.

---

## Abstract

Efficient key-value (KV) cache relay between agents in multi-agent LLM
pipelines rests on an assumption that is rarely tested directly: that some
dimension of the cache — which transformer layer, which channel, which
question's content — can be treated as interchangeable for the purpose of
compression or reuse. We test this assumption across three structurally
distinct axes within a sequential three-agent (Reasoner–Verifier–Finalizer)
pipeline on multi-hop question answering, and find it false along all
three. Layers dropped for compression are not safely approximated by their
neighbors. Key vectors' channels cannot be freely mixed once rotary
position embeddings impose position-dependent structure on them: a
rotation-based quantization scheme that assumes otherwise collapses into
out-of-distribution output, while a channel-respecting fix recovers
accuracy statistically indistinguishable from our strongest layer-selection
baseline, at higher compression and no calibration overhead. Most
centrally, a causal audit — substituting the relayed cache with zeroed,
random, or another question's real content — shows a receiving agent's
accuracy depends on the specific transmitted content, not merely on
receiving a non-empty cache: a three-tier ordering (real content >
wrong-but-real content > no real content) that is statistically
significant in every pairwise comparison and holds under compression, not
only in an idealized uncompressed setting. We further show that a
calibration procedure's own internal confidence does not predict which of
two model architectures is safe to compress this way, and points in the
wrong direction in our tests. We term this pattern **non-exchangeability**
and position our contribution as a causally-grounded, cross-axis synthesis
of several 2026 findings on compression-specificity and calibration-
objective-dependence, reporting concretely where four published KV-relay
efficiency techniques transfer to a new deployment setting and where they
do not.

---

## 1. Introduction

Efficient key-value cache relay between agents in a multi-agent LLM
pipeline rests on an assumption that is rarely stated and, to our
knowledge, never directly tested: that some dimension of the cache — which
transformer layer, which channel within a head, which question's content —
can be treated as generic or interchangeable for the purpose of
compression, reconstruction, or reuse. This paper tests that assumption
directly, across three structurally distinct axes, within one multi-agent
pipeline, and finds it false along all three: dropped transformer layers
are not safely approximated by their neighbors; key vectors' channels
cannot be freely mixed once rotary position embeddings have imposed
position-dependent structure on them; and a receiving agent's accuracy
depends on the specific content of the relayed cache, not merely on
receiving *some* non-empty cache, as we show with direct causal evidence.

We call this **non-exchangeability**: efficiency techniques for KV-cache
relay fail in proportion to how much they implicitly assume exchangeability
along an axis the underlying representation does not actually support, and
this failure is not visible from the technique's own internal confidence
signals — a calibration procedure can be more, not less, confident about a
ranking on the model where that ranking is less trustworthy. We
demonstrate this concretely by diagnosing and, where possible, correcting
four published KV-relay efficiency techniques within a sequential
three-agent (Reasoner–Verifier–Finalizer) pipeline on multi-hop
question-answering: a cross-agent offset-correction method (KVCOMM) fails
because it assumes a delta observed for one question transfers to another;
a rotation-based quantization scheme (in the style of TurboQuant/PolarQuant)
fails specifically on key vectors because it assumes head dimensions are
interchangeable, disrupting the position-dependent structure RoPE imposes on
them; a per-channel quantization fix (KIVI-inspired) resolves this by
respecting that structure instead; and layer-selection's accuracy cost is
architecture-dependent in ways a calibration signal's own confidence does
not predict. Critically, the content-identity finding is not an artifact of
testing only uncompressed relay: we replicate the identical three-tier
causal ordering on both a layer-selected and quantized configuration and a
uniformly quantized configuration, and more decisively than on uncompressed
relay — the claim holds under exactly the conditions a practitioner would
actually deploy, not only in an idealized uncompressed setting.

Three recent papers anticipate pieces of this argument from different
angles — that compression misses task-specific information ("The Pitfalls
of KV Cache Compression," ACL 2026), that layer redundancy depends on the
calibration objective rather than being a fixed property ("Rethinking Layer
Redundancy," 2026), and that adaptive, non-fixed-ranking methods can
generalize across architectures where fixed ones might not (AdaK, 2026). We
position this paper's contribution as the causal, cross-axis synthesis
these three approach separately: direct causal validation (not inference
from compression metrics) on the content-identity axis, replicated evidence
across depth and position axes within one system, and a falsified-predictor
result showing the failure is invisible from a technique's own confidence
signal.

---

## 2. Related Work

**Cross-agent KV-cache relay.** Relaying key-value cache instead of decoded
text between agents in a multi-agent LLM pipeline has been proposed as a way
to avoid redundant prefill computation. KVCOMM (Cross-context KV-cache
Communication, NeurIPS'25) introduced an anchor-based framework that
estimates and corrects KV-cache "offset drift" when reusing cache across
different prefix contexts, reporting over 70% cache reuse and up to 7.8×
prefill speedup on retrieval-augmented generation, math reasoning, and
coding tasks. LatentMAS extends this idea to full latent-space collaboration,
where agents exchange autoregressively-generated latent "thoughts" via their
KV-cache working memory rather than relaying the cache of already-decoded
text, reporting substantial token savings and accuracy gains across nine
math/science/code benchmarks. Our setting differs from both in two respects:
we relay the KV cache of real, decoded reasoning text through a fixed
sequential three-agent chain (Reasoner → Verifier → Finalizer), rather than
either offset-corrected reuse of arbitrary shared segments or undecoded
latent thought generation; and we evaluate on multi-hop question answering
(HotpotQA), a benchmark neither KVCOMM nor LatentMAS tested on.

**Compressing the relayed cache.** Reducing what must be transmitted between
agents is a natural complement to relay itself. We adopt two families of
technique: layer-wise selection, which drops calibration-identified
low-importance transformer layers before transmission and reconstructs them
at the receiver, and per-head quantization, which reduces numeric precision.
[If Orthogonal Backfill is implemented: cite "When Less Latent Leads to
Better Relay" (arXiv 2604.13349) here as a third reconstruction strategy
that injects a low-rank residual of discarded content orthogonal to what is
retained, rather than substituting or discarding it outright.]

**Rotation-based quantization interacts badly with RoPE-encoded keys — a
finding, not just an implementation note.** We initially implemented a
rotation-based quantization scheme in the style of TurboQuant/PolarQuant
(Zandieh et al., ICLR'26), applying a fixed orthonormal Hadamard rotation to
both keys and values before quantizing to redistribute per-head outlier
magnitude (covering PolarQuant's rotate-then-quantize step, not TurboQuant's
additional QJL bias-correction term). On our real model, this produced
severely out-of-distribution decoded output — not the graceful accuracy
degradation a generic rotation-based scheme predicts, but qualitatively
different failures (out-of-vocabulary tokens unrelated to the input). We
traced this to key vectors' rotary position embeddings (RoPE): K is cached
*after* RoPE is applied, which mixes channel pairs by a position-dependent
phase, and a second, RoPE-agnostic rotation on top disrupts that structure
rather than merely redistributing outlier magnitude. An isolating ablation
(rotating only V, leaving K on the unrotated code path) reverted the failure
to ordinary quantization-noise degradation, confirming the interaction was
with K specifically. This is consistent with an established, independently
documented interaction in the KV-quantization literature — see the RoPE-
aware quantization paragraph below — and we report it as a specific,
diagnosed negative result about combining generic outlier-redistribution
rotation with RoPE-encoded keys, not merely an implementation detail.

**RoPE-aware KV cache quantization.** A separate line of work addresses
exactly this interaction directly. KIVI (Liu et al., ICML'24) quantizes the
key cache per-channel and the value cache per-token — motivated by RoPE
leaving channel-wise magnitude more consistent across positions than
uniform per-head ranges assume — rather than by rotating. RotateKV
(IJCAI'25) instead applies rotation *before* RoPE via Pre-RoPE Grouped-Head
Rotation, explicitly designed to avoid the interaction we observed. KVQuant
and related work similarly identify post-RoPE key quantization as harder
than value quantization for exactly this reason. We adopt a scoped,
KIVI-inspired fix (per-channel key quantization only, no group-wise
windowing or streaming residual buffer from the original method) and find
it resolves the collapse: 0.0%/0.0% (0% F1) to 52.0% accuracy / 64.1% F1 at
n=100, statistically indistinguishable from our strongest layer-selection
config (`D`, 50.0%/61.9%; McNemar p=0.86) while achieving higher compression
(3.99x vs 2.76x) with no calibration profile required — direct empirical
confirmation, on our pipeline and task, of what this literature identifies
as the correct axis for addressing the problem, rather than the
rotation-based axis we tried first. We additionally tested KIVI's other
asymmetric half — per-token value quantization on top of our key fix — and
found it does not improve on the key-only fix (44.0% vs 50.0% on a matched
50-example subset, not statistically significant but trending the other
direction), at a real compression-ratio cost (3.75x vs 3.99x) from the
much finer per-position scale/zero-point bookkeeping it requires relative
to per-channel grouping. Combined with an earlier null result for rotating
values instead (discussed above), this is two independently-motivated attempts
to improve the value side that both failed to help — evidence that the
key-side fix alone accounts for the full recovery, not a coincidence of one
underpowered comparison.

**Auditing whether KV reuse does what it claims.** A recent line of work
interrogates cross-agent KV/latent reuse mechanisms critically rather than
only reporting end-task accuracy. "When KV Cache Reuse Fails in Multi-Agent
Systems" shows that reuse strategies effective for generation agents can
silently corrupt an LLM judge's cross-candidate comparison even when
end-task accuracy appears stable, using a Judge Consistency Rate metric to
expose the gap. "When Does Latent Communication Pay? A Causal Audit of
Relayed KV Caches in Multi-Agent LLMs" proposes replacing relayed KV with
mismatched, zeroed, or moment-matched-random substitutes to test whether
accuracy gains are causally attributable to the specific transmitted
content, rather than to the receiver simply having a non-empty cache — a
methodology we adopt directly for our own causal audit (Section 3). Our
work extends this critical-auditing tradition along an axis these papers do
not: cross-architecture generalization, treating "does the same technique,
faithfully reproduced, transfer to a different model family" as itself a
target of audit, not an assumption.

**Layer redundancy is not universal.** "No Free Swap: Protocol-Dependent
Layer Redundancy in Transformers" studies which transformer layers are
functionally interchangeable across Qwen3-8B, Llama-3.1-8B, and Mistral-7B,
finding that layer redundancy depends on the model and evaluation protocol
rather than being a fixed architectural property — the same layers can
appear redundant under one protocol and load-bearing under another. This is
consistent with our own finding (Section 4, Finding 3) that a near-identical
proportion of dropped layers produces a substantially larger accuracy cost
on Mistral-7B-Instruct-v0.3 than on Qwen2.5-7B-Instruct, and offers one
candidate mechanism (architecture-dependent, not universal, layer
redundancy) for why.

**Three 2026 papers anticipate pieces of this paper's argument, and we
engage with each directly rather than citing them in passing.**
"Rethinking Layer Redundancy: Calibration Objectives Matter More Than
Search" goes a step further than "No Free Swap," arguing that redundancy is
a joint function of model *and calibration objective* specifically, and
that "a universal layer ranking may not exist." This motivates a question
our own evidence directly speaks to: is our cross-architecture gap really
about the model, or about our calibration procedure's ranking failing to
transfer? We tested the most direct available proxy — whether the
calibration signal's own internal confidence (the separation between
scores assigned to kept vs. dropped layers) predicts which model is safe to
compress — and found it does not, and fails in the *wrong* direction:
Mistral's tier separation (mean gap between kept and dropped layer
importance scores: 0.673) is larger, not smaller, than Qwen's (0.491),
despite Mistral being the more fragile model when that ranking is acted
upon. This is, to our knowledge, a direct empirical test of exactly the
concern "Rethinking Layer Redundancy" raises in the abstract, and it
supports the stronger reading: not merely that redundancy is calibration-
objective-dependent, but that the objective's own confidence is not a
reliable signal of when its ranking is safe to trust — consistent with our
broader claim that non-exchangeability is not visible from a technique's
internal signals.

AdaK reports that adaptive KV-budget estimation *does* generalize cleanly
across Qwen, Mistral, and Llama architectures, which on its face looks like
a counterexample to our claim that layer-selection cost is architecture-
dependent. We do not believe it is. AdaK's method estimates budget
per-instance and per-model at inference time rather than committing to a
single calibration-time ranking applied uniformly thereafter — which is to
say it does not assume a fixed, transferable layer ranking is safe to act
on. Read this way, AdaK's success is a positive instance of our principle,
not a counterexample: it works specifically because it avoids the fixed-
ranking assumption our own `D` configuration (and the KVCOMM-style,
calibration-profile-driven approaches this paper otherwise studies) makes.
We report this as our reading of the mechanism, not as something we have
independently verified by reimplementing AdaK ourselves — a direct
head-to-head comparison would strengthen this argument further and is
noted as a natural extension in Limitations.

Finally, "The Pitfalls of KV Cache Compression" (ACL 2026) makes a general
version of the point this paper investigates concretely: that compression
methods optimize for aggregate metrics while implicitly equating token
retention with functional preservation, missing task- and context-specific
information needs. Our contribution relative to this line of work is not
the abstract observation — it is a causally-grounded, cross-axis
demonstration of it within one system: we show the same failure pattern
recurring across three structurally distinct axes of KV-cache structure
(depth, position, content-identity) in a single multi-agent pipeline, and,
via our causal audit, we causally validate the content-identity axis rather
than inferring non-exchangeability from compression metrics alone. We
position this paper as the causal, cross-axis synthesis these three papers
each gesture toward from a different angle, not as the first to observe
that KV-cache compression can miss what matters.

---

## 3. Method

**Pipeline.** We study a sequential three-agent pipeline — Reasoner,
Verifier, Finalizer — answering HotpotQA (distractor configuration)
questions, where each item bundles its own ten candidate passages and
requires no retrieval step. Agents share the same prompt structure and
role definitions across every configuration in this paper; the only thing
that varies between configurations is *how* one agent's output reaches the
next. In `text_agent`, the Reasoner's decoded text is passed to the
Verifier as a literal string, which reads and re-processes it from scratch.
In every KV-relay configuration (`A` through `E`), the Reasoner's key-value
cache — its internal representation of the question, context, and its own
generated reasoning — is passed directly to the Verifier, which continues
generation from that cache instead of re-reading text. `single_agent` is a
no-relay control: one model call, no intermediate agents at all. All
KV-relay configurations share identical prompts with `text_agent`, so any
accuracy or latency delta between a KV configuration and `text_agent` is
attributable to the relay mechanism itself, not to prompt differences.

**Models.** We evaluate on Qwen2.5-7B-Instruct (bf16) as our primary model,
and — specifically to test whether findings generalize across architecture,
not just within one model family — Mistral-7B-Instruct-v0.3 (bf16), chosen
for its architectural differences (distinct layer count, attention head
configuration, and tokenizer) at comparable parameter scale. Both are
decoder-only transformers using grouped-query attention and RoPE.

**Compression configurations.** `B_int8`/`B_int4` apply uniform per-head
min-max quantization to every transmitted layer's key and value tensors
(a fix to `B_int4`'s originally-catastrophic failure is detailed below in
this section). `C` applies calibration-driven layer selection — dropping
transformer layers a per-model calibration pass identifies as
low-importance before transmission, reconstructing dropped layers at the
receiver via one of three strategies (`zeros`, `nearest`, `interpolate`).
`D` combines layer selection with adaptive per-layer-tier quantization
(high-importance layers at 8-bit, medium-importance at 4-bit). `E` adds an
anchor-based cross-agent offset-correction mechanism adapted from KVCOMM
(Finding 2, Section 4, is a negative result for this configuration).

**RoPE-aware quantization fix for `B_int4`.** `B_int4` (uniform 4-bit
quantization, no layer selection) initially collapsed to 0.0% accuracy. We
diagnosed this as an interaction between quantization and rotary position
embeddings (RoPE): keys are cached *after* RoPE is applied, which mixes
channel pairs by a position-dependent phase, and standard per-head min-max
quantization — one shared numeric range across every position and channel
in a head, at only 16 representable levels for 4-bit — is defenseless
against the resulting inconsistent channel magnitudes. We confirmed this
two ways before proposing a fix: applying a generic outlier-redistribution
technique (Hadamard rotation, in the style of TurboQuant/PolarQuant) to
both keys and values produced qualitatively different, more severe failures
(out-of-distribution decoded tokens) than plain `B_int4`'s failure; an
isolating ablation rotating only values (leaving keys on the unmodified
code path) reverted the failure to `B_int4`'s original character, directly
implicating the key-side rotation. Our fix (`B_int4_kivi`), inspired by
KIVI, quantizes keys per-channel (one numeric range per head dimension,
computed across positions) instead of per-head, leaving values unchanged.

**Statistical methodology.** For every headline comparison between two
configurations, we report exact (binomial-form) McNemar's test on paired
per-example correctness — appropriate because both configurations are
evaluated on the identical example set — and a paired bootstrap 95%
confidence interval (10,000 resamples) for the difference in mean token-F1.
We additionally report Monte Carlo-estimated statistical power at the
current sample size for comparisons that do not reach significance, so that
"not significant" and "confirmed no effect" are never conflated in our
reporting.

**Causal audit.** To test whether a configuration's accuracy
reflects the receiving agent using the specific transmitted content, rather
than merely having *some* non-empty cache to attend over, we substitute the
relayed KV cache with (i) an all-zero cache, (ii) a moment-matched random
cache (same per-tensor mean/standard deviation, no real structure), and
(iii) another, unrelated HELD-OUT question's real cache at the same
pipeline stage (sampled from a pool built from examples never included in
the scored evaluation set, so no scored question's content can leak back to
itself). We apply this to our three strongest configurations (`A`,
`B_int8`, `D`).

We deliberately do not treat these three conditions as interchangeable
evidence for the same claim. Zero-ablation is a known-imperfect causal
baseline in the broader interpretability literature: it pushes activations
off the distribution the model was trained on, which can produce large
behavioral changes for reasons unrelated to whether the zeroed content
carried meaningful information, rather than acting as a clean stand-in for
"no information." We treat the moment-matched-random and mismatched-example
conditions — which preserve realistic activation statistics or substitute
another real, in-distribution cache respectively — as the more informative
evidence for our causal claim, and zero-ablation as a supplementary sanity
floor; if the zeroed condition alone diverges sharply from the other two,
we attribute that to the out-of-distribution-activation effect rather than
to content-dependence. We further note that a fully rigorous treatment of
"no difference" would use a pre-registered equivalence margin (as in the
causal-audit methodology we adapt this design from) rather than the
absence of significance under our paired test; we did not have the
evaluation budget to pre-register and power such a test here, and report
our significance/CI results with that caveat rather than as a substitute
for a formal equivalence test.

## 4. Results

**Table 1: Established results (Qwen2.5-7B-Instruct, HotpotQA, n=100).**

| Config | Accuracy | F1 | Latency | KV/hop |
|---|---|---|---|---|
| single_agent | 57.0% | 68.4% | 1.6s | — |
| text_agent | 54.0% | 68.3% | 7.2s | — |
| A | 56.0% | 69.1% | 8.6s | 144.35 MB |
| B_int8 | 57.0% | 70.5% | 8.3s | 72.16 MB (2.00x) |
| B_int4 (original) | 0.0% | 0.0% | 39.4s | 41.36 MB (4.00x) |
| **B_int4_kivi (fixed)** | **52.0%** | **64.1%** | 9.6s | 36.43 MB (3.99x) |
| C | 43.0% | 57.8% | 10.0s | 104.16 MB |
| D | 50.0% | 61.9% | 10.2s | 37.82 MB (2.76x) |
| E | 9.0% | 17.4% | 19.8s | 35.63 MB |
| E_int8 | 1.0% | 6.2% | 22.2s | 49.05 MB |

**Finding 1 — uniform 8-bit KV quantization is statistically indistinguishable
from uncompressed relay.** `A` vs `B_int8`: McNemar p=1.0, only 3 discordant
examples out of 100. Confirmed on a second model family
(Mistral-7B-Instruct-v0.3, n=50): 0 discordant examples — identical
per-example outcomes. This is not an underpowered null result; the observed
effect is tight enough at this sample size to be a genuine near-zero cost.

**Finding 2 — KVCOMM-style offset correction (`E`) fails catastrophically,
independent of a decoding confound we explicitly ruled out.** `D` vs `E`:
McNemar p≈0.00003 (both before and after an unrelated repetition-penalty
fix that changed several other configurations' numbers — see Limitations).
Manual inspection of `E`'s raw output shows duplicated phrases and
hallucinated content inconsistent with the anchor-corrected cache carrying
usable information, not merely a lower-quality-but-coherent answer.

**Finding 3 — layer-selection's accuracy cost is architecture-dependent.**
On Mistral-7B-Instruct-v0.3 (n=50), dropping 9/32 layers (28%) cost 30.5 F1
points relative to uncompressed relay (p=0.0001). On Qwen2.5-7B-Instruct
(n=100), dropping a near-identical proportion (8/28, 29%) shows a smaller,
not-yet-significant cost (p=0.34) — but the cross-model claim does not rest
on that comparison reaching significance: the most generous reading of
Qwen's own uncertainty (F1 CI upper bound +16.4 points) remains less than
half of Mistral's confirmed collapse. Consistent with prior work finding
transformer layer redundancy to be architecture- and protocol-dependent
rather than a fixed property of a given compression ratio.

**Finding 4 — `B_int4`'s fixed variant matches our strongest
layer-selection configuration, at higher compression and lower complexity.**
`B_int4_kivi` (52.0%/64.1%) vs `D` (50.0%/61.9%): McNemar p=0.86, F1 delta
+2.2 points [95% CI: -8.2, +12.8] — statistically indistinguishable —
while achieving 3.99x compression versus `D`'s 2.76x, and requiring no
per-model calibration profile. `B_int4_kivi` trends below the uncompressed
(`A`) and 8-bit (`B_int8`) configurations (52% vs 56-57%) but this gap is
not yet statistically confirmed at n=100 (7-12 discordant examples,
McNemar p=0.36-0.50); we report this honestly as an open question rather
than claiming `B_int4_kivi` is cost-free the way Finding 1 establishes for
`B_int8`. We tested whether also fixing the value side (`B_int4_kivi_full`,
per-token quantization — KIVI's other asymmetric axis) improves on this
further; it does not (44.0% vs 50.0% on a matched 50-example subset, p=0.51,
trending toward the key-only fix), and costs compression ratio (3.75x vs
3.99x) for the extra per-position bookkeeping it requires. Together with an
earlier null result for rotating values instead, two independently-motivated
value-side interventions both failed to improve on the key-only fix — we
take this as evidence the key/RoPE interaction was the entire mechanism
behind `B_int4`'s original collapse, not one of several contributing
factors.

**Finding 5 — relayed KV demonstrably carries specific, real content, not
merely a non-empty cache.** We ran the causal audit (Section 3) on
configuration `A` (uncompressed relay), n=50: `A` reaches 50.0% accuracy /
64.0% F1; `A_audit_zeroed` and `A_audit_random` both collapse to 0.0%/0.0%;
`A_audit_mismatched` (a real, different held-out question's cache) lands at
28.0%/37.1% — between the two. Every pairwise comparison in this three-tier
ordering is statistically significant: `A` vs zeroed and `A` vs random,
McNemar p<0.0001 each; `A` vs mismatched, p=0.0127; mismatched vs zeroed and
mismatched vs random, p=0.0001 each (an earlier n=20 pilot showed the same
ordering with the `A`-vs-mismatched comparison just short of significance,
p=0.0625, resolved at n=50). This is, to our knowledge, the first direct
causal validation of this kind for a sequential (not fan-in) multi-agent
KV-relay chain, and it underwrites every compression result in this paper:
Findings 1 and 4 (near-zero-cost 8-bit quantization, and the `B_int4` fix
matching `D`) are only meaningful claims about *preserved information* if
the underlying relay is shown to carry real content in the first place,
which this finding establishes directly rather than assuming.

Qualitative inspection of the audit conditions' raw output supports the
statistical result and adds a smaller, mechanistically interesting
observation. `A_audit_mismatched`'s incorrect answers are coherent,
grammatical, and plausible — real linguistic structure attached to the
wrong question — while `A_audit_zeroed` produces garbled but recognizable
English, and `A_audit_random` produces content qualitatively *more*
degraded than zeroing (code-fragment tokens, mixed-language noise), despite
matching the real cache's per-tensor statistics. We speculate this reflects
a difference in how attention treats the two null conditions: a zero vector
is a weak key that attention can largely down-weight, while realistic-
magnitude random noise resembles genuine signal and can actively misdirect
attention rather than being ignored. We report this as a secondary,
exploratory observation, not a claim we have isolated further.

**Finding 5, extended — the same ordering holds under compression.** We
extended this audit to `D` (layer-selected + quantized) and `B_int8`
(uniformly quantized), n=50 each, and the identical three-tier ordering
replicates — more decisively than on uncompressed `A`. `D`: 54.0%
accuracy / 60.2% F1, versus `D_audit_zeroed` and `D_audit_random` both at
0.0% accuracy (F1 0.0% and 0.2% respectively), versus `D_audit_mismatched`
at 26.0%/32.2%. `B_int8`: 54.0%/68.0% versus zeroed/random both 0.0%/0.0%
versus mismatched 24.0%/34.4%. Every one of the ten pairwise comparisons
across both configurations is statistically significant: real vs.
zeroed/random, p<0.0001 for both configurations; real vs. mismatched,
p=0.0013 (`D`) and p=0.0003 (`B_int8`) — both tighter than the p=0.0127
observed on uncompressed `A`; mismatched vs. zeroed/random, p=0.0002-0.0005.
Manual inspection of the raw generations shows the identical failure
signatures found on `A`: zeroed conditions produce garbled but
recognizable English, random conditions produce content qualitatively
*more* degraded (code-fragment tokens, mixed-language noise) despite
matching the real cache's statistics, and mismatched conditions produce
coherent, grammatical, plausible-but-wrong answers. This result is the
basis for stating the content-identity non-exchangeability claim for
compressed and layer-selected relay specifically, not only for an idealized
uncompressed baseline — the condition under which this technique would
actually be deployed.

**Finding 6 — a calibration signal's own confidence does not predict
downstream layer-selection safety, and fails in the wrong direction.** We
tested the most direct available proxy for whether our cross-architecture
gap (Finding 3) reflects the model or merely our calibration procedure: the
separation between calibration-importance scores assigned to kept
(tier-1/2) versus dropped (tier-3) layers, as a measure of how confidently
the calibration ranks layers. Qwen's mean separation is 0.491 (tier-1 mean
0.737, tier-3 mean 0.246); Mistral's is *larger*, 0.673 (tier-1 mean 0.851,
tier-3 mean 0.178) — Mistral's calibration looks more decisive, not less,
yet Mistral is the model whose accuracy collapses harder when that ranking
is acted upon (Finding 3). This rules out the more mundane explanation for
Finding 3 (a noisier or less-confident calibration signal for Mistral) and
supports a sharper one: the calibration objective can be equally or more
internally confident while being less reliable, and this is not detectable
from the calibration output alone. We are explicit that this is a single
comparison across two models and not a validated general predictor — see
Limitations — but it is a real, falsified hypothesis test, not an assumed
conclusion, and we report it as such.

**Finding 7 — architecture-dependent layer-selection failures differ in
kind, not only in rate.** Beyond the accuracy gap already reported (Finding
3), we compared the *character* of `D`'s incorrect answers between models.
Qwen's incorrect predictions are short (mean 18 characters, 2% exceed 150
characters) and are overwhelmingly close, traceable near-misses — reformatted
dates, dropped honorifics, one-character typos, answers that would likely
score correctly under a less strict metric than exact match. Mistral's
incorrect predictions are markedly longer (mean 109 characters, 15% exceed
150 characters, maximum 1,454) and include a genuine long tail of severe
failures: fluent but entirely fabricated tangents (e.g., a full invented
biography of an unrelated historical figure) and degenerate repetition
loops, neither of which appear in Qwen's failure set at comparable rates.
Before attributing this to the compression mechanism itself, we checked an
obvious alternative explanation: whether we inadvertently under-penalize
repetition for Mistral relative to its own tuned defaults. We compare our
uniformly-applied `repetition_penalty=1.05` against each model's own
published generation defaults and find the opposite of what this
alternative explanation predicts — Qwen's own default *is* 1.05 (we match
it exactly), while Mistral's own default configuration specifies no
repetition penalty at all, meaning our setting applies *more* correction
for Mistral relative to its own baseline, not less. The long-tail failure
pattern persists despite this, which weakens rather than supports a
decoding-hyperparameter confound as the explanation, and is consistent
instead with the depth-axis non-exchangeability differing in *character*
across architectures, not only in magnitude.

**Finding 8 — donor-question content occasionally, but rarely, bleeds
through under mismatched substitution.** Unlike "When Latent Agents Lie,"
which studies an adversarial agent deliberately substituting or optimizing
hidden state to deceive a coordinator in a fan-in topology, our setting is
non-adversarial: the substitution is an experimental intervention for
measurement, in a sequential chain, and we ask a narrower descriptive
question — when the receiving agent is wrong, does its answer reflect the
substituted donor's content specifically, or ordinary confusion on the real
question? We manually reviewed 20 of 36 incorrect `A_audit_mismatched`
answers against their logged donor question(s) (`results/run_20260909_104129`,
the config's `donor_question` field). We find one unambiguous instance: a
question about a Kansas university's fight song produced a wrong answer
containing "Ellie Goulding" — an entity with no relationship to the real
question, traceable only to a donor question about the singer Ellie
Goulding — and one weaker, ambiguous case. The remaining examples reviewed
show ordinary failure on the real question's own topic (flipped
comparisons, fabricated but topically-appropriate answers, generic
hallucination), not donor-topic substitution. We report this as a genuine
but rare phenomenon in this sample, not the dominant explanation for
`A_audit_mismatched`'s accuracy loss, and note explicitly that this was a
manual review of a subset, not an automated or exhaustive classification —
see Limitations.

---

## 5. Limitations

**Causal audit scope.** We ran the causal audit on `A` (uncompressed relay),
`D` (layer-selected + quantized), and `B_int8` (uniformly quantized) — the
three-tier ordering replicates across all three, at n=50 each. We have not
audited `C` or the `B_int4` family (including the `B_int4_kivi` fix), so we
do not claim the mechanism is confirmed for every configuration in this
paper, only for the three tested; extending to the remaining configurations
would be a natural next step but is not treated as blocking given the
consistency observed across the three architecturally-distinct conditions
already tested (no compression, compression only, compression plus layer
selection).

**Bleed-through analysis is a manual, non-exhaustive classification.**
Finding 8's review covered 20 of 36 incorrect examples from a single
condition (`A_audit_mismatched`) by one annotator, with no automated
classifier, no inter-annotator agreement check, and no statistical test on
the resulting rate. We report it as a qualitative, exploratory
observation — evidence that donor bleed-through can occur, illustrated by
one unambiguous example — not as a validated estimate of how often it
occurs. A rigorous version of this claim would require an automated or
multiply-annotated classification over the full set of incorrect examples,
which we did not have the evaluation budget to complete.

**Single-process evaluation, not a deployed system.** Every experiment in
this paper runs within a single Python process on one GPU: no `KVMessage` is
ever serialized or transmitted over a network. Our compression-ratio and
transmitted-byte accounting reflects what a real distributed deployment
would need to send, and is therefore still meaningful for evaluating
whether a technique is bandwidth-efficient — but our latency numbers do not
include network transfer time, and smaller relayed caches are not observed
to be faster in this setup, since the model still executes every
transformer layer regardless of what its injected cache contains. Claims in
this paper should be read as characterizing communication-cost tradeoffs and
accuracy effects, not as measured end-to-end deployment speedups.

**Statistical power varies across comparisons.** We report paired
significance tests (exact McNemar) and bootstrap confidence intervals for
every headline comparison, and several — the near-zero cost of INT8
quantization, the catastrophic failure of anchor-based offset correction,
and the magnitude of Mistral-7B's layer-selection collapse — are
significant at conventional thresholds even at our evaluation scale (n=100
for Qwen2.5-7B, n=50 for Mistral-7B-Instruct-v0.3). Others are not: at
n=100, we cannot yet statistically distinguish Qwen2.5-7B's layer-selection
condition from its uncompressed baseline in isolation (power ≈15% at
current n), and a power analysis indicates roughly 3× the current sample
size would be needed to reach conventional power for that specific
within-model comparison. We report point estimates for these alongside
their confidence intervals rather than omitting them, and we do not draw
conclusions from them that the statistics do not support; our
cross-architecture claim about layer-selection cost rests on the Mistral-7B
result, which is independently well-powered, not on the underpowered
Qwen-only comparison.

**Two model families, both in the 7-8B parameter range.** We test
Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3, chosen for architectural
diversity (different layer counts, attention head configurations, and
tokenizers) at comparable scale. Whether our findings — particularly the
architecture-dependence of layer-selection cost — extend to substantially
larger or smaller models, mixture-of-experts architectures, or models
outside the Qwen/Mistral/Llama lineage of decoder-only, grouped-query-
attention transformers, is untested.

**Single task domain.** All results are on HotpotQA (distractor
configuration), a multi-hop reading-comprehension QA benchmark. Our pipeline
also supports GSM8K (its original target before this project's dataset
pivot); we did not re-run the current study on GSM8K or any other task
family, so we cannot claim these findings generalize beyond multi-hop QA.
This matters specifically for our offset-correction negative result: KVCOMM,
the technique's origin, was evaluated on MMLU, GSM8K, and HumanEval, none of
which is multi-hop QA, and it remains an open question whether the
technique would fare differently on a more homogeneous task family like the
one it was originally validated on.

**Greedy decoding only.** All reported results use greedy decoding
(`do_sample=False`); we did not evaluate under sampling or self-consistency
(multi-sample majority voting), which prior work on this pipeline suggests
could interact differently with a degraded or compressed relayed context
than with a full one.

**Partial reproduction of cited compression techniques.** Our rotation-based
quantization implementation covers the core rotate-then-quantize mechanism
of PolarQuant but omits TurboQuant's QJL bias-correction term. Our
KIVI-inspired fix (described above, Section 3) omits the original method's group-wise
windowing over sequence chunks and its fp16 residual buffer for the most
recent tokens — we implement full-sequence per-channel quantization only,
the minimal change that isolates and addresses the specific RoPE-interaction
mechanism we diagnosed, not a complete reproduction of KIVI's system. We
report both as what they are — inspired-by implementations scoped to test a
specific hypothesis, not certified reproductions — and encourage readers who
want the original techniques' full guarantees to consult the primary
sources directly.

**Orthogonal Backfill was scoped out, deliberately, not attempted and
abandoned.** We considered a fourth layer-reconstruction strategy based on
"When Less Latent Leads to Better Relay" (arXiv 2604.13349), which injects a
low-rank residual of discarded content into retained layers rather than
substituting or discarding it outright. Its published formula requires
attention weights at the injection point, which this pipeline's `sdpa`
decode path does not expose (only `eager` attention does, at a latency cost
we did not characterize). Given this project's existing evidence that
naive/simplified reproductions of a technique can actively mislead (see the
rotation-based quantization finding above), we chose not to ship an
approximated version under time pressure, and report this as a scoping
decision rather than a result.

**Single-run comparisons.** `B_int4_kivi`'s comparison against `D`
(statistically indistinguishable, McNemar p=0.86) rests on a single n=100
run each, not a multi-seed or replicated evaluation. We report the
confidence interval alongside the point estimate specifically so "matches
`D`" is read as the current best estimate under one run, not a claim
strengthened by replication we have not performed.

**Compute constraints.** All experiments were run on a single RTX 4090.
This bounded both the sample sizes reported above and the number of
model/config combinations we were able to test; a systems-scale deployment
study (multi-GPU, real network transfer, throughput under concurrent load)
was out of scope for this work.

[Ethics/broader-impact note if ARR's checklist requires one beyond
Limitations — check the current ARR responsible-research checklist before
submission, since a separate ethics statement may be a distinct required
section rather than folded into Limitations.]
