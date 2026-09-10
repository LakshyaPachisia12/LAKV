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
baseline, at higher compression and no calibration overhead. Two results
sit at the center of this paper. First, a causal audit — substituting the
relayed cache with zeroed, random, or another question's real content —
shows a receiving agent's accuracy depends on the specific transmitted
content, not merely on receiving a non-empty cache: a three-tier ordering
(real content > wrong-but-real content > no real content) that is
statistically significant in every pairwise comparison and holds under
compression, not only in an idealized uncompressed setting. Second, and
more surprising: a calibration procedure's own internal confidence does
*not* predict which of two model architectures is safe to compress this
way — and points in the wrong direction in our tests, with the more
fragile architecture producing the more confident-looking calibration
signal. The failure that matters is invisible from the inside. We term
this pattern **non-exchangeability**
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
along an axis the underlying representation does not actually support. The
paper's sharpest single result is that this failure is invisible from the
inside: a calibration procedure's own confidence does not predict which of
two model architectures is safe to compress along the depth axis, and
fails in the *wrong* direction — the architecture that collapses harder
under layer-selection produces the *more* confident-looking signal
(Section 4, Finding 6). We demonstrate the broader pattern by diagnosing
and, where possible, correcting four published KV-relay efficiency
techniques within our pipeline: a cross-agent offset-correction method
(KVCOMM) fails because it assumes a delta observed for one question
transfers to another; a rotation-based quantization scheme (TurboQuant and
PolarQuant-style) fails specifically on key vectors because it assumes
head dimensions are interchangeable, disrupting RoPE's position-dependent
structure; a per-channel quantization fix (KIVI-inspired) resolves this by
respecting that structure instead; and layer-selection's cost is
architecture-dependent in a way its own calibration signal does not
predict. The content-identity finding is not an artifact of testing only
uncompressed relay: the identical three-tier causal ordering replicates on
both a layer-selected/quantized configuration and a uniformly quantized
one, more decisively than on uncompressed relay — the claim holds under
the conditions a practitioner would actually deploy.

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
text between agents has been proposed to avoid redundant prefill
computation. KVCOMM (NeurIPS'25) introduced an anchor-based framework that
estimates and corrects KV-cache "offset drift" when reusing cache across
different prefix contexts, reporting over 70% cache reuse and up to 7.8×
prefill speedup on retrieval-augmented generation, math reasoning, and
coding tasks. LatentMAS (Zou et al., ICML'26) extends this to full
latent-space collaboration, exchanging autoregressively-generated latent
"thoughts" via KV-cache
working memory rather than relaying already-decoded text, reporting token
savings and accuracy gains across nine math/science/code benchmarks. We
differ from both in relaying real, decoded reasoning text through a fixed
sequential three-agent chain (Reasoner → Verifier → Finalizer) rather than
offset-corrected reuse or latent thought generation, and in evaluating on
multi-hop QA (HotpotQA), which neither tested on.

**Compressing the relayed cache.** Reducing what must be transmitted between
agents is a natural complement to relay itself. We adopt two families of
technique: layer-wise selection, which drops calibration-identified
low-importance transformer layers before transmission and reconstructs them
at the receiver via `zeros`, `nearest`, or `interpolate`, and per-head
quantization, which reduces numeric precision. A third reconstruction
strategy, Orthogonal Backfill ("When Less Latent Leads to Better Relay,"
arXiv 2604.13349), injects a low-rank residual of discarded content
orthogonal to what is retained rather than substituting or discarding it
outright; we considered but did not implement it (see Limitations for why).
KV-cache compression is also under active industrial development: KVTC
(Staniszewski and Łańcucki, NVIDIA, ICLR'26) applies media-compression-
style transform coding — PCA decorrelation, adaptive quantization, entropy
coding — for up to 20x compression, a third paradigm distinct from uniform
quantization (ours, KIVI) and rotation (TurboQuant/PolarQuant); NVIDIA's
kvpress library and TensorRT-LLM's reuse-aware cache confirm this is an
actively contested axis of real deployed systems, not a narrow academic
concern.

**Rotation-based quantization interacts badly with RoPE-encoded keys — a
finding, not just an implementation note.** We initially implemented a
rotation-based quantization scheme in the style of TurboQuant (Zandieh et
al., ICLR'26) and PolarQuant (Han et al., AISTATS'26) — two distinct papers
sharing only two authors, not one work under two names — applying a fixed
orthonormal Hadamard rotation to both keys and values before quantizing to
redistribute per-head outlier magnitude (covering PolarQuant's
rotate-then-quantize step, not TurboQuant's separate QJL bias-correction
term). On our real model this produced severely
out-of-distribution decoded output, not the graceful degradation a generic
rotation-based scheme predicts. We traced this to keys' rotary position
embeddings (RoPE): K is cached *after* RoPE is applied, which mixes channel
pairs by a position-dependent phase, and a second, RoPE-agnostic rotation on
top disrupts that structure rather than merely redistributing outlier
magnitude. An isolating ablation (rotating only V, leaving K unrotated)
reverted the failure to ordinary quantization-noise degradation, confirming
the interaction was with K specifically — consistent with an established
interaction in the KV-quantization literature (next paragraph), and reported
here as a diagnosed negative result, not an implementation detail.

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
it resolves the collapse — 0.0%/0.0% to 52.0%/64.1% F1 at n=100,
statistically indistinguishable from our strongest layer-selection config
(`D`; McNemar p=0.86) at higher compression (3.96x vs 3.82x, both vs.
uncompressed relay) with no calibration profile required — direct
empirical confirmation, on our
pipeline and task, of what this literature identifies as the correct axis
for addressing the problem, rather than the rotation-based axis we tried
first. Whether KIVI's other asymmetric half (per-token value quantization)
adds anything on top is tested in Results, Finding 4.

**Auditing whether KV reuse does what it claims.** A recent line of work
interrogates cross-agent KV/latent reuse critically rather than only
reporting end-task accuracy. "When KV Cache Reuse Fails in Multi-Agent
Systems" (Liang et al., 2026, arXiv preprint) shows reuse strategies
effective for generation agents can silently corrupt an LLM judge's
cross-candidate comparison even when end-task accuracy appears stable.
"When Does Latent Communication Pay? A Causal Audit of Relayed KV Caches in
Multi-Agent LLMs" (Cheng et al., 2026, arXiv preprint) proposes replacing
relayed KV with mismatched, zeroed, or moment-matched-random substitutes to
test whether accuracy is causally attributable to the specific transmitted
content rather than to the receiver simply having a non-empty cache — a
methodology we adopt directly for our own causal audit (Section 3). Cheng et
al. already sweep three model families and five checkpoints for this audit,
so we do not claim cross-architecture testing of the causal audit itself as
novel; our own sweep instead applies architecture-generalization testing to
a different axis — layer-selection's accuracy cost (Section 4, Finding 3) —
which neither Cheng et al. nor "Do Latent Channels Actually Communicate? A
Causal Audit of Latent Multi-Agent LLM Communication" (Zhang and Emu, 2026,
arXiv preprint), a concurrent causal-audit paper using a five-metric
decomposition (encoded-sender-information, receiver-sensitivity,
content-value, and cross-agent-value measurements) rather than our
zeroed/random/mismatched ladder, addresses. Notably, Cheng et al.'s own
"natural regime" (receiver capable of the task without relay) finds
near-null true-vs-mismatched effects on GSM8K and ARC-Challenge across most
Qwen3 checkpoints — the opposite of what we find on GSM8K and HotpotQA with
Qwen2.5-7B-Instruct (Section 4, Findings 6 and 10), where true-vs-mismatched
gaps are large and significant in every comparison. We do not yet know
whether this reflects their LatentMAS-style compressed latent-thought relay
versus our full/compressed literal KV relay, their single sender-receiver
hop versus our three-hop sequential chain, or a genuine task-dependent
effect — flagged here as an open discrepancy rather than resolved.

**Layer redundancy is not universal.** "No Free Swap: Protocol-Dependent
Layer Redundancy in Transformers" (Garcia, 2026, arXiv preprint) finds that
which transformer layers are functionally interchangeable across Qwen3-8B,
Llama-3.1-8B, and Mistral-7B-v0.1 depends on the model and evaluation
protocol rather than being a fixed architectural property. This is
consistent with our own finding (Section 4, Finding 3) that a near-identical
proportion of dropped layers costs substantially more accuracy on
Mistral-7B-Instruct-v0.3 than on Qwen2.5-7B-Instruct, and offers one
candidate mechanism for why.

**Three 2026 papers anticipate pieces of this paper's argument.**
"Rethinking Layer Redundancy: Calibration Matters More Than Search in LLM
Depth Pruning" (Kim et al., 2026) argues redundancy is a joint function of
model *and calibration objective*, and that "a universal layer ranking may
not exist" — raising
the question of whether our cross-architecture gap is really about the
model, or about our calibration ranking failing to transfer. We tested the
most direct proxy: whether the calibration signal's own confidence (the
separation between kept- and dropped-layer importance scores) predicts
which model is safe to compress. It does not, and fails in the *wrong*
direction — Mistral's tier separation (0.673) is larger, not smaller, than
Qwen's (0.491), despite Mistral being the more fragile model when that
ranking is acted upon. This is a direct empirical test of exactly the
concern that paper raises, and supports a stronger reading: not just that
redundancy is calibration-objective-dependent, but that the objective's
own confidence is not a reliable signal of when its ranking is safe to
trust — consistent with our broader claim that non-exchangeability is not
visible from a technique's internal signals.

AdaK reports that adaptive KV-budget estimation generalizes across Qwen3
(4B/8B) and Mistral-7B-Instruct-v0.2 — the same two architecture families
this paper tests, not a third (Llama is not among AdaK's evaluated models,
though a superficial reading of the abstract could suggest otherwise) —
which on its face looks like a counterexample to our architecture-
dependence claim. We do not believe it is: AdaK estimates budget
per-instance and per-model at inference time rather than committing to a
single calibration-time ranking applied uniformly thereafter, so it never
assumes a fixed, transferable ranking is safe to act on. Read this way,
AdaK's success is a positive instance of our principle, not a counterexample
— it works specifically because it avoids the fixed-ranking assumption our
own `D` configuration makes. This is our reading of the mechanism, not
something we verified by reimplementing AdaK ourselves; a direct
head-to-head comparison on our own pipeline is noted as a natural extension
in Limitations.

Finally, "The Pitfalls of KV Cache Compression" (ACL 2026) makes a general
version of the point this paper investigates concretely: that compression
methods optimize for aggregate metrics while implicitly equating token
retention with functional preservation. Our contribution is not the
abstract observation but a causally-grounded, cross-axis demonstration of
it within one system — the same failure pattern recurring across three
structurally distinct axes of KV-cache structure (depth, position,
content-identity) in a single pipeline, with the content-identity axis
causally validated via our audit rather than inferred from compression
metrics alone. We position this paper as the causal, cross-axis synthesis
these three papers each gesture toward from a different angle.

---

## 3. Method

**Pipeline.** We study a sequential three-agent pipeline — Reasoner,
Verifier, Finalizer — answering HotpotQA (Yang et al., 2018, distractor
configuration) questions, where each item bundles its own ten candidate
passages and requires no retrieval step; Section 4's GSM8K generalization
check (Cobbe et al., 2021) uses the same pipeline structure. Agents share
the same prompt structure and
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

**Models.** We evaluate on Qwen2.5-7B-Instruct (Qwen Team, 2024, bf16) as
our primary model, and — specifically to test whether findings generalize
across architecture, not just within one model family — Mistral-7B-
Instruct-v0.3 (Jiang et al., 2023, bf16), chosen for its architectural
differences (distinct layer count, attention head configuration, and
tokenizer) at comparable parameter scale. Both are decoder-only
transformers using grouped-query attention and RoPE.

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

**RoPE-aware quantization fix for `B_int4`.** `B_int4` (uniform 4-bit,
no layer selection) initially collapsed to 0.0% accuracy — diagnosed as a
quantization/RoPE interaction and confirmed via two ablations (see Related
Work for the full diagnostic story). Our fix (`B_int4_kivi`), inspired by
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
baseline in the interpretability literature: it pushes activations off the
training distribution, which can produce large behavioral changes for
reasons unrelated to whether the zeroed content carried meaningful
information. We treat the moment-matched-random and mismatched-example
conditions — which preserve realistic activation statistics or substitute
another real, in-distribution cache — as the more informative evidence, and
zero-ablation as a supplementary sanity floor; if zeroing alone diverges
sharply from the other two, we attribute that to the out-of-distribution
effect rather than content-dependence. A fully rigorous treatment of "no
difference" would also use a pre-registered equivalence margin rather than
absence of significance under our paired test; we did not have the budget
to pre-register and power such a test, and report significance/CI results
with that caveat.

## 4. Results

Eight findings follow. Two are the paper's headline results: Finding 5,
the causal audit establishing that relayed KV carries specific, real
content rather than merely being non-empty, and Finding 6, that a
calibration procedure's own confidence fails to predict — and actively
misdirects on — which architecture is safe to compress. The remaining
findings characterize specific published techniques against this backdrop.

**Table 1: Established results (Qwen2.5-7B-Instruct, HotpotQA, n=100).**

| Config | Accuracy | F1 | Latency | KV/hop |
|---|---|---|---|---|
| single_agent | 57.0% | 68.4% | 1.6s | — |
| text_agent | 54.0% | 68.3% | 7.2s | — |
| A | 56.0% | 69.1% | 8.6s | 144.35 MB |
| B_int8 | 57.0% | 70.5% | 8.3s | 72.16 MB (2.00x) |
| B_int4 (original) | 0.0% | 0.0% | 39.4s | 38.38 MB (4.00x) |
| **B_int4_kivi (fixed)** | **52.0%** | **64.1%** | 9.6s | 36.43 MB (3.96x vs `A`) |
| C | 43.0% | 57.8% | 10.0s | 104.16 MB |
| D | 50.0% | 61.9% | 10.2s | 37.82 MB (3.82x vs `A`) |
| E | 9.0% | 17.4% | 19.8s | 35.63 MB |
| E_int8 | 1.0% | 6.2% | 22.2s | 49.05 MB |

Ratios are compressed-vs-`A`'s 144.35 MB, except `B_int4`/`B_int8`, whose
self-referential ratio equals their vs.-`A` ratio exactly since neither
drops any layers. See Method for the real nibble-packing that makes 4-bit
configs genuinely, not just nominally, smaller than 8-bit ones.

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
(n=100), a near-identical proportion (8/28, 29%) shows a smaller,
not-yet-significant cost (p=0.34) — the cross-model claim does not rest on
that reaching significance, though: even Qwen's most generous uncertainty
bound (F1 CI upper +16.4 points) is under half of Mistral's confirmed
collapse. Consistent with prior work finding layer redundancy
architecture- and protocol-dependent, not a fixed property of a given
compression ratio.

**Finding 4 — `B_int4`'s fixed variant matches our strongest
layer-selection configuration's accuracy at genuinely higher compression
and lower complexity.** `B_int4_kivi` (52.0%/64.1%) vs `D` (50.0%/61.9%):
McNemar p=0.86, F1 delta +2.2 points [95% CI: -8.2, +12.8] —
statistically indistinguishable — while achieving 3.96x compression versus
`D`'s 3.82x (both vs. uncompressed `A`), and requiring no per-model
calibration profile. This number went through a real correction cycle we
report for transparency: this codebase's quantized tensors were initially
stored as one full `torch.uint8` per element regardless of bit-width — no
nibble-packing existed anywhere in the compressor, so a 4-bit value took
the same byte an 8-bit one did, and our first-reported ratios (silently
assuming packing that did not exist) were wrong in a way that happened to
be numerically optimistic. We caught this, then rather than only
correcting the byte *accounting*, implemented real two-values-per-byte
nibble packing (unit-tested: round-trip pack/unpack exactness, and
`compress`/`decompress` bit-identical output with and without packing —
packing is a storage-format change, not a numeric one) so the reported
ratio reflects genuine physical storage. The corrected, packing-backed
numbers land back at the originally-reported figures almost exactly,
because the original formula's *arithmetic* (0.5 bytes/element for 4-bit)
was always what real packing would produce — the bug was that nothing in
the compressor actually packed anything. `B_int4_kivi` also trends below
`A`/`B_int8` on accuracy (52% vs 56-57%), but this gap is not yet
statistically confirmed at n=100 (7-12 discordant examples, p=0.36-0.50);
we report this as an open question, not as cost-free the way Finding 1
establishes for `B_int8`. Also fixing the value side (`B_int4_kivi_full`,
per-token quantization) does not improve on this further (44.0% vs 50.0%
on a matched 50-example subset, p=0.51, trending the other way) and is
still worse on compression (3.75x vs `B_int4_kivi`'s 3.99x self-referential)
for a real, mechanical reason: per-token quantization stores a
scale/zero-point pair per sequence position, and sequence length (hundreds
to low thousands of tokens) vastly exceeds K's per-channel grouping (128
channels) — real overhead, no accuracy benefit, and packing does not
remove it. Together with an earlier null result for rotating values
instead, two independently-motivated value-side interventions both failed
— evidence the key/RoPE interaction was the entire mechanism behind
`B_int4`'s collapse, not one of several factors.

**Finding 5 — headline result: relayed KV demonstrably carries specific,
real content, not merely a non-empty cache.** We ran the causal audit (Section 3) on
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

Qualitative inspection adds a smaller, mechanistically interesting
observation: `A_audit_mismatched`'s incorrect answers are coherent,
grammatical, and plausible — real linguistic structure attached to the
wrong question — while `A_audit_zeroed` produces garbled but recognizable
English, and `A_audit_random` produces content qualitatively *more*
degraded (code-fragment tokens, mixed-language noise) despite matching the
real cache's per-tensor statistics. We speculate this reflects how
attention treats the two null conditions differently: a zero vector is a
weak key attention can largely down-weight, while realistic-magnitude
random noise resembles genuine signal and can actively misdirect attention
rather than being ignored. A secondary, exploratory observation, not a
claim we isolated further.

**Finding 5, extended — the same ordering holds under compression and on a
second task family.** Extending the audit to `D` (layer-selected +
quantized) and `B_int8` (uniformly quantized), n=50 each on HotpotQA, the
identical three-tier ordering replicates more decisively than on
uncompressed `A`: `D` 54.0%/60.2% vs. `D_audit_zeroed`/`D_audit_random`
≈0.0% vs. `D_audit_mismatched` 26.0%/32.2%; `B_int8` 54.0%/68.0% vs.
zeroed/random 0.0%/0.0% vs. mismatched 24.0%/34.4%. All ten pairwise
comparisons across both configurations are significant (real vs.
zeroed/random p<0.0001 for both; real vs. mismatched p=0.0013 `D` /
p=0.0003 `B_int8`, both tighter than `A`'s p=0.0127; mismatched vs.
zeroed/random p=0.0002-0.0005), and raw generations show the identical
failure signatures found on `A`. A single further confirmatory run on
GSM8K (Qwen, n=50, `A` and the full causal audit on `A`) shows the same
pattern still more decisively: `A` 90.0%, `A_audit_zeroed` 0.0%,
`A_audit_random` 2.0%, `A_audit_mismatched` 28.0%, every comparison
significant (p<0.0002); the same run's `D` matched `A`'s accuracy exactly
(McNemar p=1.0) — layer-selection was statistically free on this task,
unlike on HotpotQA. Together these ground the content-identity claim for
compressed, layer-selected relay and for a second task family, not only an
idealized uncompressed HotpotQA baseline — though the GSM8K check is one
run on one model, reported as suggestive, not a systematic cross-task
study (see Limitations).

**Finding 6 — headline result: a calibration signal's own confidence does
not predict downstream layer-selection safety, and fails in the wrong
direction.** We tested the most direct available proxy for whether our
cross-architecture gap (Finding 3) reflects the model or merely our
calibration procedure: the separation between calibration-importance scores
assigned to kept (tier-1/2) versus dropped (tier-3) layers, as a measure of
how confidently the calibration ranks layers. Qwen's mean separation is
0.491 (tier-1 0.737, tier-3 0.246); Mistral's is *larger*, 0.673 (tier-1
0.851, tier-3 0.178) — Mistral's calibration looks more decisive, not less,
yet Mistral is the model whose accuracy collapses harder when that ranking
is acted upon (Finding 3). This rules out a noisier or less-confident
calibration signal for Mistral as the explanation for Finding 3, and
supports a sharper one: the calibration objective can be equally or more
internally confident while being less reliable, undetectable from the
calibration output alone. This is a single comparison across two models,
not a validated general predictor (see Limitations), but a real, falsified
hypothesis test, not an assumed conclusion.

**Finding 7 — architecture-dependent layer-selection failures differ in
kind, not only in rate.** Beyond the accuracy gap already reported (Finding
3), `D`'s incorrect answers differ in character between models. Qwen's are
short (mean 18 characters, 2% exceed 150) and overwhelmingly close,
traceable near-misses. Mistral's are markedly longer (mean 109 characters,
15% exceed 150, maximum 1,454) and include a genuine long tail of severe
failures — fluent but entirely fabricated tangents, degenerate repetition
loops — absent from Qwen's failure set. We checked an obvious alternative
explanation — under-penalizing repetition for Mistral relative to its own
tuned defaults — against each model's published generation defaults, and
find the opposite: Qwen's own default *is* our `repetition_penalty=1.05`,
while Mistral's specifies none, so our setting applies *more* correction
for Mistral, not less. The long-tail pattern persists anyway, weakening a
decoding confound as the explanation and supporting depth-axis
non-exchangeability differing in *character*, not only magnitude, across
architectures.

**Finding 8 — donor-question content occasionally, but rarely, bleeds
through under mismatched substitution.** Unlike "When Latent Agents Lie"
(Brito and Baquero, 2026), which studies an adversarial agent deliberately
substituting hidden state in a fan-in topology, our substitution is a
non-adversarial intervention in a sequential chain: when the receiving
agent is wrong, does its answer reflect the donor's content, or ordinary
confusion on the real question? Manually reviewing 20 of 36 incorrect
`A_audit_mismatched` answers against their donor question
(`results/run_20260909_104129`), we find one unambiguous instance — a
Kansas university fight-song question produced "Ellie Goulding" in its
wrong answer, traceable only to a donor question about that singer — and
one weaker, ambiguous case; the remainder show ordinary failure on the
real question's own topic. A genuine but rare phenomenon, not the dominant
explanation for `A_audit_mismatched`'s accuracy loss — a manual review of
a subset, not an exhaustive classification (see Limitations).

---

## 5. Limitations

**Causal audit and bleed-through scope.** We ran the causal audit on `A`
(uncompressed relay), `D` (layer-selected + quantized), and `B_int8`
(uniformly quantized) — the three-tier ordering replicates across all
three, at n=50 each — but not on `C` or the `B_int4` family (including the
`B_int4_kivi` fix); we claim the mechanism confirmed for the three tested
configurations, not for every configuration in this paper, though extending
it is a natural next step given the consistency already observed across
three architecturally distinct conditions (no compression, compression
only, compression plus layer selection). The bleed-through analysis
(Finding 8) is narrower still: a manual review of 20 of 36 incorrect
`A_audit_mismatched` examples by one annotator, with no automated
classifier, inter-annotator agreement check, or statistical test on the
resulting rate. We report it as a qualitative, exploratory observation —
evidence that donor bleed-through can occur, illustrated by one unambiguous
example — not a validated estimate of how often it occurs; a rigorous
version would require an automated or multiply-annotated classification
over the full set of incorrect examples, which we did not have the budget
to complete. Topology is also fixed: every result in this paper is on a
sequential, non-adversarial chain. "When Latent Agents Lie" audits an
adversarial fan-in topology instead (Section 2), and LatentMAS collaborates
via layer-wise KV concatenation across a different multi-agent structure
than our hop-by-hop handoff; whether the same three-tier causal ordering
holds under a fan-in or concatenation-based topology is a natural, cited
extension we did not attempt, not one we found and characterized. Since
writing this, we built and mechanically validated (unit-tested on synthetic
tensors, not yet run on a real model) a prototype extending our relay
mechanism to exactly this fan-in setting: a static, depth-1 decomposition
of HotpotQA's ten passages across two child agents, whose independently-
computed KV caches are merged via a RoPE position-shift before an
aggregator attends over them, contrasted against a text-relay baseline that
mirrors Recursive Language Models' (Zhang, Kraska, and Khattab, arXiv
2512.24601) actual mechanism of discarding a sub-call's computed
representation and relaying only a token-limited text answer. We confirmed
this gap is real rather than assumed: the closest adjacent work we found,
"Recursive Models for Long-Horizon Reasoning" (arXiv 2603.02112), discusses
KV-cache-based return values only as an unimplemented assumption for a
theoretical speedup bound, and its real experiments discard a child call's
reasoning the same way RLM does. RecursiveMAS (arXiv 2604.25917) is the
closest published system that relays something other than text
recursively, but passes a single trained, adapter-compressed last-layer
hidden state through a sequential agent loop, not a full multi-layer KV
cache through a parallel fan-in decomposition, and requires fine-tuning
lightweight link modules rather than working training-free as our relay
mechanism throughout this paper does. Whether the fan-in merge we
mechanically validated produces coherent, let alone accurate, output on a
real model is unresolved at time of writing — reported here as a scoped,
in-progress extension, not a result.

**Single-process, single-GPU evaluation.** Every experiment runs within one
Python process on one RTX 4090: no `KVMessage` is ever serialized or
transmitted over a network. Compression-ratio and transmitted-byte
accounting reflects what a real distributed deployment would need to send
and is meaningful on that basis, but our latency numbers exclude network
transfer time, and smaller relayed caches are not observed to be faster in
this setup, since the model still executes every transformer layer
regardless of what its injected cache contains — claims here characterize
communication-cost tradeoffs and accuracy effects, not measured end-to-end
deployment speedups. This single-GPU constraint also bounded the sample
sizes and the number of model/config combinations we were able to test; a
systems-scale deployment study (multi-GPU, real network transfer,
throughput under concurrent load) was out of scope.

**Statistical power and replication.** We report paired significance tests
(exact McNemar) and bootstrap confidence intervals for every headline
comparison; several — INT8's near-zero cost, offset correction's
catastrophic failure, Mistral-7B's layer-selection collapse — are
significant at conventional thresholds even at our evaluation scale. Others
are not: at n=100, we cannot yet statistically distinguish Qwen2.5-7B's
layer-selection condition from its uncompressed baseline in isolation
(power ≈15%; roughly 3× the current sample size would be needed). Our
cross-architecture claim rests on the independently well-powered Mistral-7B
result, not the underpowered Qwen-only comparison. Relatedly,
`B_int4_kivi`'s comparison against `D` (statistically indistinguishable,
McNemar p=0.86) rests on a single n=100 run each, not a multi-seed or
replicated evaluation — we report the confidence interval alongside the
point estimate specifically so "matches `D`" is read as the current best
estimate under one run, not a claim strengthened by replication we have not
performed.

**Scope: two models, mostly one task, greedy decoding.** We test
Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (chosen for architectural
diversity at comparable scale); whether findings — particularly
layer-selection's architecture-dependence — extend to substantially larger
or smaller models, mixture-of-experts architectures, or models outside this
decoder-only, grouped-query-attention, RoPE lineage is untested (Candidate
model families for this extension are in progress at time of writing;
one methodological risk worth flagging in advance: "When Does Latent
Communication Pay?" reports at least one model pairing where the receiver
could not consume a relayed cache at all, i.e. cross-architecture relay
can fail at the precondition stage, not just on accuracy — a null result
under those conditions would itself be informative, not a bug). All of
our own architecture variation is *between* pipeline runs (the same model
family fills every agent role within a given run); whether a single
pipeline mixing architectures across agent roles — a Qwen Reasoner
handing its cache to a Mistral Verifier, for instance — is even viable is
untested and, to our knowledge, an open problem in this literature more
broadly: KVCOMM itself flags relay between "agents with identical
architectures but different weights" and agents with "different attention
formulations" as work it leaves for future exploration, and we inherit
that same gap rather than closing it. Nearly all results are on HotpotQA
(distractor configuration); we ran one
n=50 confirmatory check on GSM8K (Qwen, `A`/`D`/full causal audit on `A`)
and both headline findings replicated cleanly — `D` matched `A`'s accuracy
exactly (McNemar p=1.0, more decisively free than on HotpotQA) and the
causal audit's three-tier ordering held with every pairwise comparison
significant (p<0.0002), larger effect sizes than the original HotpotQA
run. This is one run on one additional task family, not a systematic
multi-task study — we did not extend it to `B_int8`/`D`-family causal
audits or Mistral, and report it as suggestive, not as evidence the claims
generalize broadly across tasks. This matters specifically for the
offset-correction negative result: KVCOMM was evaluated on MMLU, GSM8K, and
HumanEval, none of which is multi-hop QA, and whether the technique fares
differently on the more homogeneous task family it was originally
validated on remains open (our own GSM8K check did not re-test `E`). All
results additionally use greedy decoding (`do_sample=False`); we did not
evaluate under sampling or self-consistency, which could interact
differently with a degraded or compressed relayed context than with a full
one.

**Nibble-packing is unit-tested, not deployment-tested.** We implement
real two-values-per-byte packing for every bits==4 tensor (see Finding 4),
verified via round-trip pack/unpack exactness and bit-identical
`compress`/`decompress` output with and without packing on synthetic
tensors. This is standalone tensor-level testing, not an end-to-end
GPU rerun with the new packing path in the live pipeline; accuracy/F1 are
unaffected in principle (packing changes storage format, not values, and
the unit tests confirm this directly), but we have not re-run the full
n=100 HotpotQA evaluation with packing enabled to confirm no integration
issue exists between the compressor and the rest of the pipeline (cache
reconstruction, device placement, batch handling). The reported accuracy
numbers throughout this paper come from pre-packing runs; only the
compression-ratio figures reflect the packing-enabled compressor.

**Partial reproduction of adapted and considered techniques.** Our
rotation-based quantization covers PolarQuant's core rotate-then-quantize
mechanism but omits TurboQuant's QJL bias-correction term; our KIVI-inspired
fix (Section 3) omits the original method's group-wise windowing and fp16
residual buffer, implementing full-sequence per-channel quantization only —
the minimal change isolating the RoPE-interaction mechanism we diagnosed,
not a complete reproduction. We report both as inspired-by implementations
scoped to test a specific hypothesis, not certified reproductions, and
encourage readers wanting the original techniques' full guarantees to
consult the primary sources. We considered but did not implement a fourth
layer-reconstruction strategy, Orthogonal Backfill ("When Less Latent Leads
to Better Relay," arXiv 2604.13349): its published formula requires
attention weights this pipeline's `sdpa` decode path does not expose, and
given our own evidence that naive/simplified reproductions can actively
mislead (the rotation-based quantization finding above), we chose not to
ship an approximated version under time pressure — a scoping decision, not
a result.

**Checked against ARR's actual guidance (2026-09-09):** a dedicated
"Ethical considerations" section is optional, not required — recommended
only when a paper raises specific ethical concerns, and (if included)
titled exactly that so it can be correctly excluded from the page count.
The Responsible NLP Research checklist itself is a separate structured
form completed on the submission platform (referencing this Limitations
section by number where relevant), not additional paper prose. We do not
include a separate Ethical considerations section: this work uses two
long-established public benchmarks (HotpotQA, GSM8K), two openly-released
model checkpoints, no human subjects, no crowdsourcing, and no new data
collection — we are not aware of a concern specific to this paper beyond
what Limitations already covers (compute cost, single-GPU deployment
claims, and the scope caveats above).

---

## References

Full BibTeX entries, verified against primary sources (arXiv, ACL
Anthology, IJCAI/ICML/NeurIPS/ICLR/AISTATS proceedings, OpenReview) on
2026-09-09, are in `docs/references.bib` — pull that in wholesale for the
LaTeX submission rather than re-typing citations from this prose draft.
Two corrections surfaced during verification that are already reflected in
the prose above and are worth flagging explicitly so they don't regress in
a later edit: (1) TurboQuant and PolarQuant are two distinct papers with
different venues (ICLR'26 vs. AISTATS'26) and mostly-different author
lists, not one work under two names; (2) AdaK's cross-architecture claim is
tested on Qwen3 and Mistral only, not Llama — the earlier draft's "Qwen,
Mistral, and Llama" phrasing was incorrect and has been fixed. Six of the
fourteen cited works are arXiv preprints with no confirmed peer-reviewed
venue as of this verification pass (`kvreusefails2026`, `latentcommpay2026`,
`nofreeswap2026`, `rethinkredundancy2026`, `latentagentslie2026`,
`orthobackfill2026` in the .bib file) — cite them as preprints, not as
peer-reviewed venue papers, unless a later check finds they've since been
accepted somewhere.
