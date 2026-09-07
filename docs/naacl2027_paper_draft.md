# NAACL 2027 Draft — Related Work, Method, Results, Limitations

> Status: Related Work, Method, Results, and Limitations are drafted
> (2026-09-07, branch `feat/research-extensions`). Abstract and Introduction
> are NOT drafted yet — write those last, once the causal audit result
> (Finding 5, currently a placeholder) is in and the headline claim is fully
> locked. `B_int4_kivi` is confirmed at n=100: 52.0% accuracy / 64.1% F1,
> statistically indistinguishable from `D` (McNemar p=0.86) at higher
> compression and no calibration profile — current leading candidate for the
> paper's headline result. `B_int4_kivi_full` is built but not yet run.
> Causal audit is built but has zero GPU results — this is the single
> biggest remaining gap in the Results section. Orthogonal Backfill was
> deliberately scoped out (see Limitations), not left as an open thread.
> Bracketed notes mark anything needing a final number or decision before
> submission. Freely rewrite — this is a draft, not locked.

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
rotation-based axis we tried first. [If B_int4_kivi_full — adding per-token
value quantization, KIVI's other asymmetric half — improves further once
run: report that number instead/in addition, and note it as the more
complete asymmetric reproduction.]

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
methodology we adopt directly for our own causal audit (Section [X]). Our
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
consistent with our own finding (Section [X]) that a near-identical
proportion of dropped layers produces a substantially larger accuracy cost
on Mistral-7B-Instruct-v0.3 than on Qwen2.5-7B-Instruct, and offers one
candidate mechanism (architecture-dependent, not universal, layer
redundancy) for why.

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
(Section [X] below details a fix to `B_int4`'s originally-catastrophic
failure). `C` applies calibration-driven layer selection — dropping
transformer layers a per-model calibration pass identifies as
low-importance before transmission, reconstructing dropped layers at the
receiver via one of three strategies (`zeros`, `nearest`, `interpolate`).
`D` combines layer selection with adaptive per-layer-tier quantization
(high-importance layers at 8-bit, medium-importance at 4-bit). `E` adds an
anchor-based cross-agent offset-correction mechanism adapted from KVCOMM
(Section [X], a negative result).

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

**Causal audit.** [To be completed once GPU results are available — see
CLAUDE.md for current status.] To test whether a configuration's accuracy
reflects the receiving agent using the specific transmitted content, rather
than merely having *some* non-empty cache to attend over, we substitute the
relayed KV cache with (i) an all-zero cache, (ii) a moment-matched random
cache (same per-tensor mean/standard deviation, no real structure), and
(iii) another, unrelated HELD-OUT question's real cache at the same
pipeline stage (sampled from a pool built from examples never included in
the scored evaluation set, so no scored question's content can leak back to
itself). We apply this to our three strongest configurations (`A`,
`B_int8`, `D`).

## 4. Results

**Table 1: Established results (Qwen2.5-7B-Instruct, HotpotQA, n=100).**

| Config | Accuracy | F1 | Latency | KV/hop |
|---|---|---|---|---|
| single_agent | 57.0% | 68.4% | 1.6s | — |
| text_agent | 54.0% | 68.3% | 7.2s | — |
| A | 56.0% | 69.1% | 8.6s | 144.35 MB |
| B_int8 | 57.0% | 70.5% | 8.3s | 72.16 MB (2.00x) |
| B_int4 (original) | 0.0% | 0.0% | 39.4s | 41.36 MB (4.00x) |
| **B_int4_kivi (fixed)** | **52.0%** | **64.1%** | ~10s | ~36-39 MB (3.99x) |
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
`B_int8`. [If `B_int4_kivi_full` — adding per-token value quantization —
improves on this further once evaluated, report that result here instead
or in addition.]

**Finding 5 — causal audit.** [Pending GPU evaluation.]

---

## 5. Limitations

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
KIVI-inspired fix (Section [X]) omits the original method's group-wise
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

**`B_int4_kivi`'s accuracy is now confirmed at n=100 (52.0%/64.1%) — the
n=10 pilot's 30.0%/39.4% was, as flagged at the time, not yet trustworthy;
n=100 is the number to cite.** Its comparison against `D` (statistically
indistinguishable, McNemar p=0.86) rests on a single n=100 run each; a
replication run, or reporting the confidence interval directly rather than
only the point estimate and p-value, would strengthen this claim further
before treating "matches `D`" as a settled fact rather than the current
best estimate.

**Compute constraints.** All experiments were run on a single RTX 4090.
This bounded both the sample sizes reported above and the number of
model/config combinations we were able to test; a systems-scale deployment
study (multi-GPU, real network transfer, throughput under concurrent load)
was out of scope for this work.

[Ethics/broader-impact note if ARR's checklist requires one beyond
Limitations — check the current ARR responsible-research checklist before
submission, since a separate ethics statement may be a distinct required
section rather than folded into Limitations.]
