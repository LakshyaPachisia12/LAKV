# NAACL 2027 Draft — Related Work + Limitations

> Status: first-pass draft of two sections, written 2026-09-07 on branch
> `feat/research-extensions`. Everything else (Abstract, Introduction,
> Method, Results) is not drafted yet — these two sections don't depend on
> results still pending (causal audit, TurboQuant GPU runs), so they're safe
> to write now. Bracketed notes mark anything that needs a final number or a
> decision before submission. Freely rewrite — this is a starting point, not
> a locked draft.

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
For the latter, we additionally implement a rotation-based quantization
scheme in the style of TurboQuant/PolarQuant (Zandieh et al., ICLR'26),
which applies a fixed orthonormal Hadamard rotation before quantizing to
redistribute per-head outlier magnitude — our implementation covers the
PolarQuant rotation-and-quantize step but not TurboQuant's additional QJL
bias-correction term [confirm this framing is still accurate if QJL gets
added later]. [If Orthogonal Backfill is implemented: cite "When Less Latent
Leads to Better Relay" (arXiv 2604.13349) here as a third reconstruction
strategy that injects a low-rank residual of discarded content orthogonal to
what is retained, rather than substituting or discarding it outright.]

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
of PolarQuant but omits TurboQuant's QJL bias-correction term [update once
finalized]. [If applicable: our Orthogonal Backfill implementation
approximates the source paper's attention-weighted residual averaging with
uniform averaging, since faithful reproduction would require switching this
pipeline's decode path from `sdpa` to `eager` attention specifically to
expose attention weights, at a latency cost we did not have time to fully
characterize before submission.] We report these as what they are —
inspired-by implementations, not certified reproductions — and encourage
readers who want the original techniques' full guarantees to consult the
primary sources directly.

**Compute constraints.** All experiments were run on a single RTX 4090.
This bounded both the sample sizes reported above and the number of
model/config combinations we were able to test; a systems-scale deployment
study (multi-GPU, real network transfer, throughput under concurrent load)
was out of scope for this work.

[Ethics/broader-impact note if ARR's checklist requires one beyond
Limitations — check the current ARR responsible-research checklist before
submission, since a separate ethics statement may be a distinct required
section rather than folded into Limitations.]
