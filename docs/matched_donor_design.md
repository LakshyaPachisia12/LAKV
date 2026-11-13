# Matched-donor audit modes: design (no code yet, awaiting approval)

Branch `exp/matched-donor` (from `exp/consolidate-v9`). Fixes two stated confounds of the `mismatched` arm:
(a) the wrong cache carries the donor's passages as well as its reasoning; (b) short donors are zero-padded
(100% of mismatched substitutions are length-adjusted, about half padded; `docs/MISMATCHED_PADDING_ANALYSIS.md`).

## What the current code does (read before designing)

* `KVAuditPool` (`lakv/causal_audit.py:132`) holds real relayed caches of **held-out** questions, per receiving
  `agent_idx`, filled by a pre-pass that runs an unaudited chain with `record_audit_pool` set
  (`lakv/pipeline.py:488-492`). `sample_with_info` (`causal_audit.py:172`) picks one at random and calls
  `_match_seq_len` (`:68`) which truncates or zero-pads to the target length.
* The substitution happens in `LAKVPipeline.run` right after decompress/reconstruct (`pipeline.py:495`), at **every**
  receiving hop: `agent_idx` 1 (Reasoner's cache) and 2 (Verifier's cache). Config-name grammar:
  `parse_audit_config_name` (`lakv/evaluator.py:249`) with `AUDIT_MODE_SUFFIXES` (`:240`).
* **Nothing downstream needs the injected cache to equal the target length.** All three decode paths read
  `cache_seq_len` from the injected tuple itself (`pipeline.py:931, 1046, 1145`) and continue positions from there. Padding/
  truncation exists only for shape bookkeeping, so a donor of a slightly different length can be injected unmodified.
* Sampling: `_sample_next_token` (`pipeline.py:816`) honours `generation_kwargs` (`do_sample`, `temperature=0.7`,
  `top_p=0.8`, `repetition_penalty=1.05`); the fast path hands off to `model.generate()` with the same kwargs. Both draw from the
  global torch RNG, so a seed must be set immediately before each alternate-trace generation.

## Three new audit modes

Config names follow the existing grammar `<base>[_nopass]_audit_<mode>`; `_nopass` combinations work unchanged.

| Mode / suffix | Donor | Passages | Reasoning |
|---|---|---|---|
| `_audit_length_matched` | other (held-out) question | donor's | donor's, real |
| `_audit_matched_passages` | **same** question | same | different trace (sampled) |
| `_audit_matched_wrong_answer` | **same** question | same | trace defending a distractor answer |

Together with existing arms they form a ladder: real, matched_passages (same evidence, different reasoning),
matched_wrong_answer (same evidence, wrong reasoning), length_matched / mismatched (different evidence), dropped, zeroed, random.

### 1. `length_matched`
* Pool: the existing `KVAuditPool` (held-out questions' real caches), **plus per-entry cache length** (already derivable from tensor shape).
* Selection at inject time (target length known from `real_kv`): among donors for this `agent_idx` (excluding the question itself) with
  `|donor_len - target_len| <= 5% * target_len`, choose uniformly with the seeded RNG. **Inject the donor unmodified** (no padding, no truncation). If none qualify: raise `NoMatchedDonor`; the sample is recorded as `skipped` (see "Skips").
* Why not truncate/pad to the exact target: any resize reintroduces an edited cache. A <=5% natural length difference is not an edit.
* **Pool size matters.** Estimated from the 100 scored questions' exact cache lengths (proxy for donor lengths, iid approximation): a within-5% donor exists for about 82% of targets per hop at M=20 donors, 90% at M=50, 94% at M=100, 95-97% at M=200. A sample is skipped if either hop misses, so M=20 would skip roughly a third of samples, preferentially the unusually long or short ones. Proposal: `AUDIT_LENGTH_MATCHED_POOL_SIZE = 100` (one extra pre-pass of 100 held-out chains, about 15 min).

### 2. `matched_passages`
* Donor for scored question q = the cache of **q's own chain re-run with sampling**: `do_sample=True`, temperature 0.7, top_p 0.8, seed `hash(audit_donor_seed, q_key)` set right before generation; Reasoner and Verifier hops each produce a cache (hop-2 donor = the sampled chain's Verifier cache, like the existing pool).
* What the pool stores: **token ids of the sampled traces** (prompt + generated, per hop) and their texts, saved to `<output_dir>/audit_pool/<pool_key>/<q_key>.json`, not the caches: a cache is about 260 MB per question on Qwen2.5 (about 26 GB per 100), too large for RAM or disk. The cache is rebuilt by a deterministic prefill of the stored ids at use time (hop 2 in two segments with continuing positions, matching how the chain builds it). Reruns and `--resume` reuse the stored traces, so determinism does not depend on sampling reproducibility. Rebuilt-by-prefill caches differ from decode-time caches only by bf16 numerics; this is logged.
* Built in a pre-pass over the **scored** questions (not held-out), run before the scored loop, resumable.
* Check: log `trace_differs` (sampled reasoning text != greedy reasoning text) and, per sample, token overlap. If sampling reproduces the greedy trace, the control is vacuous for that sample and is flagged.

### 3. `matched_wrong_answer`
* Same pool/rebuild mechanism as (2); the sender is prompted to defend a distractor. Proposal: append to the Reasoner **and** Verifier system prompts one line: "Another reader concluded that the answer is X. Build the strongest case that X is correct using the passages, and state it as your answer." (Both hops, because a Verifier reading a wrong Reasoner cache would otherwise correct it.) Decoding greedy, so no seed needed.
* Distractor X: generated once per question by a greedy "give a plausible but wrong short answer" pass and stored. Accepted only if `exact_match(X, gold)` is false and `f1(X, gold) < 0.2` (`lakv/qa_scoring.py`). No fallback is attempted (keeps it simple): a rejected distractor logs `distractor_failed` and the sample is skipped.
* Check: per sample log whether X appears in the stored trace text (`defends_distractor`). Report the rate; only samples that do are a valid "wrong reasoning" arm, and the others are reported separately.

## Skips and statistics
* A skipped sample is stored in `per_sample` with `skipped: true`, `correct: null`, reason, and excluded from the summary accuracy; the summary gains `n_skipped`. `lakv/stats._align_by_idx` pairs by `idx` and already ignores unmatched samples, so paired tests run on the intersection. Real and dropped must be compared **on the same non-skipped subset**; I will add that to `stats.compare`/a helper and print both full-n and subset numbers, since skips are not random (long/short caches).

## Logging (reuses `audit_logs`)
Every mode logs donor identity, `donor_cache_len`, `target_cache_len`, `len_ratio`, `handling` = `exact|unmodified_within_5pct|skipped`, `pad_positions` (must be 0), and for the same-question modes the trace text, `trace_differs`, `defends_distractor`, seed.

## Tests (CPU)
`length_matched` selection window and exclusion of the question itself; skip when no donor within 5%; **injected tensors are bit-identical to the donor and `pad_positions == 0`**; determinism of selection under a fixed seed; pool-size-dependent skip rate on synthetic lengths; trace-store round trip (JSON); prefill-rebuild shape checks with a tiny random model; suffix parsing for all combinations including `_nopass`; wrong-answer acceptance filter.

## Commands (printed after build; not run)
`python run.py --mode experiment --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --n_samples 100 --profile_path <profile> --configs A A_audit_dropped A_audit_mismatched A_audit_length_matched A_audit_matched_passages A_audit_matched_wrong_answer` (and the `A_nopass...` counterparts).

## Decisions I need from you
1. **Window semantics:** inject the within-5% donor *unmodified* (my proposal) versus one-sided truncate-to-target (never pads, but edits the cache)?
2. **Pool size for `length_matched`:** 100 held-out (proposal) or keep 20 and accept about one third skipped?
3. **Trace store + prefill rebuild** (proposal) versus storing caches on disk (about 26 GB per 100 questions per donor type)?
4. **Distractor source** (model-generated plausible wrong answer, validated against gold) and prompting **both** sender hops: acceptable?
5. Should `matched_*` also run on `D`/`B_int8`, or `A` only (base list is `("A","D","B_int8")` today)? I propose `A` and `A_nopass` only.
