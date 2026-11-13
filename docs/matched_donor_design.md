# Matched-donor audit modes: design and GPU checklist

**Status: implemented on `exp/matched-donor`, UNTESTED ON GPU.** 254 CPU tests pass (1 skipped: the GPU parity test), including an end-to-end run through the real `LAKVPipeline` on a tiny random model, but nothing here has run with a real model. Do not treat any behavior of the new modes as validated until checklist step 2 passes.

Fixes two confounds of the `mismatched` arm: (a) the wrong cache carries the donor's passages as well as its reasoning; (b) short donors are zero-padded (100% of mismatched substitutions are length-adjusted, about half padded; `docs/MISMATCHED_PADDING_ANALYSIS.md`).

## What the code relies on (verified by reading)

* Every decode path reads `cache_seq_len` from the injected tuple (`lakv/pipeline.py` `_generate`, `_generate_intermediate`, `_generate_intermediate_with_hidden`), so a donor of any length can be injected unmodified; padding/truncation existed only for shape bookkeeping.
* A real hop's cache holds the prompt plus the generated tokens **minus the final one** on the `generate()` path (HF never forwards its last token). The manual-loop path caches all non-EOS tokens. So traces store `cache_gen_len` next to the ids instead of assuming `len(gen_ids)`.
* `question_key` is an md5, process-stable. Sampling uses the global torch RNG, so each sampled hop is seeded immediately before generating.

## Modes (config suffixes; base `A` and `A_nopass` only)

| Suffix | Donor | Passages | Reasoning |
|---|---|---|---|
| `_audit_length_matched` | other held-out question's real cache | donor's | donor's, real |
| `_audit_matched_passages` | **same** question | same | different trace (sampled) |
| `_audit_matched_wrong_answer` | **same** question | same | trace that defended a distractor |
| `_audit_self_replay` | the question's **own** chain | same | same (validation only) |

Valid names are listed by `lakv.evaluator.valid_audit_config_names()`. Other bases (`D`, `B_int8`) raise a clear error. All donors are injected **unmodified**: no padding, no truncation.

### length_matched (disk-backed)
* Pool = `DiskKVAuditPool` (`lakv/causal_audit.py`): each held-out question's real hop caches are `torch.save`d to `<run>/audit_pool/<A|A_nopass>/length_matched/hop{1,2}/NNNN_<key>.pt`; RAM holds only a small JSON index. A complete pool of exactly the requested size is reused on resume; any other size is rebuilt.
* Selection: donors with `|donor_len - target_len| <= 5% * target_len` (inclusive), own question excluded, uniform choice with the seeded `audit_donor_seed` RNG, **exactly one donor file loaded** (straight to the target device). None eligible -> `NoMatchedDonor` -> sample recorded as skipped.
* Default pool 100 (`--length_matched_pool_size`): estimated from existing cache lengths (iid approximation) a within-5% donor exists for ~94% of targets per hop at 100 donors vs ~82% at 20; a sample is skipped if either hop misses.

### matched_passages / matched_wrong_answer (token ids only)
* **Stored per scored question** (`<run>/audit_pool/<pool_key>/<mode[_seedN]>/<question_key>.json`): generated token ids per hop, `cache_gen_len` per hop, seeds, and scalar checks (`hop0_differs_from_greedy`, `defends_distractor`, the validated distractor string). Never a KV cache, never prompt or instruction text.
* **Rebuild at use time** by deterministic prefill of the **STANDARD** prompt ids (`LAKVPipeline._agent_input_ids(question, hop)` with no suffix; the same helper `run()` uses) + stored ids, hop 2 as two segments with continuing positions. The "defend X" line exists only inside `generate_chain_ids(system_suffix=...)` during donor generation; `run()` and every rebuild call the helper without it, so it is in no cache and no scored-chain prompt (unit- and end-to-end-tested).
* `matched_passages`: do_sample with the pipeline's temperature 0.7 / top_p 0.8, seed `stable_seed(audit_donor_seed, mode, question_key, hop)` = `zlib.crc32` (never `hash()`), set right before each hop; the directory includes the seed so another `--audit_donor_seed` never reuses these traces. A greedy hop-0 trace is generated once to record whether sampling actually produced a different trace (`hop0_differs_from_greedy`).
* `matched_wrong_answer`: greedy. A distractor is proposed by the model ("plausible but incorrect short answer"), accepted only if not an exact match to gold and F1 < 0.2; otherwise the sample is skipped (`distractor_failed`). During generation both the Reasoner and Verifier system prompts carry the defend line; whether the trace actually contains the distractor is stored (`defends_distractor`) and must be reported.

### self_replay (validation)
Rebuilds the chain's **own** trace from its recorded prompt/generated ids and injects it; logs `rebuild_vs_real` (length equality, `max_abs_diff`, `min_cosine`, `max_rel_l2`) against the cache the chain actually produced. It should reproduce the real arm; it is how the rebuild machinery and the `cache_gen_len` rule are validated on GPU.

### Skips and statistics
Skipped samples are stored with `skipped: true`, `correct: null`, a reason and the audit log; they are excluded from the arm's accuracy denominator (`summary.n_skipped`, `n_scored`). `lakv.stats` pairs only non-skipped samples, and `compare()` now also returns `n_pairs`, `n_skipped_a/b` and both arms' accuracy on the paired subset. Compare real/dropped on that same subset (skips are not random: unusually long/short caches).

## Memory and disk (estimates)

Per held-out question, both relayed hops, from the exact mean cache lengths of the v9 runs (hop 1 / hop 2 positions x bytes per position):

| Model | positions (hop1 / hop2) | bytes/position | MB per question | 100 donors (disk) |
|---|---|---|---|---|
| Qwen2.5-7B | 1696 / 3427 | 57,344 | 97 + 197 = 294 | **29.4 GB** |
| Mistral-7B-v0.3 | 1816 / 3642 | 131,072 | 238 + 477 = 715 | 71.5 GB |
| Qwen3-8B | 1679 / 3373 | 147,456 | 248 + 497 = 745 | 74.5 GB |

* **RAM:** the in-RAM `KVAuditPool` used by `mismatched` holds 20 donors = ~5.9 GB (Qwen2.5); the same design at 100 donors would need **~29 GB RAM**, which is why `length_matched` is disk-backed. Disk-backed RAM is the JSON index plus one donor at a time (a hop-2 donor is ~0.2-0.3 GB, p90 4,374 positions = 251 MB), loaded directly onto the GPU (transient, in addition to the real cache).
* `A_nopass` keeps a separate pool (hop-2 caches are shorter without passages): budget another ~25-29 GB if you run both. Token-id traces are negligible (a few KB per question).
* Pre-pass time (estimates from existing latencies, not measured): length_matched pool ~100 chains x 8-10 s = 15 min; matched_passages traces ~100 x 7 s = 12 min; matched_wrong_answer ~100 x 5 s = 8 min; one extra prefill (<~1-2 s) per substitution.

## Known limitations (read before trusting results)
* **GPU-untested.** The installed transformers on the CPU dev box (4.44) cannot run the pipeline's "prime, then hand off to `generate()`" fast path (an IndexError in rotary embedding, reproduced on the unmodified `feat/research-extensions` tip), so the end-to-end CPU test forces the manual decode loop for hops >= 1. The `generate()`-handoff `cache_gen_len` rule for hops >= 1 is therefore verified only by GPU step 2. The GPU box ran transformers 5.16.1 (`run_metadata.json`); `requirements.txt` still says `>=4.44.2`, which the pipeline does not actually support.
* Rebuilt-by-prefill caches differ from decode-built caches by bf16 numerics; self-replay measures how much. Sampling is not guaranteed bit-reproducible across hardware, which is why the sampled ids are stored.
* `matched_wrong_answer` pairs standard-prompt K/V with reasoning tokens produced under a different (defend) prompt, by design, so the instruction never enters a cache. The Verifier's trace was generated while attending the generation-time Reasoner cache but is rebuilt on the standard-prompt Reasoner cache.
* Skipped samples bias the arms that skip (`length_matched`, `matched_wrong_answer`); read results on the paired subset.

## Ordered GPU checklist (nothing below has been run)

`PROFILE` = your existing HotpotQA calibration profile (e.g. `profiles/<run>/qwen_hotpotqa.json`; profiles are not in git; `run.py` requires one even for `A`). Free disk: >= 35 GB for the `A` runs, ~65 GB if you also run `A_nopass`. Check `python -c "import transformers; print(transformers.__version__)"` is 5.x first.

0. **Regression check against the paper run (UNTESTED, do this first).** Confirms the `pipeline.py` edits on this branch (prompt helper moved out of `run()`, write-only side channels, constructor kwarg) did not change the paper's behavior. It reruns config `A` on the first 20 examples and compares every `raw_answer` and parsed `predicted` with `results/run_20261009_113254`, the n=100 Qwen2.5 chain run behind the v9 paper's chain row (`docs/latex/table_no_relay.tex`, header comment):
   ```
   python run.py --mode experiment --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --n_samples 20 --profile_path $PROFILE --output_dir results/regress_check --configs A
   python scripts/check_regression_vs_baseline.py results/regress_check/run_*/experiment_results.json results/run_20261009_113254/experiment_results.json --config A
   ```
   Pass = exit status 0 (20/20 identical). Why this command reproduces the baseline's `A` arm: `--n_samples 20` takes `dataset[:20]` of the deterministic HotpotQA `validation` (distractor) split, which are idx 0-19 of the baseline; `A` has no layer selection or compression, so the profile is never read (`lakv/evaluator.py:576` attaches it only for layer selection or adaptive compression; any existing file satisfies the required flag); decoding is greedy by default (`generation_kwargs["do_sample"] = False`), so `--greedy` is immaterial; `--arch` defaults to `legacy`; the baseline's other arms (`A_audit_dropped`, `A_audit_mismatched`) do not affect `A`. Supporting evidence the gate is meaningful: an independent earlier run, `results/run_20261008_215649` (n=50), reproduces the baseline's first 50 `A` answers exactly (checked with this script on CPU). `hop_texts` differences are printed as a diagnostic but do not change the exit status.
   **If it fails,** do not conclude the edits are at fault yet: run the same two commands on branch `exp/consolidate-v9` (before the matched-donor edits). If that also differs, the cause is the environment (versions, GPU, driver), not this branch; if it passes there, the difference is in this branch's `pipeline.py` edits: stop.
   **UNVERIFIED (could not be determined from the repo; check these on the GPU box):**
   * *Exact original command.* Nothing recorded it: the run predates the `run_metadata.json` writer and `_run_metadata`, and no doc quotes it. The flags above are inferred from the result file (configs `A`, `A_audit_dropped`, `A_audit_mismatched`, n=100, 28 layers), `docs/PREDICTIONS_BEFORE_RUNS_2026-10-11.md:79` (donor seed 0 = this run) and `run.py` defaults.
   * *Model id.* Not stored in the file; `Qwen/Qwen2.5-7B-Instruct` is the `run.py` default, the layer count (28) matches it, and the paper/`CLAUDE.md` attribute the run to Qwen2.5-7B.
   * *Profile.* Unknown, and irrelevant for `A` (see above).
   * *`--greedy`, `--arch`, `--device`.* Unknown; defaults assumed (see above for why `--greedy` does not matter).
   * *Code version.* No commit hash was recorded. `_audit_dropped` first appears in committed code in `f17c136` (2026-10-29), after this run (2026-10-09), and the last commit before that touching the pipeline was `ab9a49d` (2026-09-22), so the run used an uncommitted working tree. The pipeline logic for `A` is presumed unchanged since; step 0 is exactly the test of that presumption.
   * *Library versions and hardware.* Not recorded for this run. Only the 2026-09-08 smoke run recorded `transformers 5.16.1`, `torch 2.11.0+cu128`, `datasets 5.0.1`; `CLAUDE.md` says the GPU is a single RTX 4090. Confirm the GPU box matches the pins in `requirements.txt` and the same GPU/driver, or bf16 numerics may differ.
1. **pytest** (stop on any failure; the parity test runs here instead of skipping):
   `python -m pytest tests -q`
   CPU result on the dev box: 254 passed, 1 skipped.
2. **self-replay** (validates rebuild + `cache_gen_len`), n=20:
   ```
   python run.py --mode experiment --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --n_samples 20 --profile_path $PROFILE --output_dir results/matched_selfreplay --configs A A_audit_self_replay
   python scripts/check_matched_donors.py results/matched_selfreplay/run_*/experiment_results.json
   ```
   **Gate (decides pass/fail; the script exits 1 on failure):** at least **19 of the 20 final answers agree** between `A` and `A_audit_self_replay` (the parsed `predicted` answers are equal after answer normalization), **and** `not_comparable == 0`. **Stop if `not_comparable > 0`:** the rebuilt cache length differs from the real cache, so the `cache_gen_len` rule is wrong for the `generate()` path. The script also fails the gate if fewer or more than 20 samples were compared (`--expect_pairs`, `--min_agree` override the defaults) or if rebuild logs are missing.
   **Diagnostics only (printed, never gate):** cosine similarity, `max_abs_diff` and relative L2 of rebuilt vs real caches per hop (`min_cosine_worst` etc.), raw-text identity (`identical_answer_frac`) and the McNemar line. bf16 prefill vs incremental decode is not expected to be bit-identical and there is no validated numeric threshold, so a low cosine with agreeing answers is a note to read, not a failure; read the numbers anyway, since a very low cosine with agreeing answers would mean the cache barely matters to this model.
3. **n=10 smoke** of every new arm (pool 30 so it is quick; many `length_matched` skips are expected at this size):
   ```
   python run.py --mode experiment --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --n_samples 10 --length_matched_pool_size 30 --profile_path $PROFILE --output_dir results/matched_smoke --configs A A_audit_dropped A_audit_mismatched A_audit_length_matched A_audit_matched_passages A_audit_matched_wrong_answer A_audit_self_replay
   python scripts/check_matched_donors.py results/matched_smoke/run_*/experiment_results.json --expect_pairs 10 --min_agree 9
   ```
   The smoke run includes `A_audit_self_replay` at n=10, so its gate is rescaled with `--expect_pairs 10 --min_agree 9` (the step-2 rule of at most one disagreement). Check: exit status 0 (self-replay gate passes, no padding, `length_matched` ratios within 1 +- 0.05); `hop0_differs_from_greedy_rate` is not ~0 (else sampling is vacuous: stop); `defends_distractor_rate` is high (else the wrong-answer arm is not wrong); skip counts sensible. Read a few traces:
   ```
   python - <<'EOF'
   import json, glob; from transformers import AutoTokenizer
   t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
   for f in sorted(glob.glob("results/matched_smoke/run_*/audit_pool/A/matched_wrong_answer/*.json"))[:3]:
       r = json.load(open(f)); print(r.get("distractor"), "|", t.decode(r["gen_ids"][0])[:400], "\n")
   EOF
   ```
4. **n=100** (a 100-donor pool is built fresh, because the 30-donor smoke pool does not match the requested size; interrupted runs resume with `--resume_dir results/matched_v1/run_<ts>`):
   ```
   python run.py --mode experiment --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --n_samples 100 --length_matched_pool_size 100 --profile_path $PROFILE --output_dir results/matched_v1 --configs A A_audit_dropped A_audit_mismatched A_audit_length_matched A_audit_matched_passages A_audit_matched_wrong_answer
   python scripts/check_matched_donors.py results/matched_v1/run_*/experiment_results.json
   ```
   and the passages-hidden counterpart (own output dir, own pools):
   ```
   python run.py --mode experiment --model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --n_samples 100 --length_matched_pool_size 100 --profile_path $PROFILE --output_dir results/matched_v1_nopass --configs A_nopass A_nopass_audit_dropped A_nopass_audit_mismatched A_nopass_audit_length_matched A_nopass_audit_matched_passages A_nopass_audit_matched_wrong_answer
   ```
   Then paired tests on the non-skipped subset, e.g. `python -m lakv.stats results/matched_v1/run_<ts>/experiment_results.json A_audit_dropped A_audit_length_matched` (see `n_pairs`, `n_skipped_*`, `accuracy_*_on_pairs`).
