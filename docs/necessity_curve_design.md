# Relay-necessity curve: design (STEP 1: design only, no code)

**Status: design only. Nothing in this document is implemented, and nothing here has been run on a GPU.** Branch `exp/relay-necessity-curve`, cut from `exp/consolidate-v9`. All code that follows from it will be marked untested until you run it.

**Goal.** Plot real-minus-dropped accuracy against how much information the *receivers* lack, on the Qwen2.5-7B-Instruct HotpotQA 3-agent chain (n=100, first 100 validation items). The two existing endpoints are the anchors: passages visible (`results/run_20261009_113254`: real 56%, dropped 59%, diff -3 pp, McNemar p=0.68) and passages hidden (`results/run_20261011_122408`: real 52%, dropped 20%, diff +32 pp, p=4.4e-7). The curve fills in between.

## What exists today (read before designing)

* `PipelineConfig.receiver_sees_passages` (`lakv/pipeline.py:256`); in `run()` agents >= 1 get `_question_only(question)` when it is false (`:461`). `_question_only` (`:1265`) cuts the blob at the last `"\n\nQuestion: "` and returns `"Question: <q>"`: all-or-nothing. The suffix `_nopass` sets it (`parse_audit_config_name`, `lakv/evaluator.py:249`, returns a 4-tuple that three callers and `tests/test_passage_ablation_and_text_control.py` unpack/compare exactly).
* The dataset loaders flatten each item to `{question: "Context:\n[Title] ...\n\n[Title] ...\n\nQuestion: ...", answer}` (`run.py:129`). **Gold-passage identity is not kept** (neither `load_hotpotqa` nor `recursive_pipeline.load_hotpotqa_structured` keeps `supporting_facts`).
* `dropped` = the receiver gets no upstream cache at all (`_should_inject_kv`, `lakv/pipeline.py:409`), so in the dropped arm the final answer depends only on the Finalizer's own prompt.

Facts I checked on CPU against the real data (current HotpotQA `distractor` validation parquet, first 100 items):
* Reconstructing the blobs with the loader's format reproduces the baseline runs' `question` text for **100/100** idx; both baseline runs share identical idx order and question text.
* Every item has **exactly 2 unique gold titles** (from `supporting_facts`), all present in the context; no duplicate titles; no passage contains a blank line, and the blob re-splits into its passages on `"\n\n["` for 100/100. Gold passages sit near-uniformly in the 10 positions (mean position 4.71 vs 4.5).
* **One of the 100 items (idx 27) has only 2 passages**, not 10 (both gold, no distractors). "k of 10" must therefore be defined as a share of each item's own passage count.
* Mean receiver-prompt content is 1,452 tokens at full context (Qwen2.5 tokenizer, user content only).

## 1. Two hiding axes (config suffixes)

Grammar becomes `<base>[_nopass | _hide<PCT> | _hidegold<G>][_audit_<mode>]`, e.g. `A_hide50`, `A_hide50_audit_dropped`, `A_hidegold1_audit_dropped`. Only one hiding suffix per name. To keep the existing 4-tuple parser and its tests untouched, add a sibling `parse_config_name()` returning a small dataclass that includes the hide spec; `parse_audit_config_name` stays as is for old names (and `run_metadata.decoding_for_config` is updated to use the new one).

**a) Fractional axis** `_hide20 / _hide50 / _hide80 / _hide100` (= k of 10 hidden for k = 2, 5, 8, 10; `_hide0` is the plain config, no suffix). `_hide100` must equal `_nopass` exactly (both names kept, equality tested). Rule, all integer arithmetic, per question with N passages and G gold (G = 2):
* `u = crc32(f"{hide_seed}|{question_key}|u") % 100`;
* total hidden `t = (PCT*N + u) // 100`; gold hidden `g = (PCT*G + u) // 100`, clipped so `0 <= g <= G` and `t-g <= N-G`.
* Within each class (gold / distractor) passages are ordered by `crc32(f"{hide_seed}|{question_key}|{class}|{title}")`; the first `g` gold and first `t-g` distractors are hidden. The visible passages keep their original order and the original `[Title] text` format.

This is **stratified and unbiased**: for N=10, t is exactly 2/5/8/10; expected gold hidden is exactly `0.2*k` (the extra `u` gives 0 or 1 gold at 20% and 1 or 2 at 80% with the right probability, exactly 1 at 50%). Prototype on the real 100 items: mean gold hidden 0.43 / 1.00 / 1.57 / 2.00 (expected 0.4 / 1.0 / 1.6 / 2.0); gold share of the hidden passages 21.6% / 20.2% / 19.8% (population 20%). It is also **nested** (hide20 ⊂ hide50 ⊂ hide80 ⊂ hide100 for all 100 questions, verified), so levels are a clean dose scale and share their randomness.

**b) Gold axis** `_hidegold1 / _hidegold2`: hide 1 or 2 of the 2 gold passages (0 = plain config); **all distractors stay visible** (7.9 on average). Which gold passage is hidden at level 1 is the lower-`crc32` one, so gold1 ⊂ gold2. This is the cleaner "information the receiver lacks" scale: every question loses exactly the same amount of *needed* evidence, and prompt length changes by only 7% / 15%.

**Why these choices.** Passage-level (not sentence-level) hiding keeps the visible text byte-identical to the baseline format so the endpoints can be reproduced exactly; hiding supporting *sentences* would give a finer scale but changes passage text and cannot reproduce the endpoints. Five fractional levels are what n=100 can resolve (see power in section 6); nesting means finer levels can be added later at the same cost per level. Built-in contrast: at 50% every question has exactly one gold hidden, so `hide50` and `hidegold1` differ only by the 4 hidden distractors, which isolates what distractor-hiding does. **Better/extra proposal:** also plot against the dropped arm's own accuracy ("receiver-only accuracy"), which is the x-variable of your planned predictive rule.

## 2. Identical hidden set for real and dropped; deterministic; stable hash

The hidden set is a **pure function** of `(hide_seed, question blob, gold titles, axis, level)`; neither the arm nor the audit mode enters it. Randomness is `zlib.crc32` only (never built-in `hash()`, which is randomized per process; an existing test pattern checks `PYTHONHASHSEED` independence). `hide_seed` defaults to 0 (`--hide_seed`), is recorded in run metadata, and the hidden titles are logged per sample (`per_sample["hidden_titles"]`) so a post-hoc check can assert real and dropped hid exactly the same titles for every idx.

## 3. Who sees what

The sender/Reasoner (agent 0) **always sees all passages**, at every level and in both arms. The receivers are agent 1 (Verifier) and agent 2 (Finalizer).

| Agent | Role | Prompt content at level L | Real arm: injected cache | Dropped arm |
|---|---|---|---|---|
| 0 Reasoner | sender | all passages + question | none | none (same) |
| 1 Verifier | receiver of agent 0 | `view_L`: visible passages + question | Reasoner's cache: its prompt (all passages) + its reasoning | no cache |
| 2 Finalizer | receiver of agent 1 | `view_L` (same as agent 1) | Verifier's cache = Reasoner's cache + Verifier's prompt + reasoning | no cache |

`view_0` = the original blob (exact), `view_100` = `"Question: <q>"` = `_question_only` (exact; both verified on all 100 items), otherwise `"Context:\n" + visible passages joined by "\n\n" + "\n\nQuestion: <q>"`. Consequences: in the real arm the full passages are still present in the cache through the Reasoner's prefix at every level; in the dropped arm the Verifier still runs but cannot influence the answer, which comes from the Finalizer's prompt alone.

**Mechanism (design choice for your approval).** Add `gold_titles` to each dataset dict in `load_hotpotqa` (extra key; `question` text is unchanged), have the evaluator compute the per-sample receiver view, and give `LAKVPipeline.run` an optional `receiver_question=None` that agents >= 1 use instead of `question` / `_question_only(question)` (default behavior unchanged). Alternative: put a callable into the config; rejected because it needs a per-question gold map inside the config.

## 4. Levels 0 and 100% must reproduce the existing runs

* **CPU unit tests (all 100 questions, no model):** `view_0 == blob`; `view_100 == _question_only(blob)`; hidden sets nested, identical across arms, deterministic across processes and `PYTHONHASHSEED`, stratification rates, idx 27 edge case, suffix parsing for every name including `_nopass` equivalence. I already ran the prototype of these checks, so they are expected to pass.
* **GPU regression (reuses the idea of `scripts/check_regression_vs_baseline.py`)**, n=20 on the new code, compared with idx 0-19 of the baselines by `raw_answer` and `predicted`: `A` and `A_audit_dropped` vs `run_20261009_113254`; `A_nopass`, `A_nopass_audit_dropped`, **and `A_hide100`, `A_hide100_audit_dropped`** vs `run_20261011_122408`'s `A_nopass*` arms. Needs two small changes to that script: a `--baseline-config` option (new and baseline config names differ for `A_hide100`), and getting it onto this branch (it currently exists only on `exp/matched-donor`; port or cherry-pick `bc7ce31`). Also run the same two-commit fallback as there (rerun on `exp/consolidate-v9` if it fails) to separate environment drift from the new code.

## 5. Arms per level

Real and dropped (`A_hideX`, `A_hideX_audit_dropped`; likewise `A_hidegoldG`). Mismatched is optional later: it would need a donor pool built at the same hide level and adds a donor-content confound, so it is out of scope for the first pass.

## 6. Paired statistics and the figure

* Per level: accuracy and F1 of real and dropped, difference in pp, **exact McNemar** (`lakv.stats.mcnemar_test`) on the paired per-sample outcomes, and a **paired bootstrap 95% CI for the accuracy difference** (resampling questions; the existing `bootstrap_f1_diff_ci` is F1-only, so a small accuracy variant is needed). The F1 difference with its CI is reported alongside.
* Multiplicity: Holm correction over the 7 distinct level tests (0, 20, 50, 80, 100, gold1, gold2); raw and adjusted p both shown.
* Dose-response: because levels are nested and paired by question, estimate the **slope of (real-correct minus dropped-correct) on level** with a bootstrap over questions (stacked levels), per axis, rather than comparing levels pairwise.
* **Power at n=100:** discordant fractions in the endpoints are 0.23 (visible) and 0.42 (hidden), giving a paired SE of about 4.8 to 6.5 pp and an approximate 80%-power minimum detectable difference of 13 to 18 pp. Intermediate levels can therefore resolve only large effects; the curve's *shape* is the claim, not small per-level differences.
* **Figure** (one file, both axes): two columns, fractional (x = % hidden) and gold (x = 0/1/2 gold hidden). Top row: real and dropped accuracy vs level (shows floor/ceiling effects the difference hides). Bottom row: real-minus-dropped in pp with 95% CI error bars, reused endpoint runs drawn in a different marker, n and Holm-adjusted p annotated. A third overlay panel puts both series on a common x = realized mean gold passages hidden (0, 0.43, 1.0, 1.57, 2.0 and 0, 1, 2), plus an optional panel against dropped accuracy.

## 7. Risks and confounds

* **Prompt length changes with the level** (user content, mean tokens): full 1,452; hide20 1,166 (80%); hide50 737 (51%); hide80 315 (22%); hide100 23 (2%); gold1 1,346 (93%); gold2 1,235 (85%). Length is both the mechanism and a nuisance; the gold axis keeps it nearly constant (the reason it is the cleaner scale). The cache length does *not* shrink in the real arm (it is dominated by the Reasoner's full-context prefix).
* **Hiding distractors changes difficulty.** On the fractional axis the dropped arm may get easier or harder as distractors disappear; `hide50` vs `gold1` isolates this.
* **Prompt wording is unchanged by hiding.** The Verifier prompt says "check ... the context passages given" and the Finalizer "copied from the context" even when little or no context is shown (the existing `_nopass` behavior); intermediate levels inherit it.
* **The sender always sees everything**, so this measures a supply-side asymmetry only; a "sender also lacks passages" experiment is a different one.
* **Floor/ceiling:** dropped accuracy at high hiding is near 20%, real near 52 to 56%; the difference is bounded and can mask real-arm changes (hence the accuracy rows).
* Single hide seed: which passages are hidden is one draw. A second seed would estimate that variance (doubles the cost of the new configs).
* Nested levels are positively correlated; use the paired slope, not independent-level reasoning.
* Edge cases: idx 27 (2 passages, no distractors): gold2 and hide100 show it only the question; gold identity relies on `supporting_facts` titles (verified 2/2 for the first 100 only).
* **Merge risk:** `exp/matched-donor` moved the prompt-building block out of `run()` into `_agent_input_ids` and added audit suffixes; this feature edits the same region (`run()`, `parse_*`, evaluator). Decide the merge order before implementing.

## 8. GPU time (n=100, ~10 s per question as you specified)

| Group | Configs | New runs? | Question-runs | ~Time at 10 s |
|---|---|---|---|---|
| Level 0 | `A`, `A_audit_dropped` | **reuse** `run_20261009_113254` | 0 | 0 |
| Level 100% | `A_nopass`, `A_nopass_audit_dropped` | **reuse** `run_20261011_122408` | 0 | 0 |
| Fractional 20/50/80 | 3 levels x {real, dropped} | new (6 configs) | 600 | 100 min |
| Gold 1/2 | 2 levels x {real, dropped} | new (4 configs) | 400 | 67 min |
| Regression checks (n=20) | `A`, `A_audit_dropped`, `A_nopass`, `A_nopass_audit_dropped`, `A_hide100`, `A_hide100_audit_dropped` | new | 120 | 20 min |
| **Total** | | 10 configs x 100 + 6 x 20 | **1,120** | **about 3.1 h** |

Measured latencies in the endpoint runs: real arm 8.4 s (hidden run) but **22.4 s** (visible run, cause unknown, possibly GPU contention), dropped about 7.2 to 7.3 s; using them the 10 new configs take roughly 2.2 h (8.4 s) to 4.1 h (22.4 s). Plus model load and per-run overhead. Optional, off by default: the Reasoner hop is identical across all real-arm levels (it always sees everything, decoding is greedy), so it could be cached; not worth the risk to exactness at first. Raising n to 200 for intermediate levels would double the new configs and require rerunning the endpoints on idx 100-199 (4 more configs).

## 9. UNVERIFIED (check these)

1. **The original commands** for both baseline runs. Neither was recorded (no `run_metadata.json`); `_nopass` and `_audit_dropped` first appear in committed code in `f17c136` (2026-10-29), after both runs (2026-10-09 and 10-11), so both used uncommitted working trees. Model id, profile, `--greedy`, `--arch` are inferred, as in `docs/matched_donor_design.md` step 0.
2. **Library versions and hardware** for those runs (not recorded). The version pins exist only on `exp/matched-donor` (`requirements.txt`), not on this branch.
3. That the GPU box's HotpotQA download has the same revision as the parquet I checked (it matches the baseline questions 100/100; the box's `datasets` version is the thing to confirm).
4. Gold = `supporting_facts` titles holds beyond the first 100 items (checked only on these).
5. Token counts are user-content only (Qwen2.5 tokenizer); system prompt and chat-template tokens add a constant.
6. Whether intermediate hide levels trigger degenerate Verifier behavior given the unchanged prompt wording (needs reading raw `hop_texts` after the n=20 smoke).
7. Why `run_20261009_113254`'s real arm took 22.4 s per question.
8. That the exact-reproduction test passes on the GPU box (step 4 above); the CPU prototype only shows the prompt *strings* are identical.
9. All code described here is unwritten; the CPU prototype lived outside the repo and is not committed.

## Decisions needed from you

1. **Levels and names:** `_hide20/50/80/100` + `_hidegold1/2`, level 0 = plain config, `_hide100` kept as an alias of `_nopass`? Share-of-N rule for idx 27 (2 passages) acceptable?
2. **Interface:** add `gold_titles` to the dataset dicts and an optional `receiver_question` to `run()` (my proposal), or something else?
3. **Hide seed:** one (0), or two seeds (doubles new GPU cost) to measure the variance from which passages are hidden?
4. **Branch/merge order** with `exp/matched-donor` (shared edits in `run()`/evaluator), and porting `check_regression_vs_baseline.py` (+ `--baseline-config`) here.
5. **Statistics and figure:** Holm over 7 level tests plus bootstrap slope, and the two-column layout with the common-x overlay: OK? Include the dropped-accuracy x-axis panel?
6. **Power:** accept the 13 to 18 pp minimum detectable difference at n=100, or run intermediate levels at n=200?
7. **Mismatched arm** later, yes or no; and whether to cache the Reasoner hop across levels (off by default).
