# GPU runbook: matched-donor + relay-necessity curve (one machine, three sessions)

**Status: UNTESTED ON GPU.** Compiled from `docs/matched_donor_design.md` (branch `exp/matched-donor`, tip `885f614`) and `docs/necessity_curve_design.md` (branch `exp/relay-necessity-curve`, tip `4fdf64d`), plus checks I ran on CPU. Nothing here has run with a real model. Every number marked *estimate* is derived (basis stated) and unmeasured.

The two branches are **independent** (neither contains the other's code); this runbook does **not** merge them. The branches are on GitHub (`origin`); section 1.1 clones them onto the GPU machine.

## 0. Read first

**Legend.** *Pass* = the exact condition to continue. *STOP* = do not run later steps that depend on this one (dependency map below), keep the logs, and send them back (section 5). Commands are bash; run each from the worktree named in its card (`cd $MD` or `cd $NC`). Run GPU jobs one at a time (a 4090's 24 GB holds one 7B model plus caches; nothing here needs two at once).

| If this fails | It blocks |
|---|---|
| S1.1 matched-donor pytest | every matched-donor step (S1.2, S1.3, S2.1, 3.1, 3.3, 3.6) |
| S1.2 matched-donor regression | every matched-donor scored run, **and** the reuse of the baseline arms for paired tests |
| S1.3 self-replay | the `matched_passages` / `matched_wrong_answer` arms (3.3, the two same-question arms in 3.6, and those arms in S2.1); **not** `length_matched` (it injects real, un-rebuilt caches) |
| S1.4 / S1.5 curve pytest / regression | every curve step (S2.2, 3.2, 3.4, 3.5) |

**Where this runbook goes beyond the two docs (veto any of these):**
1. *Matched-donor regression is wider than the doc's:* the doc checks `A` only; I check `A`, `A_audit_dropped`, `A_nopass`, `A_nopass_audit_dropped` (n=20) so that the existing baseline arms can be **reused** instead of rerun in Session 3. If you skip the two `_nopass` configs, step 3.6 must also rerun `A_nopass`, `A_nopass_audit_dropped`, `A_nopass_audit_mismatched` (about +50 min).
2. *Session 3 runs only the NEW arms* in one folder per stage and pairs them with the baselines from `run_20261009_113254` / `run_20261011_122408` (idx-aligned) using `paired.py` below. The matched-donor doc's n=100 commands put six configs per folder.
3. *Pools go to `$POOLS`* (`--donor_pool_dir`), a scratch location, not inside `results/`, so about 30 GB of caches never end up in the folders you send back.
4. *pytest runs before the regression* (your requested order; both docs list the regression first).
5. The thresholds marked **(mine)** below (skip rate, empty-answer rate) are not from either doc.

### Timeline at a glance (minutes are *estimates*: 10 s per question as in both docs, plus about 3 min of model load per `run.py`, unmeasured)

| Stage | Branch | What | Est. min | Disk |
|---|---|---|---|---|
| Setup | both | clone, worktrees, env check, dataset check | 10 | 0.5 GB (+15 GB if the HF model cache is empty) |
| S1.1 to S1.3 | MD | pytest, regression n=20, self-replay n=20 | 5 + 16 + 11 | < 0.1 GB |
| S1.4 to S1.5 | NC | pytest, regression n=20 | 5 + 23 | < 0.1 GB |
| **Session 1** | | | **about 70** | |
| S2.1 | MD | n=10 smoke, 7 configs | 27 | 8.8 GB (30-donor pool) |
| S2.2 | NC | n=10 smoke, 10 configs | 20 | < 0.1 GB |
| **Session 2** | | (+ about 10 min reading output) | **about 57** | |
| 3.1 | MD | `A_audit_length_matched` | 35 | **29.4 GB** pool |
| 3.2 | NC | gold axis (4 configs) | 70 | < 0.2 GB |
| 3.3 | MD | `matched_passages` + `matched_wrong_answer` | 66 | < 0.1 GB |
| 3.4 | NC | fractional axis (6 configs) | 103 | < 0.2 GB |
| 3.5 | NC | seed-1 pair (2 configs) | 36 | < 0.1 GB |
| 3.6 | MD | all `A_nopass` donor arms (3 configs) | 95 | **about 25 to 29 GB** pool |
| **Session 3** | | | **about 405 (6.75 h)** | peak about 60 GB of pools |

Session 3 range: about 5.5 h if the real arm runs at 8.4 s per question, about 10 h if it runs at the 22.4 s seen once in `run_20261009_113254` (cause unknown). Free disk recommended: **90 GB** on the volume holding `$POOLS` (model cache 15 GB + both pools 60 GB + margin); with 50 GB, delete `$POOLS/lm_A` after step 3.1's results are saved (section 6).

## 1. One-time setup (about 10 min)

### 1.1 Get the branches onto the GPU machine (clone from GitHub)
```
export LAKV_BASE=$HOME/lakv-base MD=$HOME/lakv-md NC=$HOME/lakv-nc
git clone https://github.com/LakshyaPachisia12/LAKV.git $LAKV_BASE
cd $LAKV_BASE
git fetch origin                                   # also how to pick up later commits to these branches
git checkout exp/consolidate-v9                    # a plain clone checks out `main` (an old, README-only commit); the control tree must be on exp/consolidate-v9
git worktree add -b exp/matched-donor $MD origin/exp/matched-donor
git worktree add -b exp/relay-necessity-curve $NC origin/exp/relay-necessity-curve
```
`$LAKV_BASE` stays on `exp/consolidate-v9` (no new code at all): it is the control for regression fallbacks. The tracked baseline results (`results/run_20261009_113254`, `results/run_20261011_122408`) and `profiles/qwen_gsm8k.json` are present in all three trees (tested: a fresh clone from GitHub took about 30 s, its `.git` was about 41 MB, and all three trees had both baseline runs and the profile). Record the heads: `git -C $MD rev-parse --short HEAD` (expected `885f614`), `git -C $NC rev-parse --short HEAD` (expected `4fdf64d`); if the branches have moved on origin since this runbook was written, record the actual hashes instead.

### 1.2 Shell setup (paste at the start of EVERY session) and three helper scripts (create once)
```
export LAKV_BASE=$HOME/lakv-base MD=$HOME/lakv-md NC=$HOME/lakv-nc
export POOLS=/PATH/WITH/90GB/FREE/lakv_pools       # NOT inside the repo
export OUT=$HOME/lakv-sendback PYTHONUNBUFFERED=1
export PROFILE=profiles/qwen_gsm8k.json            # unused by config A (run.py only requires the flag); use a HotpotQA profile if you have one
export COMMON="--model Qwen/Qwen2.5-7B-Instruct --dataset hotpotqa --profile_path $PROFILE"
export B0=results/run_20261009_113254/experiment_results.json    # level-0 / visible-passages baseline (n=100)
export B1=results/run_20261011_122408/experiment_results.json    # passages-hidden (_nopass) baseline (n=100)
mkdir -p $POOLS $OUT/s1 $OUT/s2 $OUT/s3 $OUT/logs
latest() { ls -dt "$1"/run_* | head -1; }          # newest run folder under an output dir
```
```
cat > $OUT/pincheck.py <<'EOF'
import re, sys, importlib.metadata as md
bad = 0
for line in open("requirements.txt", encoding="utf-8"):
    m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line.strip())
    if not m: continue
    name, want = m.groups()
    try: have = md.version(name)
    except md.PackageNotFoundError: have = None
    ok = have is not None and have.split("+")[0] == want
    bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}: pinned {want}, installed {have}")
import torch
print("torch build:", torch.__version__, "| CUDA runtime:", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0), "| memory GB:", round(torch.cuda.get_device_properties(0).total_memory/2**30, 1))
if "cu128" not in torch.__version__: print("NOTE: the recorded runs used the +cu128 build; this is", torch.__version__)
sys.exit(1 if bad or not torch.cuda.is_available() else 0)
EOF
cat > $OUT/summ.py <<'EOF'
import glob, json, sys
for f in sorted(glob.glob(sys.argv[1])):
    d = json.load(open(f, encoding="utf-8")); print(f)
    for cfg, v in d.items():
        if cfg.startswith("_") or "per_sample" not in v: continue
        s, ps = v["summary"], v["per_sample"]
        oom = sum(1 for x in ps if x.get("oom_skipped")); sk = sum(1 for x in ps if x.get("skipped")); f1 = s.get("mean_f1")
        print(f"  {cfg:42s} n={len(ps):4d} acc={100*s['accuracy']:5.1f}% f1={(100*f1 if f1 is not None else float('nan')):5.1f} "
              f"skipped={sk:3d} oom_skipped={oom:2d} latency={s['mean_latency_seconds']:6.1f}s status={s.get('status')}")
EOF
cat > $OUT/paired.py <<'EOF'
"""python paired.py "<new_results_glob>" <new_cfg> <baseline_json> <baseline_cfg> [<baseline_cfg> ...]   (run from a tree that has lakv/)"""
import glob, json, sys
sys.path.insert(0, ".")
from lakv.stats import mcnemar_test
new_glob, new_cfg, base_file, *base_cfgs = sys.argv[1:]
new = json.load(open(sorted(glob.glob(new_glob))[-1], encoding="utf-8"))[new_cfg]["per_sample"]
base = json.load(open(base_file, encoding="utf-8"))
n_skip = sum(1 for s in new if s.get("skipped"))
print(f"{new_cfg}: {len(new)} samples, {n_skip} skipped ({100*n_skip/max(len(new),1):.0f}%)")
for bc in base_cfgs:
    b = {s["idx"]: s for s in base[bc]["per_sample"]}
    pairs = [(b[s["idx"]], s) for s in new if s["idx"] in b and not s.get("skipped")]
    pa, pb = [x for x, _ in pairs], [y for _, y in pairs]
    m = mcnemar_test(pa, pb)
    acc = lambda ps: 100 * sum(bool(x["correct"]) for x in ps) / max(len(ps), 1)
    print(f"  {bc:28s} vs {new_cfg}: n_pairs={len(pairs):3d}  acc {acc(pa):5.1f}% vs {acc(pb):5.1f}%  "
          f"diff {acc(pb)-acc(pa):+6.1f} pp  McNemar p={m.p_value:.3g}  (baseline-only {m.n_a_only}, new-only {m.n_b_only})")
EOF
```
All three were run on CPU against the tracked result files (`pincheck.py` correctly FAILs on the dev box, which has transformers 4.44.2 / torch 2.4.1+cpu). Always `cd` into the worktree a card names: `lakv` is imported from the current directory, and a stray editable install would silently use the wrong tree, so check once: `for t in $MD $NC; do (cd $t && python -c "import lakv, os; print(os.path.dirname(os.path.abspath(lakv.__file__)))"); done` must print each tree's own path.

### 1.3 Environment check (about 5 min)
```
cd $MD
{ echo "== date"; date; echo "== nvidia-smi"; nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
  echo "== python"; python -V; echo "== pins vs installed ($MD/requirements.txt)"; python $OUT/pincheck.py; echo "pincheck exit=$?"
  echo "== disk"; df -h . $POOLS; echo "== heads"; git -C $MD rev-parse --short HEAD; git -C $NC rev-parse --short HEAD; git -C $LAKV_BASE rev-parse --short HEAD
  echo "== status"; git -C $MD status --short; git -C $NC status --short; } 2>&1 | tee $OUT/s1/env_report.txt
pip freeze > $OUT/s1/pip_freeze.txt
cmp $MD/requirements.txt $NC/requirements.txt && echo "requirements.txt identical in both trees"
```
**Pass:** `pincheck exit=0` (torch 2.11.0, transformers 5.16.1, datasets 5.0.1 all `OK`; CUDA available; GPU memory about 24 GB); free disk on `$POOLS` volume >= 90 GB (or >= 50 GB and you accept deleting `lm_A` early); `git status` clean in both trees. **STOP** on any pin FAIL: with transformers 4.x the pipeline's `generate()` handoff fails (IndexError), so nothing else is meaningful. The recorded runs used the `+cu128` torch build; the driver must support CUDA 12.8 (the minimum driver version is not stated anywhere in the repo: **UNVERIFIED, check NVIDIA's table**).

### 1.4 Dataset identity check (about 2 min; once, from the curve tree)
```
cd $NC
python - <<'EOF' 2>&1 | tee $OUT/s1/dataset_check.txt
import json, run
data = run.load_hotpotqa("validation", 200)
for name, cfg in (("results/run_20261009_113254/experiment_results.json", "A"), ("results/run_20261011_122408/experiment_results.json", "A_nopass")):
    base = json.load(open(name, encoding="utf-8"))[cfg]["per_sample"]
    print(name, "first 100 questions identical:", all(data[i]["question"] == base[i]["question"] for i in range(100)))
print("gold titles per item (first 100):", sorted({len(x["gold_titles"]) for x in data[:100]}))
p = run.DATASET_PROVENANCE
print(json.dumps({k: p.get(k) for k in ("datasets_version", "num_rows_split", "fingerprint", "cache_revision_dirs", "questions_answers_sha256")}, indent=1))
EOF
```
**Pass:** both "identical: True"; gold titles `[2]`; `num_rows_split` 7405; **`questions_answers_sha256` = `001d377ed9af34522052ee83de7268c2f09a30287a338c382692ed710246286e`** (what the same call printed on the dev box, datasets 4.4.1; the content digest is the dependable identity, the hub revision there was `1908d6afbbead072334abe2965f91bd2709910ab` and may legitimately differ). **STOP** if a question differs: the data is not what produced the baselines.

## 2. Session 1: environment, correctness gates (about 70 min)

### S1.1 Matched-donor pytest (`$MD`, about 5 min, disk 0)
```
cd $MD && python -m pytest tests -q 2>&1 | tee $OUT/s1/md_pytest.txt
```
**Pass:** zero failures. CPU dev box: `276 passed, 1 skipped` (the doc's "254" is stale). On the GPU box `tests/test_cache_relay_parity.py` (native vs relay) runs instead of skipping, so expect `277 passed`. **STOP** on any failure; a parity failure means the cache relay differs from native generation on the real model, the most important gate in this runbook.

### S1.2 Matched-donor regression, n=20 (`$MD`, about 16 min, disk < 0.1 GB)
Confirms the `pipeline.py` edits (prompt helper moved out of `run()`, write-only side channels) did not change the paper's behavior.
```
cd $MD
python run.py --mode experiment $COMMON --n_samples 20 --output_dir results/regress_md --configs A A_audit_dropped A_nopass A_nopass_audit_dropped 2>&1 | tee $OUT/logs/md_regress.log
R=$(latest results/regress_md)/experiment_results.json
chk() { python scripts/check_regression_vs_baseline.py "$@" | tee -a $OUT/s1/md_regress_checks.txt; echo "exit=${PIPESTATUS[0]}" | tee -a $OUT/s1/md_regress_checks.txt; }
chk $R $B0 --config A;                chk $R $B0 --config A_audit_dropped
chk $R $B1 --config A_nopass;         chk $R $B1 --config A_nopass_audit_dropped
```
**Pass:** all four `exit=0` (20/20 identical `raw_answer` and `predicted`; `hop_texts` differences are a printed diagnostic only). An independent earlier run (`run_20261008_215649`, n=50) reproduces the baseline's first 50 `A` answers exactly (checked on CPU), so exact agreement is the expected outcome on identical software.
**On failure, do not blame the branch yet:** run the same four configs on the control tree and compare with the *same* script:
```
cd $LAKV_BASE && python run.py --mode experiment $COMMON --n_samples 20 --output_dir results/regress_base --configs A A_audit_dropped A_nopass A_nopass_audit_dropped 2>&1 | tee $OUT/logs/base_regress.log
RB=$(latest results/regress_base)/experiment_results.json
for c in A A_audit_dropped; do python $MD/scripts/check_regression_vs_baseline.py $RB $B0 --config $c; done
for c in A_nopass A_nopass_audit_dropped; do python $MD/scripts/check_regression_vs_baseline.py $RB $B1 --config $c; done
```
If the control also differs, the cause is the environment (versions, GPU, driver); if the control passes, the difference is in this branch: **STOP**. Either way send the outputs back.

### S1.3 Self-replay, n=20 (`$MD`, about 11 min, disk < 0.1 GB)
Validates the cache rebuild and the `cache_gen_len` rule on the real `generate()` path.
```
cd $MD
python run.py --mode experiment $COMMON --n_samples 20 --output_dir results/matched_selfreplay --configs A A_audit_self_replay 2>&1 | tee $OUT/logs/md_selfreplay.log
python scripts/check_matched_donors.py $(latest results/matched_selfreplay)/experiment_results.json | tee $OUT/s1/md_selfreplay_check.txt; echo "exit=${PIPESTATUS[0]}"
```
**Pass:** `GATE: PASS` (exit 0): at least 19 of 20 final answers agree **and** `not_comparable == 0`. The cosine, `max_abs_diff`, relative L2, raw-text identity and McNemar lines are **diagnostics, not gates**; read them anyway (a very low cosine with agreeing answers would mean the cache barely matters to this model).
**STOP, blocking the same-question arms (not `length_matched`):** `not_comparable > 0` means the rebuilt cache length differs from the real one, so the cache-length rule is wrong for the generate() path. If `not_comparable == 0` but fewer than 19 answers agree, bf16 prefill-vs-decode numerics may be flipping tokens: send the output back and wait for a decision before running matched_passages / matched_wrong_answer.

### S1.4 Curve pytest (`$NC`, about 5 min, disk 0)
```
cd $NC && python -m pytest tests -q 2>&1 | tee $OUT/s1/nc_pytest.txt
```
**Pass:** zero failures. CPU dev box: `302 passed, 1 skipped`; on the GPU box expect `303 passed` (parity test runs). **STOP** on any failure.

### S1.5 Curve regression at the 0% and 100% levels, n=20 (`$NC`, about 23 min, disk < 0.1 GB)
```
cd $NC
python run.py --mode experiment $COMMON --n_samples 20 --output_dir results/necessity_regress --configs A A_audit_dropped A_nopass A_nopass_audit_dropped A_hide100 A_hide100_audit_dropped 2>&1 | tee $OUT/logs/nc_regress.log
R=$(latest results/necessity_regress)/experiment_results.json
chk() { python scripts/check_regression_vs_baseline.py "$@" | tee -a $OUT/s1/nc_regress_checks.txt; echo "exit=${PIPESTATUS[0]}" | tee -a $OUT/s1/nc_regress_checks.txt; }
chk $R $B0 --config A;                           chk $R $B0 --config A_audit_dropped
chk $R $B1 --config A_nopass;                    chk $R $B1 --config A_nopass_audit_dropped
chk $R $B1 --config A_hide100 --baseline-config A_nopass
chk $R $B1 --config A_hide100_audit_dropped --baseline-config A_nopass_audit_dropped
chk $R $R  --config A_hide100 --baseline-config A_nopass           # same-run alias check
```
**Pass:** all seven `exit=0`. The first four test that the evaluator/pipeline edits did not move the paper's behavior; the last three test that `_hide100` is exactly `_nopass` (generated text, not just the config mapping). **On failure of the first four:** the control-tree fallback exactly as in S1.2 (use `$NC/scripts/...` for the script). **STOP** on any failure.

### Session 1 gate
Send back the Session 1 files (section 5) **if any step failed**. If everything passed, continue to Session 2; a good Session 1 also means `results/run_20261009_113254` and `run_20261011_122408` are valid, reusable baselines on this machine.

## 3. Session 2: n=10 smoke runs (about 57 min)

### S2.1 Matched-donor smoke (`$MD`, about 27 min, disk 8.8 GB pool)
All new arms at n=10 with a 30-donor pool (many `length_matched` skips are expected at this size). Estimate basis: mismatched pool 20 chains (3 min) + `length_matched` pool 30 chains (5 min) + traces (2 min) + 7 configs x 10 questions (12 min) + rebuilds (1.5 min) + load (3 min).
```
cd $MD
python run.py --mode experiment $COMMON --n_samples 10 --length_matched_pool_size 30 --donor_pool_dir $POOLS/smoke --output_dir results/matched_smoke --configs A A_audit_dropped A_audit_mismatched A_audit_length_matched A_audit_matched_passages A_audit_matched_wrong_answer A_audit_self_replay 2>&1 | tee $OUT/logs/md_smoke.log
python scripts/check_matched_donors.py $(latest results/matched_smoke)/experiment_results.json --expect_pairs 10 --min_agree 9 | tee $OUT/s2/md_smoke_check.txt; echo "exit=${PIPESTATUS[0]}"
python $OUT/summ.py "results/matched_smoke/run_*/experiment_results.json" | tee $OUT/s2/md_smoke_summary.txt
python - <<'EOF' 2>&1 | tee $OUT/s2/md_smoke_traces.txt
import json, glob, os; from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
for f in sorted(glob.glob(os.environ["POOLS"] + "/smoke/A/matched_wrong_answer/*.json"))[:3]:
    r = json.load(open(f)); print(r.get("distractor"), "|", t.decode(r["gen_ids"][0])[:400], "\n")
EOF
```
**Pass (from the doc):** checker exit 0 (self-replay gate at the rescaled 9 of 10, no padding, `length_matched` ratios within 1 +- 0.05); `hop0_differs_from_greedy_rate` not about 0; `defends_distractor_rate` high; skip counts sensible; no `oom_skipped`. **STOP conditions:** `hop0_differs_from_greedy_rate` = 0.0 (sampling is vacuous: `matched_passages` would equal the greedy trace; send back); `defends_distractor_rate` below 0.5 **(mine)** (the wrong-answer arm is not wrong; read the printed traces); any `oom_skipped` > 0; checker exit 1. Disk cleanup after review (optional): `rm -rf $POOLS/smoke`.

### S2.2 Curve smoke (`$NC`, about 20 min, disk < 0.1 GB)
```
cd $NC
python run.py --mode experiment $COMMON --n_samples 10 --output_dir results/necessity_smoke --hide_seed 0 --configs A_hidegold1 A_hidegold1_audit_dropped A_hidegold2 A_hidegold2_audit_dropped A_hide20 A_hide20_audit_dropped A_hide50 A_hide50_audit_dropped A_hide80 A_hide80_audit_dropped 2>&1 | tee $OUT/logs/nc_smoke.log
python scripts/check_hide_run.py $(latest results/necessity_smoke)/experiment_results.json --expect_seed 0 | tee $OUT/s2/nc_smoke_check.txt; echo "exit=${PIPESTATUS[0]}"
python $OUT/summ.py "results/necessity_smoke/run_*/experiment_results.json" | tee $OUT/s2/nc_smoke_summary.txt
python - <<'EOF' 2>&1 | tee $OUT/s2/nc_smoke_hop_texts.txt
import json, glob
d = json.load(open(sorted(glob.glob("results/necessity_smoke/run_*/experiment_results.json"))[-1], encoding="utf-8"))
for cfg in ("A_hide80", "A_hidegold2"):
    for s in d[cfg]["per_sample"][:3]:
        h = s.get("hide") or {}
        print(cfg, "| gold:", s["gold"], "| pred:", s["predicted"], "| hidden:", (h.get("hidden_titles") or [])[:3])
        print("   verifier:", (s["hop_texts"][1] if len(s["hop_texts"]) > 1 else "")[:300].replace("\n", " "))
EOF
```
The run prints a hiding summary line per hide config *before* the model loads (CPU fail-fast; the mean gold hidden should be about 0.4 / 1.0 / 1.6 for hide20/50/80 and exactly 1 / 2 for the gold levels). **Pass:** `RESULT: PASS` (real and dropped arms hid identical passages for every idx; exact counts; one seed); no `oom_skipped`. **STOP:** `RESULT: FAIL` (hard); or the Verifier output at `hide80` / `hidegold2` is degenerate (empty, repeated, or garbage; the prompt still says "the context passages given"): send it back and wait.

## 4. Session 3: n=100 runs in priority order (about 405 min)

Each stage is its own folder. Interrupted runs resume with the same command plus `--resume_dir results/<dir>/run_<ts>` (and the same `--donor_pool_dir`). **After every run:** `python $OUT/summ.py "results/<dir>/run_*/experiment_results.json"`; **any `oom_skipped` > 0 in a new arm is a STOP** (the evaluator records an OOM as a skipped sample and a resume does not retry it): rerun that stage in a fresh folder after freeing the GPU (`nvidia-smi`).

### 3.1 `A_audit_length_matched` (`$MD`, about 35 min, **disk 29.4 GB** in `$POOLS/lm_A`)
Basis: 100-chain pool (15 min, doc estimate) + 100 questions (17 min) + load.
```
cd $MD
python run.py --mode experiment $COMMON --n_samples 100 --length_matched_pool_size 100 --donor_pool_dir $POOLS/lm_A --output_dir results/matched_lm --configs A_audit_length_matched 2>&1 | tee $OUT/logs/md_lm.log
python scripts/check_matched_donors.py $(latest results/matched_lm)/experiment_results.json | tee $OUT/s3/md_lm_check.txt; echo "exit=${PIPESTATUS[0]}"
python $OUT/paired.py "results/matched_lm/run_*/experiment_results.json" A_audit_length_matched $B0 A A_audit_dropped A_audit_mismatched | tee $OUT/s3/md_lm_paired.txt
python $OUT/summ.py "results/matched_lm/run_*/experiment_results.json" | tee $OUT/s3/md_lm_summary.txt
```
**Pass:** checker exit 0 (no padding or truncation; every ratio within 1 +- 0.05). Skip rate expected about 12% (**estimate**: about 94% of targets have a within-5% donor per hop at 100 donors, both hops needed: 1 - 0.94^2). **STOP / investigate:** checker exit 1; skip rate above 25% **(mine)** (pool too small or lengths unlike the estimate); `oom_skipped`. Read results on the paired subset (`paired.py` prints `n_pairs`; skips are not random).

### 3.2 Curve gold axis, PRIMARY (`$NC`, about 70 min, disk < 0.2 GB)
```
cd $NC
python run.py --mode experiment $COMMON --n_samples 100 --output_dir results/necessity_gold --hide_seed 0 --configs A_hidegold1 A_hidegold1_audit_dropped A_hidegold2 A_hidegold2_audit_dropped 2>&1 | tee $OUT/logs/nc_gold.log
python scripts/check_hide_run.py $(latest results/necessity_gold)/experiment_results.json --expect_seed 0 | tee $OUT/s3/nc_gold_check.txt; echo "exit=${PIPESTATUS[0]}"
python $OUT/summ.py "results/necessity_gold/run_*/experiment_results.json" | tee $OUT/s3/nc_gold_summary.txt
```
**Pass:** `RESULT: PASS` (the same hidden passages in the real and dropped arms is a hard requirement). **STOP:** FAIL; `frac_empty_answer` above 0.05 **(mine)** in any arm (degenerate generations); `oom_skipped`.

### 3.3 `matched_passages` + `matched_wrong_answer` (`$MD`, about 66 min, disk < 0.1 GB of token-id traces)
Needs S1.3 to have passed. Basis: traces 12 + 8 min (doc), 2 configs x 100 questions (33 min), rebuild prefills about 10 min, load 3.
```
cd $MD
python run.py --mode experiment $COMMON --n_samples 100 --donor_pool_dir $POOLS/same_A --output_dir results/matched_same --configs A_audit_matched_passages A_audit_matched_wrong_answer 2>&1 | tee $OUT/logs/md_same.log
python scripts/check_matched_donors.py $(latest results/matched_same)/experiment_results.json | tee $OUT/s3/md_same_check.txt; echo "exit=${PIPESTATUS[0]}"
python $OUT/paired.py "results/matched_same/run_*/experiment_results.json" A_audit_matched_passages $B0 A A_audit_dropped A_audit_mismatched | tee $OUT/s3/md_same_paired.txt
python $OUT/paired.py "results/matched_same/run_*/experiment_results.json" A_audit_matched_wrong_answer $B0 A A_audit_dropped A_audit_mismatched | tee -a $OUT/s3/md_same_paired.txt
```
**Pass:** checker exit 0; `hop0_differs_from_greedy_rate` clearly above 0 (else the sampled trace is the greedy one and the control is vacuous); `defends_distractor_rate` high. **STOP / report:** exit 1; `hop0_differs_from_greedy_rate` about 0 (same stop as the smoke); a large `distractor_failed` skip count in `matched_wrong_answer` (**mine:** above 25% of questions).

### 3.4 Curve fractional axis, secondary (`$NC`, about 103 min, disk < 0.2 GB)
```
cd $NC
python run.py --mode experiment $COMMON --n_samples 100 --output_dir results/necessity_frac --hide_seed 0 --configs A_hide20 A_hide20_audit_dropped A_hide50 A_hide50_audit_dropped A_hide80 A_hide80_audit_dropped 2>&1 | tee $OUT/logs/nc_frac.log
python scripts/check_hide_run.py $(latest results/necessity_frac)/experiment_results.json --expect_seed 0 | tee $OUT/s3/nc_frac_check.txt; echo "exit=${PIPESTATUS[0]}"
python $OUT/summ.py "results/necessity_frac/run_*/experiment_results.json" | tee $OUT/s3/nc_frac_summary.txt
```
Pass and STOP as in 3.2.

### 3.5 Seed-1 sensitivity pair at hide50 (`$NC`, about 36 min, disk < 0.1 GB)
```
cd $NC
python run.py --mode experiment $COMMON --n_samples 100 --output_dir results/necessity_seed1 --hide_seed 1 --configs A_hide50 A_hide50_audit_dropped 2>&1 | tee $OUT/logs/nc_seed1.log
python scripts/check_hide_run.py $(latest results/necessity_seed1)/experiment_results.json --expect_seed 1 | tee $OUT/s3/nc_seed1_check.txt; echo "exit=${PIPESTATUS[0]}"
```
Pass and STOP as in 3.2 (note `--expect_seed 1`). This is the pair `A_hide50` / `A_hide50_audit_dropped` run with `--hide_seed 1` (the seed is a run flag, not part of the config name; see UNVERIFIED N12).

### 3.5b Curve table and figures (`$NC`, CPU, about 2 min)
```
cd $NC
G=$(basename $(latest results/necessity_gold)); F=$(basename $(latest results/necessity_frac)); S=$(basename $(latest results/necessity_seed1))
sed -e "s#<TS_GOLD>#${G#run_}#g; s#<TS_FRAC>#${F#run_}#g; s#<TS_SEED1>#${S#run_}#g" docs/necessity_curve_manifest.example.json > docs/necessity_curve_manifest.local.json
python scripts/make_necessity_curve.py --manifest docs/necessity_curve_manifest.local.json --out_dir results/necessity_curve 2>&1 | tee $OUT/s3/nc_curve_report.txt
```
Levels 0 and 100% reuse the two baselines. **Pass:** the script finishes and writes `necessity_curve.png`, `necessity_curve_accuracy.png` and `necessity_curve_table.csv`; the level-0 and 100% rows should read about -3 pp (p about 0.68) and +32 pp (p about 4e-7), as in the baselines. **STOP:** a `KeyError` naming a config (a wrong run folder in the manifest).

### 3.6 All `A_nopass` matched-donor variants, LAST (`$MD`, about 95 min, **disk about 25 to 29 GB** in `$POOLS/nopass`)
Needs S1.3 (for the same-question arms) and S1.2's `_nopass` regression (to reuse the baseline `A_nopass*` arms). Basis: 100-chain pool (15 min), traces 20 min, 3 configs x 100 questions (50 min), rebuilds about 7 min, load 3.
```
cd $MD
python run.py --mode experiment $COMMON --n_samples 100 --length_matched_pool_size 100 --donor_pool_dir $POOLS/nopass --output_dir results/matched_nopass --configs A_nopass_audit_length_matched A_nopass_audit_matched_passages A_nopass_audit_matched_wrong_answer 2>&1 | tee $OUT/logs/md_nopass.log
python scripts/check_matched_donors.py $(latest results/matched_nopass)/experiment_results.json | tee $OUT/s3/md_nopass_check.txt; echo "exit=${PIPESTATUS[0]}"
for c in A_nopass_audit_length_matched A_nopass_audit_matched_passages A_nopass_audit_matched_wrong_answer; do
  python $OUT/paired.py "results/matched_nopass/run_*/experiment_results.json" $c $B1 A_nopass A_nopass_audit_dropped A_nopass_audit_mismatched | tee -a $OUT/s3/md_nopass_paired.txt; done
```
(The curve branch has no further `_nopass` runs: its `_hide100` is only the regression alias.) **Optional:** to rerun the three baseline arms too, as the matched-donor doc does, add `A_nopass A_nopass_audit_dropped A_nopass_audit_mismatched` to `--configs` (about +50 min). **Pass and STOP** as in 3.1 and 3.3.

## 5. What to send back

Create each archive from the worktree that produced it; pools (`*.pt`, `$POOLS`) are never needed. If a run crashed, also include its `experiment_results.partial.json`.
```
pack() { name=$1; shift; tar czf $OUT/$name.tgz --exclude='*.pt' --exclude='audit_pool' "$@"; }   # e.g. (cd $MD && pack s1_md results/regress_md results/matched_selfreplay)
```

| Stage | Send back |
|---|---|
| Setup / env | `$OUT/s1/env_report.txt`, `pip_freeze.txt`, `dataset_check.txt` |
| S1.1, S1.4 | `md_pytest.txt`, `nc_pytest.txt` (full output on any failure) |
| S1.2 | `results/regress_md/run_*/{run_metadata.json,experiment_results.json}`, `$OUT/s1/md_regress_checks.txt`, `logs/md_regress.log`; on failure also the control-tree outputs (`results/regress_base`, `logs/base_regress.log`) |
| S1.3 | `results/matched_selfreplay/run_*/{run_metadata.json,experiment_results.json}`, `md_selfreplay_check.txt`, `logs/md_selfreplay.log` |
| S1.5 | `results/necessity_regress/run_*/{run_metadata.json,experiment_results.json}`, `nc_regress_checks.txt`, `logs/nc_regress.log` |
| S2.1 | `results/matched_smoke/run_*/{run_metadata.json,experiment_results.json}`, `md_smoke_check.txt`, `md_smoke_summary.txt`, `md_smoke_traces.txt`, `$POOLS/smoke/A/matched_passages_seed0/*.json` and `.../matched_wrong_answer/*.json` (small JSON only), `logs/md_smoke.log` |
| S2.2 | `results/necessity_smoke/run_*/{run_metadata.json,experiment_results.json}`, `nc_smoke_check.txt`, `nc_smoke_summary.txt`, `nc_smoke_hop_texts.txt`, `logs/nc_smoke.log` |
| 3.1 | `results/matched_lm/run_*/{run_metadata.json,experiment_results.json,results_table.csv,per_hop_stats.json}`, `md_lm_check.txt`, `md_lm_paired.txt`, `md_lm_summary.txt`, `logs/md_lm.log` |
| 3.2, 3.4, 3.5 | the same four files for `results/necessity_gold`, `necessity_frac`, `necessity_seed1`, plus `nc_{gold,frac,seed1}_check.txt`, `nc_*_summary.txt`, logs |
| 3.5b | `results/necessity_curve/` (two PNGs, CSV, JSON), `nc_curve_report.txt`, `docs/necessity_curve_manifest.local.json` |
| 3.3 | `results/matched_same/...` (four files), `md_same_check.txt`, `md_same_paired.txt`, the trace JSON from `$POOLS/same_A/*/` (small), `logs/md_same.log` |
| 3.6 | `results/matched_nopass/...` (four files), `md_nopass_check.txt`, `md_nopass_paired.txt`, `logs/md_nopass.log` |

Always include the checker output **text** (not only the exit code): the checkers print the diagnostics that matter (`min_cosine_worst`, `defends_distractor_rate`, `hop0_differs_from_greedy_rate`, realized gold hidden, empty-answer rates).

## 6. Failure playbook and cleanup

| Symptom | Action |
|---|---|
| any traceback | send `logs/<name>.log`, `git rev-parse HEAD` of the tree, `pip_freeze.txt`; do not rerun blindly |
| `oom_skipped` > 0 | free the GPU (`nvidia-smi`, kill stray processes), rerun the stage in a fresh folder (resume does not retry OOM samples) |
| run interrupted | rerun the same command with `--resume_dir results/<dir>/run_<ts>` (and the same `--donor_pool_dir`); `length_matched` pools are reused only if complete and of exactly the requested size |
| disk nearly full | after a stage's results are archived: `rm -rf $POOLS/smoke` (S2.1), `rm -rf $POOLS/lm_A` (after 3.1; frees about 29 GB but prevents reuse or resume of that pool) |
| regression fails | the control-tree fallback in S1.2 / S1.5; never proceed on that branch |
| `RESULT: FAIL` from `check_hide_run.py` with "hidden passages differ" | a bug, not noise (the real and dropped arms must hide identical passages); stop the curve and send the output |

## 7. UNVERIFIED items (both docs, plus this runbook)

Status for every row is **open** until you report otherwise. "Resolved by" names the step that settles it.

| # | Item | From | Resolved by |
|---|---|---|---|
| M1 | Exact original command of `run_20261009_113254` was never recorded; the flags in the regression are inferred from the result file, `PREDICTIONS_BEFORE_RUNS_2026-10-11.md:79` (donor seed 0) and `run.py` defaults | matched-donor step 0 | S1.2 / S1.5 (exact agreement supports the inference; it does not recover the command) |
| M2 | Model id of that run is inferred (`Qwen/Qwen2.5-7B-Instruct`: default, 28 layers, paper/CLAUDE.md attribution) | matched-donor | S1.2 |
| M3 | The profile that run used is unknown (irrelevant for `A`, never read) | matched-donor | not needed |
| M4 | `--greedy`, `--arch`, `--device` of that run unknown; defaults assumed (greedy is already the default `do_sample=False`) | matched-donor | S1.2 |
| M5 | Code version of that run: no commit hash; `_audit_dropped` first committed in `f17c136` (2026-10-29), after the run, so it used an uncommitted working tree | matched-donor | S1.2 (tests the presumption that `A`'s logic is unchanged) |
| M6 | Library versions and hardware of the baseline runs not recorded; pins come from four 2026-09-08 GSM8K smoke runs (n=3) whose recorded commit `a47168d` is on `feat/lakv_nvidia`, not on these branches | matched-donor / pins | 1.3 and S1.2 (same-machine reproduction) |
| M7 | All matched-donor code never ran with a real model; the `generate()`-handoff `cache_gen_len` rule for hops >= 1 is verified only by GPU self-replay | matched-donor | S1.3 |
| M8 | Rebuilt-by-prefill vs decode-built caches differ by bf16 numerics; there is no validated numeric threshold (the 19/20 answer-agreement gate is a chosen criterion, cosine is diagnostic only) | matched-donor | S1.3 (read the diagnostics) |
| M9 | Sampling is not guaranteed bit-reproducible across hardware (hence stored token ids) | matched-donor | design; not testable |
| M10 | `matched_wrong_answer` pairs standard-prompt K/V with reasoning produced under the defend prompt; the Verifier trace was generated on the generation-time Reasoner cache but rebuilt on the standard one | matched-donor | design caveat; interpret, do not "fix" |
| M11 | Skipped samples bias the arms that skip (`length_matched`, `matched_wrong_answer`) | matched-donor | read paired subsets (3.1, 3.3, 3.6) |
| M12 | Pool/trace build times and per-substitution rebuild cost (15 / 12 / 8 min; 1 to 2 s) are estimates; disk sizes (29.4 GB) come from mean cache lengths of the v9 runs | matched-donor | timings in the logs |
| M13 | The expected skip rate (about 12%) is an iid estimate from scored-question cache lengths | matched-donor | 3.1 |
| M14 | CPU tests ran on transformers 4.44.2 / torch 2.4.1, not the pinned stack; the end-to-end tiny-model tests force the manual decode loop | matched-donor | S1.1 on the GPU box |
| N1 | Original commands of `run_20261011_122408` (the `_nopass` baseline) also unrecorded; `_nopass` first committed in `f17c136`, after both baselines | necessity | S1.2 / S1.5 |
| N2 | Library versions and hardware of both baselines not recorded | necessity | 1.3, S1.5 |
| N3 | That the GPU box's HotpotQA download is the same revision as the one checked: on the dev box the real loader reproduced both baselines' 100 questions exactly (datasets 4.4.1, hub revision `1908d6af...`); the GPU box's datasets is 5.0.1 | necessity | 1.4 (digest and equality lines) |
| N4 | Gold = `supporting_facts` titles holds beyond the first 100 items (checked: exactly 2 gold titles for the first 100 only) | necessity | 1.4 prints it for the first 100; items 100 to 199 only matter for pools |
| N5 | Prompt-token counts are user-content only (Qwen2.5 tokenizer); system prompt and chat-template tokens add a constant | necessity | none needed |
| N6 | Whether intermediate hide levels trigger degenerate Verifier output (prompt wording is unchanged by hiding) | necessity | S2.2, 3.2, 3.4 (hop_texts, empty-answer rates) |
| N7 | Why `run_20261009_113254`'s real arm took 22.4 s per question (8.4 s in the hidden run) | necessity | latency lines in `summ.py` output |
| N8 | That `A_hide100` reproduces `A_nopass` on the GPU (the CPU prototype and tests show only prompt-string identity) | necessity | S1.5 |
| N9 | The hide path through `LAKV pipeline run`, the evaluator wiring and the figure script on real results never ran with a real model | necessity | S2.2, 3.2 to 3.5b |
| N10 | CPU test environment is not the pinned stack (tiny-model tests force the manual decode loop; the handoff path is covered only by GPU regression) | necessity | S1.4, S1.5 |
| N11 | Dataset-revision logging is best-effort: the cache-revision directory is read from the `datasets` cache path (worked on the dev box with datasets 4.4.1: `cache_revision_dirs` was filled) and may differ under 5.0.1; the SHA-256 content digests are the dependable identity | necessity | 1.4 and `run_metadata.json` of any curve run |
| N12 | Naming of the seed-1 pair: `A_hide50` and `A_hide50_audit_dropped` with `--hide_seed 1` (you wrote `A_audit_dropped_hide50`, which does not match the suffix grammar); manifest placeholders `<TS_*>` | necessity | your confirmation; 3.5b |
| N13 | `A_hide100` equals `A_nopass` by construction in the evaluator (alias), not by a separate implementation | necessity | S1.5 |
| R1 | GPU machine OS, shell and Python are unknown; commands are bash and untested on that machine (the dev box is Windows; the clone, checkout and worktree commands of 1.1 (run against GitHub), the three helper scripts and the dataset load were tested here in temporary directories with CPU torch) | this runbook | 1.1 to 1.4 |
| R2 | Model load time per `run.py` (assumed 3 min), whether the 15 GB model is already cached, and the parity test's duration | this runbook | logs |
| R3 | The driver version needed for the `cu128` build is not stated in the repo | this runbook | 1.3 (check NVIDIA's table) |
| R4 | `PROFILE=profiles/qwen_gsm8k.json` is a GSM8K profile used only to satisfy the required flag (config `A` never reads it, `lakv/evaluator.py` attaches a profile only for layer selection or adaptive compression); its hash is recorded as `profile_sha256` in the metadata, which is therefore not a HotpotQA calibration | this runbook | use a HotpotQA profile if you have one |
| R5 | The thresholds marked **(mine)** (skip rate 25%, empty answers 5%, `defends_distractor_rate` 0.5) are judgement calls, not from either doc | this runbook | your review |
| R6 | Total time and disk figures (about 70 min, 57 min, 6.75 h, peak about 60 GB) are sums of estimates; the real-arm latency is the largest uncertainty (8.4 to 22.4 s) | this runbook | logs |
