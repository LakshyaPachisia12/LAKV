# Predictions recorded before the new runs (2026-10-11)

Written on 2026-10-11 (this file's creation date, 11:25 local time), before the runs
below were started. NOT committed to git: it has no commit timestamp, so the
"written in advance" claim rests on file-system dates only and this is not a formal
preregistration. Do not edit the predictions after runs start; add a dated
"Outcome" section at the bottom instead.

All runs: HotpotQA validation, greedy decoding, paired exact McNemar + paired
bootstrap F1 CI, same examples as the existing runs they pair with.

## Run 1 -- chain passage-access ablation (Qwen2.5-7B, n=100)
Configs: `A_nopass`, `A_nopass_audit_dropped`, `A_nopass_audit_mismatched`
(Verifier/Finalizer see only the bare question; the cache is their only route).

Prediction: with passages removed, real beats dropped clearly (dropped falls
toward the closed-book level seen in the fan-in, ~25%), and mismatched is still
below dropped. If real stays ~ equal to dropped, the "sole route" explanation is
WRONG and the paper's mechanism claim must be rewritten.

## Run 2 -- wrong-text control (Qwen2.5-7B, n=100)
Configs: `text_agent`, `text_agent_audit_dropped`, `text_agent_audit_mismatched`.

No directional prediction on KV-specificity. Decision rule fixed in advance:
- mismatched-text harm (dropped minus mismatched) within ~10 accuracy points of
  the mismatched-cache harm (~30 points) -> report the harm as generic wrong
  inter-agent-message harm, NOT KV-specific.
- mismatched-text harm clearly smaller -> report KV-specific amplification.

## Run 3 -- fan-in with passages given to the aggregator (n=100)
`--aggregator_sees_passages`, channels `kv kv_audit_dropped kv_audit_mismatched`.

Prediction: dropped catches up to real (the aggregator no longer depends on the
cache), and mismatched is still worse than dropped.

## Runs 4-6 -- scale and robustness
- Mistral-7B and Qwen3-8B at n=100 (A, A_audit_dropped, A_audit_mismatched):
  prediction -- Mistral keeps dropped > mismatched (p < 0.01); Qwen3-8B stays
  insensitive (mismatched ~ dropped).
- Donor seeds 1 and 2 (Qwen2.5 A_audit_mismatched, n=100): prediction -- the
  mismatched accuracy stays within ~8 points of the seed-0 result (28-29%).

## Outcome
(to be filled in after the runs)

### Outcome: Run 3b -- fan-in text-relay row (2026-10-11, results/recursive_poc_check_text/run_20261011_120349)
No prediction was recorded for this run (exploratory). Result, Qwen2.5-7B, n=100:
text 51.0% EM / 0.609 F1 vs KV 37.0% / 0.466; exact McNemar p=0.0094 (20 text-only,
6 KV-only); paired bootstrap F1 diff +0.144 [+0.050, +0.238]. Text vs dropped
(closed-book, 25.0%): p=6e-6. The KV channel's per-example outcomes are identical
to the earlier n=100 fan-in run (100/100), confirming determinism.

### Outcome: Run 1 -- chain passage-access ablation (results/run_20261011_122408, n=100, Qwen2.5-7B)
Prediction CONFIRMED on all three parts. A_nopass 52.0% EM / 66.0 F1; A_nopass_audit_dropped
20.0% / 29.6 (near the closed-book ~25% of the fan-in); A_nopass_audit_mismatched 7.0% / 12.1.
Paired exact McNemar within the ablation: real vs dropped p=4.4e-7 (37 vs 5 discordant), F1
diff +36.5 [+25.6,+47.2]; real vs mismatched p=4.4e-12; dropped vs mismatched p=9.8e-4 (14 vs 1).
Against the standard chain (passages present, run_20261009_113254): real 56 vs 52, p=0.39 (n.s.);
dropped 59 vs 20, p=2.2e-10; mismatched 29 vs 7, p=1.1e-5.

### Outcome: Run 3 -- fan-in with passages given to the aggregator (results/recursive_poc_check_withpass/run_20261011_132223, n=100, Qwen2.5-7B)
Prediction CONFIRMED. kv 55.0% / 68.6 F1; kv_audit_dropped 57.0% / 68.5; kv_audit_mismatched 30.0% / 41.1.
Real vs dropped p=0.80 (7 vs 9 discordant), F1 diff +0.2 [-5.9,+6.0]; real vs mismatched p=1.1e-4;
dropped vs mismatched p=1.4e-5. Against the standard fan-in (passages absent, run_20261009_125426):
real 37 -> 55 (p=2.8e-4); dropped 25 -> 57 (p=6.7e-8); mismatched 8 -> 30 (p=2.7e-5).
Real-minus-dropped moves from +12 points (p=0.050) to -2 points (p=0.80).

### Outcome: Run 4 -- Mistral-7B chain at n=100 (results/run_20261011_141020)
Prediction (Mistral keeps dropped > mismatched, p < 0.01): CONFIRMED, but the effect is smaller.
A 53.0% / 65.1 F1; A_audit_dropped 49.0% / 59.7; A_audit_mismatched 26.0% / 36.5.
real vs dropped p=0.454 (F1 +5.4 [-1.7,+13.2]); real vs mismatched p=1.4e-6 (30 vs 3);
dropped vs mismatched p=1.9e-4 (30 vs 7), F1 +23.2 [+13.2,+33.1].
Wrong-cache penalty (dropped minus mismatched): 32 points at n=50, 23 at n=100. The first 50
questions reproduce real (56%) and dropped (48%) exactly; mismatched differs (16% -> 22%)
because the donor pool is drawn from held-out examples beyond n and therefore changes with n.

### Outcome: Run 5 (donor seed 1 only) -- Qwen2.5 A_audit_mismatched, n=100 (results/run_20261011_150444)
Seed identified by simulating random.Random(seed) donor draws against the logged donor questions
(seed 0 matches run_20261009_113254; seed 1 matches this run). Prediction (within ~8 points of the
seed-0 result, 29%): CONFIRMED -- 23.0% EM / 34.8 F1, a 6-point difference.
Seed 0 vs seed 1: 29 vs 23, paired McNemar p=0.26 (13 vs 7 discordant), F1 diff +5.8 [-2.4,+14.0].
No cache (59%) vs mismatched seed 1: p=1.5e-9 (38 vs 2 discordant). Real (56%) vs mismatched seed 1: p=2.5e-7.

### Outcome: Run 5 (donor seed 2) -- Qwen2.5 A_audit_mismatched, n=100 (results/run_20261011_184238)
Seed 2 confirmed by the same donor-draw simulation. 23.0% EM / 35.8 F1. Prediction CONFIRMED.
Three draws (seeds 0, 1, 2): 29.0 / 23.0 / 23.0% EM (mean 25.0), F1 40.6 / 34.8 / 35.8.
Pairwise paired McNemar: seed0 vs seed2 p=0.24 (12 vs 6); seed1 vs seed2 p=1.0 (12 vs 12).
No cache (59%) vs mismatched seed 2: p=2.8e-10 (37 vs 1); real (56%) vs mismatched seed 2: p=1.0e-7.
Wrong-cache penalty vs no cache across the three draws: 30, 36, 36 points.

### Outcome: Run 2 -- wrong-text control, Qwen2.5 chain (results/run_20261011_202231, n=100)
Decision rule recorded in advance: text harm within ~10 points of the cache harm (~30) -> generic harm;
clearly smaller -> KV-specific amplification. Result: text relay 54.0% / 59.0% (dropped) / 47.0% (mismatched),
harm 12 points vs 30 for the cache, an 18-point gap (paired bootstrap 95% CI [7,29]) -> amplification rule met.
BUT wrong text is itself significantly worse than no relay (59 vs 47, exact McNemar p=0.0005, 12 vs 0
discordant), so the harm is NOT KV-only. Wrong text vs wrong cache (47 vs 29): p=0.0021. Real text vs
real cache (54 vs 56): p=0.82. Text real vs dropped p=0.36 (n.s., same pattern as KV).
Caveat: wrong cache carries the donor's passages as well as its reasoning; wrong text carries reasoning only.
Note: the first launch crashed (empty text pool, a bug in the evaluator's plain text_agent shortcut);
fixed with regression tests (173 pass) and resumed.

### Reproduction: Run 2 rerun from scratch (results/run_20261011_211406)
A fresh, non-resumed rerun of text_agent / text_agent_audit_dropped / text_agent_audit_mismatched gives
54.0 / 59.0 / 47.0% EM (F1 68.3 / 68.9 / 57.8), identical to run_20261011_202231: 100/100 identical
correctness and raw answers in every config, and 100/100 identical donor assignments for the mismatched
config. All Run 2 statistics above therefore stand unchanged.

### Outcome: Run 6 -- Qwen3-8B chain at n=100 (results/run_20261011_221317)
Prediction (Qwen3-8B stays insensitive, mismatched ~ dropped): CONFIRMED.
A 63.0% / 75.3 F1; A_audit_dropped 62.0% / 72.8; A_audit_mismatched 62.0% / 71.6.
All three pairwise exact McNemar p = 1.0 (real-dropped 12 vs 11; real-mismatched 14 vs 13; dropped-mismatched 3 vs 3).
Output identity: mismatched = dropped on 80/100 raw answers; real = dropped on 41/100; real = mismatched on 32/100
(the model ignores a wrong cache but does respond to the real one in its wording, with no accuracy change).
Holm family recomputed with all primary legs at n=100: every significant comparison stays significant;
the two Qwen3-8B legs are 1.0 -> 1.0.

### Outcome: text relay with passages hidden (results/run_20261011_231449, n=100, Qwen2.5-7B) -- exploratory, no prediction recorded
text_agent_nopass 48.0% / 59.2 F1; _audit_dropped 20.0% / 30.0; _audit_mismatched 14.0% / 18.6.
Text real vs dropped p=4.3e-6 (33 vs 5): the sole-route benefit is NOT KV-specific (KV: 52 vs 20, p=4.4e-7).
Text dropped vs mismatched p=0.146 (9 vs 3), harm 6 points (KV 13); wrong text vs wrong cache 14 vs 7, p=0.065;
harm gap 7 points, paired bootstrap 95% CI [0,15] (floor effect: dropped is only 20%).
Real KV vs real text 52 vs 48, p=0.45 (F1 +6.8 [+0.2,+13.7]) -> a TIE on accuracy.
Consequence: the fan-in deficit (text 51 vs KV 37, p=0.009) is specific to merging two independently
computed caches, not to the KV channel; the earlier claim "text beats KV where the cache is the only
route" was too broad and has been narrowed in the paper.

### Outcome: Mistral donor seed 1 -- A_audit_mismatched, n=100 (results/run_20261011_234807)
Seed identified by donor-draw simulation (seed 1). 19.0% EM / 30.8 F1 vs 26.0% / 36.5 for seed 0.
Seed 0 vs seed 1: paired McNemar p=0.17 (13 vs 6), F1 diff +5.7 [-1.7,+13.6]. No cache (49%) vs mismatched
(this draw): p=2.3e-7 (33 vs 3); real (53%) vs mismatched: p=1.1e-9. Mistral wrong-cache penalty: 23 and 30 points.

### Outcome: Mistral donor seed 2 -- A_audit_mismatched, n=100 (results/run_20261012_093630)
Seed 2 confirmed by donor-draw simulation. 21.0% EM / 33.1 F1. Three Mistral draws: 26 / 19 / 21% (mean 22.0).
Pairwise paired McNemar between draws: seed0-seed1 p=0.17; seed0-seed2 p=0.38; seed1-seed2 p=0.80 (none significant).
No cache (49%) vs mismatched seed 2: p=1.9e-6 (32 vs 4); real (53%) vs mismatched seed 2: p=4.1e-9.
Wrong-cache penalty on Mistral across draws: 23, 30, 28 points.
