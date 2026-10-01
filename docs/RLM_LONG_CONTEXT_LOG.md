# RLM+KV long-context experiment log (branch `feat/rlm-long-context`)

This is a separate, deliberately isolated experimental log for this
branch only — not part of `CLAUDE.md`'s numbered-findings sequence,
which tracks the actual submittable-paper narrative on
`feat/research-extensions`. Branched off `feat/research-extensions` at
`ab9a49d` on 2026-09-22, specifically to test whether RLM+KV's `kv`-vs-
`text` comparison and its causal audit behave differently in the context
regime RLM's own paper (Zhang, Kraska, Khattab, arXiv 2512.24601)
reports as where recursive decomposition should matter (beyond
2^14 = 16,384 tokens) — HotpotQA's own distractor setting (~10 passages,
~2-3K tokens) doesn't need decomposition at all, which is a real,
sharp critique of testing RLM-style delegation on it. See the
2026-09-22 conversation thread (research-audit turn) for the full
motivation and the explicit go/no-go checkpoint discipline this branch
was built under: cheap smoke tests first, cut losses if the signal
doesn't hold up as n grows, keep the safe `feat/research-extensions`
work completely protected regardless of outcome.

## Infrastructure built

- `lakv/long_context_hotpotqa.py`: needle-in-haystack loader.
  Combines each question's real passages (needle) with filler drawn
  from a disjoint pool of other questions (haystack), grown to a
  target token count via the real tokenizer, shuffled together. 6 unit
  tests, CPU-only.
- `lakv/stats.py` additions (this branch only, not yet on
  `feat/research-extensions`): `cochran_armitage_trend_test` and
  `discordant_pair_trend_across_conditions` — tests whether a
  proportion (or a paired win-rate) trends across ordered conditions
  (e.g. context length), more statistically efficient than one large
  run at a single length when the hypothesis is genuinely trend-shaped.
  7 unit tests.
- `scripts/rlm_long_context_sweep.py`: runs `text`/`kv` at several
  context lengths on the same underlying questions (nested haystacks
  via a shared seed), tests for a trend in kv's discordant-pair win
  rate across lengths.
- `scripts/rlm_long_context_audit_check.py`: causal audit
  (zeroed/random/mismatched vs. real `kv`) at one context length,
  reusing `RLMKVSession`'s existing audit machinery directly. Reports
  both raw EM/F1 and the delegated-and-completed rate (see
  `CLAUDE.md`'s finding 20 UPDATE 2026-09-22 on
  `feat/research-extensions` for where that metric came from).

All CPU-only mechanics are unit-tested; full suite 134/134 passing on
this branch as of the last commit.

## Results so far, and the honest read

### Dose-response sweep (does kv's accuracy edge over text grow with length?)

Three runs, increasing n, same three lengths (2000/8000/16000 tokens):

| n | 2000 tokens gap | 8000 tokens gap | 16000 tokens gap | Trend test |
|---|---|---|---|---|
| 3 | +66.7 pts | — | — | not computed |
| 8 | +12.5 pts | +0.0 pts | +12.5 pts | not significant |
| 20 | **+0.0 pts** | +0.0 pts | +5.0 pts | z=0.44, p=0.66 |

**The apparent kv advantage shrank as n grew, not stabilized or
strengthened** — the classic shape of an early false positive fading
under real data, and near-identical to the pattern `CLAUDE.md` Finding
18 already documented once for the native-context `kv`-vs-`text`
comparison (an n=25 pilot showing a clean win that didn't survive an
n=50 rerun). **Verdict: this specific hypothesis (kv's raw-accuracy
edge grows with context length) is not supported by the data collected
so far, and further scaling on the identical design is a low-probability
bet, not a promising one to keep pushing without changing something.**

Checked for a plumbing bug before accepting this: pulled several
8000-token transcripts directly. All showed coherent, real attempts
("no" instead of "yes" on a yes/no question, "Margaret Thatcher" as a
wrong-but-plausible guess, a genuine 13-delegation-call search that
ran out of turns) — not garbage, not repetition loops. The low, flat
accuracy reflects genuine task difficulty at this scale with this
model/scaffolding, not a broken loop.

**Real methodological finding along the way**: the loader's passage
shuffling (deliberate, to avoid the needle always sitting in a
predictable position) measurably changes the baseline even at minimal
filler. Same 20 questions, native unshuffled order
(`results/rlm_kv_check/run_20260916_203322`): `kv` 30%, `text` 20%.
Same 20 questions through the shuffled "2000 tokens" bucket: `kv` 15%,
`text` 15%. Not significant at this n (McNemar p=0.375 for kv, p=1.0
for text — too few discordant pairs to prove it), but numerically
consistent for both channels. Practical implication: don't compare this
branch's sweep numbers directly against the established native-context
baseline as if they're the same condition — they aren't, independent of
context length.

### Causal audit pivot (does real content beat corrupted content at length, regardless of raw accuracy?)

Pivoted here (2026-09-22) because every causal audit anywhere else in
this project has found a real effect regardless of whether `kv` ever
beat `text` on raw accuracy — a fundamentally easier bar than the
dose-response comparison needed. Concentrated all GPU time at one
length (16,000 tokens, closest to RLM's own reported crossover) instead
of spreading across several.

n=5 smoke test: delegated-and-completed rate `kv` 60%, `kv_audit_zeroed`
40%, `kv_audit_mismatched` 40%, `kv_audit_random` 0% — the right
qualitative direction, though obviously unpowered at this n (p=0.25-1.0
everywhere). One qualitatively interesting single example (idx 4, "Big
Stone Gap" director question) got the identical wrong-but-partial-credit
answer ("New York City") under `kv`, `kv_audit_zeroed`, AND
`kv_audit_mismatched` (all genuinely delegated, confirmed via
`audit_logs` — not a leak), only `kv_audit_random` broke the pattern
(timed out instead) — suggestive of the model falling back on general/
parametric knowledge regardless of which well-formed content it
received, similar in shape to Finding 17's C2C result (channel presence
mattered, channel identity didn't) — one anecdote, not a claim.

n=20 real run: **did not hold up, and for a diagnosable reason.**
EM/F1 came back numerically identical across all four channels (5.0%
EM, F1=0.102, to 3 decimal places) — checked this precisely rather than
trusting it: confirmed real, not a bug. **11 of 20 examples (55%),
identically across every channel, never delegate at all** — solved
directly via the leaky pathway (the root reading `passages[i]` itself,
same mechanism as CLAUDE.md's finding 20, now more dominant at long
context than at native scale: 55% here vs. 24% native). Of the
remaining 9 examples that do attempt delegation, most time out rather
than complete (`kv`: 2 completed / 2 timed-out-delegating;
`kv_audit_random`: 0 completed / 4 timed-out-delegating). The
genuinely-informative sample per channel is down to 0-3 examples —
nowhere near enough for the McNemar tests to say anything (p=0.5-1.0
everywhere, 1-2 discordant pairs).

**Diagnosis, not just an observation**: with 10 needle passages
shuffled uniformly among 50 filler passages, the model has decent odds
of encountering at least one real, relevant passage within its first
few direct reads — often enough to produce *a* plausible-sounding
answer (right or wrong) without ever needing to delegate. This is
higher than at native scale, not lower, which is the opposite of what
the experiment needs: more filler should be making direct reading
LESS viable, forcing more genuine delegation, not letting the model
"get lucky" more often. Two real, fixable candidates for why:
(1) `max_turns=15` may simply be too tight once delegation is
genuinely attempted against a ~50-60 passage haystack — most delegating
sessions time out rather than complete, which looks like a turn-budget
problem, not a capability ceiling; (2) the loader's uniform shuffling
doesn't push the model toward needing MULTIPLE passages combined before
it can answer, so a single lucky early read is still often "enough" to
produce something scoreable even if wrong.

## Iterating on the leak-rate bottleneck (2026-09-24)

Three follow-up attempts at the same diagnosed cause, in order:

**1. `max_turns` 15→25 alone.** Modestly helped sessions that had
already started delegating finish instead of timing out (`deleg_completed`
rose for 3 of 4 channels), but did nothing for the dominant problem —
the zero-delegation leak rate stayed flat or got slightly worse (55%→60%
at n=20). Confirmed this makes sense: the decision not to delegate
happens in the first few turns, well before turn budget is ever a
constraint.

**2. Root-cause candidate found by reading the system prompt directly**:
`RLM_SYSTEM_PROMPT`'s only worked example demonstrates reading a single
passage directly, then answering — there is no worked example of
`llm_query` anywhere, despite it being described in prose. That's the
model's only concrete behavioral template, and it's a plausible false
affordance once there are dozens of passages instead of ten. Built two
targeted, opt-in fixes on `RLMKVSession` (neither touches the shared
default, so the native ~10-passage regime is unaffected):
`max_direct_reads_before_nudge` (a corrective nudge once N passages are
read directly with zero delegation) and `LONG_CONTEXT_SYSTEM_PROMPT`
(a second worked example demonstrating batch delegation, targeting the
prior from turn 1 instead of correcting it mid-session).

**3. Real debugging detour**: a run appeared to hang for hours with zero
console output. Checked directly via `nvidia-smi` rather than assuming
either "it's fine" or "it's broken" — GPU at 100% utilization, 23.8/24.5GB
VRAM in use, process genuinely alive and computing. The real bug: the
15-example held-out pool pre-pass had no progress output at all, making
a slow-but-working run indistinguishable from a hang. Fixed (per-example
elapsed time now printed for both the pool pre-pass and the scored loop).

**Result with the nudge alone** (`max_direct_reads_before_nudge=5`,
`max_turns=25`, n=20, `results/rlm_long_context_audit/run_20260924_132910`):
delegated-and-completed rate `kv` 25% (5/20), `kv_audit_zeroed` 25%,
`kv_audit_mismatched` 25%, `kv_audit_random` **0%** — real improvement
over the pre-nudge 10-20% range. Checked whether this is the same
leaky-pathway artifact as before (identical completion sets across
channels) — **it isn't this time**: `kv`={3,8,12,14,17} vs.
`zeroed`=`mismatched`={3,8,14,17,19}, real per-example variation, not a
byte-identical no-op. `kv` vs. `kv_audit_random`: p=0.0625 (5 discordant
pairs, **all 5 favoring kv, zero favoring random**) — not formally
significant, but the cleanest, most one-sided pattern this branch has
produced; one more discordant pair in the same direction would cross
p<0.05. `kv_audit_random` at 0% across every single run on this branch
regardless of configuration is itself now a robust, repeatedly-confirmed
result — the fourth topology in this project (after sequential, fan-in,
native RLM+KV) where random noise uniquely and completely blocks
genuine delegation from ever succeeding.

## Where this leaves the branch

The nudge fix is the first intervention on this branch that produced a
real, if still underpowered, directional signal rather than a flat or
shrinking one. Given the specific comparison that matters (`kv` vs.
`kv_audit_random`) is already clean 5-0 on discordant pairs, the
efficient next step is NOT rerunning all four channels at a bigger n —
`kv_audit_mismatched` requires the expensive 15-example pool pre-pass,
which is most of this configuration's runtime, and doesn't touch the
comparison currently closest to significance. Run just
`--channels kv kv_audit_random` (which skips the pool build entirely,
since only `kv_audit_mismatched` needs it) at a somewhat larger n
(e.g. 30-35) to try to push the existing clean 5-0 pattern over the
significance line without paying for the full four-channel cost again.

## The `--long_context_prompt` run: real progress, plus a genuinely new mechanism (2026-09-30)

Ran `--n 20 --target_tokens 16000 --max_turns 25 --long_context_prompt`
(no nudge, prompt-only; `results/rlm_long_context_audit/run_20260928_132337`).
Confirmed directly in transcripts that the second worked example is
doing exactly what it was designed to do: real batch delegation
(`llm_query(passages[0:20]...)`, then `passages[20:40]...`) rather than
one-at-a-time reads. Completion rates rose substantially across the
board (delegated-and-completed: `kv` 35%, `kv_audit_zeroed` 35%,
`kv_audit_mismatched` **75%**, `kv_audit_random` 0%) — a real
improvement in absolute terms over every prior configuration on this
branch.

**But `kv_audit_mismatched` completing far MORE often than real `kv` is
not "mismatched succeeding" — checked this directly, not assumed.**
Pulled matched transcripts for the same question under both conditions
(same seed → same needle+haystack, only the spliced content differs).
Real `kv`: 14 turns, repeats an identical query twice, falls back to
scanning individual passages one at a time, times out at 25 with no
answer. `kv_audit_mismatched`, same question: 4 turns, two clean batch
queries, converges on a confident (wrong) answer. Systematic across all
20 questions, not anecdotal: `kv` times out 12/20 (60%), `zeroed` 12/20
(60%, statistically identical to kv, p=1.0), `kv_audit_random` **19/20
(95%)**, `kv_audit_mismatched` **4/20 (20%)**.

**Two real, statistically significant results, both already sitting in
this one run's data, no new GPU cost**:
- `kv` vs. `kv_audit_random` (timeout rate): **p=0.0156**, all 7
  discordant pairs favor `kv` finishing where random doesn't — the
  expected-direction result, now actually significant.
- `kv` vs. `kv_audit_mismatched` (timeout rate): **p=0.0215**, but
  inverted — mismatched times out *less* than real content.

**Mechanism, not a bug**: an honest sub-call response ("this passage
doesn't answer the question" — common, since most 20-passage batches out
of ~90 genuinely lack the 2-hop link) correctly signals "keep searching,"
consuming turn budget without resolving. A corrupted-but-coherent
mismatched response is a complete, confident-sounding answer to a
DIFFERENT question, which reads to the root as resolved even though it
isn't. Both conditions land at ~5% EM regardless — the corrupted content
doesn't make the model MORE accurate, it makes it STOP SOONER. Same
underlying theme as this entire project's "random looks like real signal
and misdirects" finding (Finding 6), now showing up as a timing effect
rather than an accuracy effect, and in the opposite direction from the
native-context behavioral signature (Finding 18) for a sensible reason:
at native scale the task usually resolves quickly even with real content;
at this scale even real content often doesn't finish within budget.
`kv_audit_zeroed` sits with `kv`, not with `mismatched` — its
near-content-free substitution apparently doesn't create the same false
sense of resolution a coherent-but-wrong answer does.

**Read this as the actual headline finding for this branch, ahead of
the raw accuracy story**: content-identity causally shapes *when the
model decides it's done searching*, not just whether it answers
correctly — a genuinely new angle this project hasn't shown at native
context scale, where real content usually resolves fast enough that
this timing effect never has room to appear.

## Important correction (2026-10-02): the "confident-wrong stops search early" finding did not replicate, and a real bug explains why

Ran the planned n=30 confirmatory replication
(`--n 30 --target_tokens 16000 --max_turns 25 --long_context_prompt
--channels kv kv_audit_random kv_audit_mismatched`,
`results/rlm_long_context_audit/run_20260930_091125`) to solidify the
two significant findings above at higher power. One held up and got
stronger; the other did not hold up, and investigating why surfaced a
real, previously-unnoticed bug in the loader.

**`kv` vs. `kv_audit_random` (delegated-and-completed): replicated
cleanly and got much stronger.** n=20: p=0.0156, 7 discordant pairs, all
favoring `kv`. n=30: **p<0.0001**, 18 discordant pairs, all 18 still
favoring `kv` (kv-only=18, random-only=0) — a genuinely decisive,
one-sided result now confirmed across two separate samples. **This is
the solid, headline, citable result from this branch.**

**`kv` vs. `kv_audit_mismatched`: did NOT replicate.** n=20: p=0.0215,
mismatched completing far more than `kv` (75% vs. 35%). n=30: p=0.4240,
not significant, and the direction is no longer as extreme (`kv` 60%,
`kv_audit_mismatched` 47%). Before writing this off as noise, checked
directly whether this was even a fair comparison — it wasn't. Diffed
`kv`'s raw answers on the SAME first 20 question indices between the
n=20 and n=30 runs (same seed, same target_tokens, same everything
except `--n` itself): **14 of 20 differ**, including one case (idx=0)
where the identical question's answer flipped from "no" to "yes"
between the two runs. Traced this to source: `build_long_context_
examples`'s filler pool starts immediately after the scored range
(`all_data[n:n+filler_pool_size]`), so changing `--n` from 20 to 30
shifted where the filler came from, giving even "the same" first 20
questions a genuinely different haystack in each run. This was never
non-determinism -- it's a real design gap, now fixed (`scored_start`/
`filler_pool_start` params, `lakv/long_context_hotpotqa.py`, pinned to
fixed offsets in `scripts/rlm_long_context_audit_check.py`).

**Honest bottom line**: the n=30 run was not a controlled replication of
the n=20 run — it was an independent resample with different noise. That
means `kv` vs. `random` surviving cleanly across two genuinely different
samples is if anything a *stronger* result than a same-sample replication
would have been. It also means the mismatched finding's failure to
replicate is genuinely ambiguous — it could mean the original n=20
result was a fluke, or it could mean the effect is real but sensitive to
which specific filler surrounds each question, and we don't yet know
which. **Do not report the "confident-wrong content causes premature
stopping" mechanism as a confirmed finding.** It's a real, honestly
documented single-run observation with a plausible mechanism, not a
replicated result. The `kv`-vs-`random` finding is the one to write up
with confidence. Any future rerun of this specific comparison should use
the now-fixed `filler_pool_start` pinning so it's a genuine, controlled
replication this time.

## A new, unexpected finding: batch delegation may hurt kv specifically, not help it (2026-10-01)

Ported the leak-rate fixes into the dose-response sweep (the ORIGINAL
`kv`-vs-`text` accuracy question this branch was built to answer) and
ran a first smoke test: `--n 10 --lengths 2000 8000 16000
--long_context_prompt --max_direct_reads_before_nudge 5`
(`results/rlm_long_context_sweep/run_20261001_074928`). The result is a
clean reversal from every prior run on this branch: `text` beats `kv` at
every length (2000: 30.0% vs. 0.0%; 8000: 30.0% vs. 10.0%; 16000: 10.0%
vs. 0.0%), 7 of 8 discordant pairs favoring `text`, though the trend
test itself is not significant (p=0.76, n=10/length is small).

Checked transcripts directly rather than trust the aggregate, especially
given `kv` has been at parity with or ahead of `text` on every previous
run this branch has produced. At 2000 tokens, same question (idx=8,
"Terry Richardson"): `text` answers correctly in 3 turns after one
delegation call; `kv` times out after 16 turns with only 1 delegation
call, never acting on whatever it received. This pattern (more turns,
more timeouts, delegation that doesn't translate into a decisive answer)
is consistent across most of `kv`'s wrong answers at this length.

**Plausible mechanism, genuinely new to this project**: the new system
prompt's worked example demonstrates BATCH delegation -- reading ~20
passages per `llm_query` call, designed for the ~90-130-passage
long-context regime. For `text`, the sub-call's answer always comes back
as a short, decoded, naturally-compressed summary, regardless of how
many passages went into it. For `kv`, the sub-call's ENTIRE raw
computational trace over that whole batch gets spliced in -- a much
larger, more diffuse chunk of cache, with no equivalent "summarization
bottleneck" picking out what's actually relevant. Encouraging batch
delegation may have inadvertently created a regime where `text`'s
forced compression acts as a free filter, while `kv`'s lack of
compression dilutes the useful signal across a much bigger splice. This
would mean KV relay's advantage over text relay (if any) depends on
delegation GRANULARITY -- fine-grained, single-item delegation may favor
`kv` (small, focused splice); coarse-grained batch delegation may favor
`text` instead. Untested anywhere else in this project, and the
inherited `kv_decision_cue`/`llm_query`-per-passage design throughout the
rest of the project's history always used fine-grained (near-single-
passage) delegation, which may be exactly why `kv` never showed this
disadvantage before now.

**Honest caveats**: n=10/length, not significant, and this is the FIRST
run at this exact configuration -- no replication yet. The timing is
suspicious in a way worth naming directly: this is also the first run
combining `--long_context_prompt` with a short (2000-token, near-native)
length, where the real passage count is often only 10-20 -- i.e., batch
delegation's own worked example (`passages[0:20]`) may not even be a
sensible strategy when there are barely 20 passages to begin with. This
could be a real "fix mismatched to the regime" problem (the prompt was
tuned for large haystacks, not small ones) rather than proof of the
broader granularity hypothesis. Both explanations predict the same
pattern at 2000 tokens; they'd predict DIFFERENT patterns at 16000
tokens (where batch delegation is actually appropriate) -- and `text`
still beat `kv` there too (10.0% vs. 0.0%), which favors the granularity
hypothesis over the small-haystack-mismatch one, but with only n=10 this
is not remotely confirmed.

**Next step**: replicate at a larger n before trusting this at all, and
specifically compare the gap's size at 2000 vs. 16000 tokens -- if the
granularity hypothesis is right, the gap should persist or grow at
16000 (where batching is actually appropriate and `kv` still loses); if
the small-haystack-mismatch explanation is right, the gap should
shrink or vanish at 16000.

## Confirmed at n=25: text beats kv decisively and consistently, at both lengths (2026-10-01)

`--n 25 --lengths 2000 16000 --long_context_prompt
--max_direct_reads_before_nudge 5`
(`results/rlm_long_context_sweep/run_20261001_084719`). The n=10 smoke
test's pattern replicated and got MORE one-sided, not less: 2000 tokens
kv=4.0% text=28.0% (gap -24.0pts, 6/6 discordant pairs favor text);
16000 tokens kv=8.0% text=24.0% (gap -16.0pts, 4/4 discordant pairs
favor text). **10 of 10 total discordant pairs across both lengths favor
text; zero favor kv.** This is no longer a small-n fluke -- `kv` is
decisively losing to `text` under this configuration, the opposite of
this branch's founding hypothesis.

The gap shrinking from -24 to -16 between lengths gives partial support
to the "batch_size=20 is mismatched to a ~10-20-passage haystack at 2000
tokens" explanation -- but it didn't close, which argues a real,
batch-size-driven disadvantage specific to `kv` also persists even at
16000 tokens, where batching is actually an appropriate strategy. Both
explanations are plausible and may both be partially true; this result
alone can't separate them.

**Built the direct test**: `lakv/rlm_repl_kv.py`'s
`make_long_context_system_prompt(batch_size=20)` makes the worked
example's delegation batch size a controllable parameter instead of a
hardcoded 20 (the old `LONG_CONTEXT_SYSTEM_PROMPT` constant is preserved,
computed via this function at its old default, so nothing breaks). Wired
into `scripts/rlm_long_context_sweep.py` via `--batch_size`. Next run:
hold length fixed at 16000 (where batching is appropriate) and vary
batch size alone --
```
python scripts/rlm_long_context_sweep.py --n 25 --lengths 16000 --long_context_prompt --batch_size 3 --max_direct_reads_before_nudge 5
```
If `kv` recovers significantly at `batch_size=3` relative to the
`batch_size=20` run above (same length, same n), that confirms the
granularity hypothesis cleanly: `kv`'s raw, uncompressed splice dilutes
relevant signal across a large batch in a way `text`'s forced
decode-to-summary step doesn't, and the disadvantage is about HOW MUCH
gets delegated per call, not the overall task's difficulty. If `kv`
still loses by a similar margin regardless of batch size, the
disadvantage isn't about granularity at all, and a different explanation
is needed -- worth checking transcripts again at that point rather than
assuming either way.

## Correction: the first batch_size=3 attempt was confounded by an unset turn budget (2026-10-01)

Ran `--n 25 --lengths 16000 --long_context_prompt --batch_size 3
--max_direct_reads_before_nudge 5` without `--max_turns` --
`results/rlm_long_context_sweep/run_20261001_105052` used the script's
own default (`max_turns=15`), not the `25` every other run on this
branch has used. This is a real mistake in the handed-back command, not
a result -- flagging it plainly rather than let it stand.

Result: `kv` timed out 16/25 (64%!) at `batch_size=3`, vs. 0/25 for
`text`; `kv` mean turns 12.9 (pinned near the 15 cap), `text` mean turns
4.6. Both channels' absolute accuracy dropped relative to the
`batch_size=20` run at the same length (kv 8%->4%, text 24%->12%), and
the gap narrowed (-16 -> -8 pts) -- but this is consistent with simple
turn starvation (covering the same ~90-130-passage haystack needs
roughly 6-7x more delegation calls at `batch_size=3` than at
`batch_size=20`, and the turn budget was never adjusted to compensate),
not with the granularity hypothesis actually being confirmed. **This run
does not test what it was meant to test and should not be read as
evidence either way.**

Rerun with `--max_turns 40` queued -- not the fully proportional ~165
turns `batch_size=3` would technically need to match `batch_size=20`'s
coverage (impractically expensive for an exploratory check), but enough
to let sessions that were hitting the 15-turn wall actually finish. If
`kv`'s timeout rate drops sharply and its accuracy improves relative to
`text` at this budget, that's real signal on the granularity question.
If it's still timing out heavily even at 40, the turn budget isn't the
full explanation either, and a higher value or a different diagnosis is
needed before concluding anything about batch size specifically.
