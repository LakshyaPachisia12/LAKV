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

## Where this leaves the branch

Two independent experiments on this branch (dose-response, causal
audit) have now shown the same shape: promising-looking direction at
very small n, weak or dominated-by-a-confound at the n actually tried.
Neither is proof the underlying non-exchangeability claim is false at
long context — both are consistent with the scaffolding (turn budget,
passage-shuffling design) not yet being matched to this much larger
passage count, exactly the risk flagged before this branch's work
started ("the hand-tuned nudge thresholds were tuned for ~10 passages...
at 100+, they'll almost certainly need retuning"). That retuning hasn't
happened yet. Next step, if continuing: raise `max_turns` substantially
(e.g. 25-30) for the causal-audit script specifically, since the
current bottleneck is measurably timeouts-during-delegation, not
garbage output or a broken mechanism, before spending more GPU time at
the current turn budget.
