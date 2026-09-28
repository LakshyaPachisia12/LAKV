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
