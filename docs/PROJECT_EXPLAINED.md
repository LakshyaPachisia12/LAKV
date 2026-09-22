# LAKV, Explained Simply

A study guide you can actually read before a meeting — short paragraphs,
one idea at a time, real numbers attached. The int-4 story (Part C) and the
causal audit (Part E) get the slowest, deepest treatment because those are
the two hardest pieces to hold in your head. Everything else moves faster
on purpose.

For the terse technical reference, see `CLAUDE.md` at the repo root.

*Updated 2026-09-18, branch `feat/research-extensions`.*

---

## If you read nothing else

- We let AI agents hand each other their internal "notes" (the **KV cache**)
  instead of re-writing everything as text.
- Squeezing those notes to a quarter of their size costs basically nothing.
- Squeezing them to a sixteenth of their size *broke everything* — we found
  out exactly why, and fixed it.
- We then proved, with real experiments, that the AI is actually *using*
  the specific notes it receives — not just performing better because it
  has *something* to hold. That proof is the heart of the whole project.
- Target: **NAACL 2027**, via ACL Rolling Review, submission deadline
  **October 12, 2026**.

---

## Part A — The basics

### A1. What is a KV cache?

When the AI reads text, it takes "notes" on every word — two per word, at
every one of its internal layers: a **Key** (a label, like "this is about
the butler's alibi") and a **Value** (the actual detail, "he said he was in
the kitchen at 9pm"). Later, to answer a question, it flips through these
notes instead of re-reading the original text.

**That stack of notes is the KV cache.** Building it costs computation
(called **prefill**). Writing the actual answer afterward, word by word, is
a separate, later step (called **decode**).

### A2. Why hand over the notes instead of a summary?

Our pipeline is three AI agents in a row: **Reasoner → Verifier →
Finalizer**. Normally, Reasoner writes a text summary and Verifier has to
*re-read it from scratch* — rebuilding its own notes for something Reasoner
already took notes on. Our idea: skip that. Hand Verifier the *real* notes
directly.

| | Text summary (`text_agent`) | Real notes (`A`) |
|---|---|---|
| Accuracy | 54.0% | 56.0% |
| Latency | 6.7s | 7.8s |

Surprising part: it's not clearly faster. **Writing the final answer
(decode) eats far more time than reading the input (prefill) ever did** —
so skipping the re-read step only saves a small slice of the total time.
And the accuracy gap isn't even statistically proven (p = 0.81 — a wash;
see Part D for what that means). Worth saying plainly in a meeting, not
hiding.

### A3. Squeezing the notes (compression)

Round every number on a card into fewer possible values:

| | 8-bit (256 values) | 4-bit (16 values) |
|---|---|---|
| Accuracy | 57.0% | **0.0%** |
| Cache size | 2× smaller | 4× smaller |

8-bit: free — statistically identical to the uncompressed version, on two
different models. 4-bit: total collapse. That collapse is the whole story
in Part C.

### A4. Throwing away whole layers

The AI doesn't read a word once — it processes it through **28 stacked
layers**, each layer keeping its own separate notes. What if we skip the
least useful layers entirely?

| | Layers only (`C`) | Layers + compression (`D`) |
|---|---|---|
| Accuracy | 43.0% | 50.0% |
| Cache size | 104 MB | 38 MB (≈4× smaller than full) |

Real cost this time — unlike compression, dropping layers loses real
accuracy. One surprising side-note: when a layer is skipped, leaving it
**blank** works *better* than filling it in with a neighboring layer's
real notes (10-16 points better). A neighbor's notes are confidently
*wrong* — a blank space just gets ignored.

---

## Part C — The 4-bit story, slowly

This is the one worth understanding properly, because it's the best piece
of real detective work in the project.

**The setup.** 8-bit compression (256 values per number) was free. So: push
further, to 4-bit (only 16 values per number). More savings — what could
go wrong?

**The break.** Everything. **0.0% accuracy.** The AI got stuck repeating
garbage forever — "0 0 0 0 0..." — and even got *slower* from looping.

### Step 1 — What's hiding inside a "Key" note

Before a Key note gets filed, the AI stamps a **position marker** onto it —
but not as a separate tag. It's baked directly into the same numbers, using
a trick called **RoPE**. Picture each pair of numbers as a clock hand,
rotated by an angle based on how far into the text that word appeared.

The consequence: you can't cleanly split "the content" from "the position"
in those numbers. They're fused into the same values.

### Step 2 — The fix that made things *worse*

A standard trick for rounding numbers harder: spin all of them evenly first
(a "rotation"), so no single number hogs all the precision, *then* round.
Normally this helps a lot. We tried it here — and got something stranger,
not better: literal fragments of Java code and random Chinese characters
showing up inside English trivia answers.

### Step 3 — The clue that cracked it

That's not what ordinary low-precision damage looks like. Ordinary damage
= the AI gets fuzzy, repeats itself, plays it safe and boring. What we saw
= the AI's internal "search system" pointed at a completely wrong shelf and
pulled out unrelated junk. **The specific shape of the garbage was the
clue** — this pointed at broken *position-matching*, not just blurry
precision.

### Step 4 — Catching the culprit red-handed

We repeated the rotation trick, but *only* on the Value notes (the content
half), leaving Key notes (the position-tangled half) untouched. Result:
back to the boring, ordinary kind of garbage — not the bizarre kind.

**Direct proof: Key was the fragile one. Value was fine all along.**

### Step 5 — The actual fix

Instead of one shared rounding "ruler" for an entire Key note, give **each
individual number-slot its own personal ruler**, sized to that slot's own
typical range.

> Analogy: don't weigh an ant and a bowling ball on the same bathroom
> scale — the ant reads "0." Give the ant a kitchen scale. Now both
> readings mean something.

### The result

| | Before | After (`B_int4_kivi`) |
|---|---|---|
| Accuracy | 0.0% | **52.0%** |
| Cache size | — | ~4× smaller than uncompressed |

That 52.0% statistically **ties our best, much more complicated method**
(the layer-dropping one from Part A4) — while needing none of its extra
setup, and compressing slightly harder. One of the strongest practical
results in the whole project, and it came from actually diagnosing the
failure instead of giving up on 4-bit.

*(We also tried improving the Value side further, twice — rotating it,
and quantizing it more finely. Neither helped. Two independent negative
results, both pointing at the same conclusion: fixing Key alone was the
whole story.)*

---

## Part D — A quick note on "statistically significant"

We compare two versions on the *same* questions and count disagreements.
The resulting **p-value** is roughly "the odds this gap is just luck."
**Below 0.05 = probably real. Below 0.0001 = essentially certain.** Above
0.05, we say so honestly rather than rounding a maybe into a yes.

Why bother? "50% vs 44%" on its own proves nothing — it could just be
which 100 questions got picked. The test is what turns a number into a
claim you can defend.

---

## Part E — The Causal Audit: why we even built this, and why it matters

### The uncomfortable question we had to rule out

Suppose Verifier does *well* after getting Reasoner's real notes. Two very
different explanations are possible:

1. Verifier is doing well because the notes contain real, specific,
   useful information.
2. Verifier is doing well simply because it has *something* to lean on —
   like a nervous student who calms down and performs better just from
   holding a cheat sheet, whether or not they actually read it.

**If explanation 2 were true, this entire project would be hollow.** Every
accuracy number we've reported would just be measuring a placebo effect,
not real information transfer — and there'd be no real reason to care about
compressing, shrinking, or carefully optimizing the content of something
that doesn't actually matter.

**So this is the one experiment the whole paper's credibility rests on.**
Everything else — compression ratios, layer-dropping trade-offs, the int-4
fix — only means something if we can first show the AI is genuinely reading
and using what we hand it.

### Why is proving this actually *useful*, beyond satisfying our own doubt?

- It tells us whether optimizing the *content* is worth doing at all — if
  fake content worked just as well, there'd be nothing to research.
- It's a reusable test, not a one-off trick — anyone building a multi-agent
  AI system could run the same experiment to check whether their own
  hand-off channel is doing real work or just a placebo.
- It turns "our system got 50% accuracy" from a bare number into a
  mechanistic claim: *specifically because of what was transmitted.*

### How we tested it

Secretly swap the real notes for one of three fakes, and see what happens:

1. **Zeroed** — completely blank notes.
2. **Random** — realistic-*looking* noise, meaningless underneath.
3. **Mismatched** — real notes, but from a totally different, unrelated
   question.

### What we found (config `A`, n=50)

| Condition | Accuracy |
|---|---|
| Real notes | 50.0% |
| Mismatched (wrong question) | 28.0% |
| Zeroed (blank) | 0.0% |
| Random (fake noise) | 0.0% |

Every gap here is statistically real (p < 0.0001 for real vs. zeroed/random;
p = 0.0127 for real vs. mismatched).

**One fun wrinkle:** random noise is *worse* than a blank page, even though
blank is "more empty." A blank note is obviously nothing, so the AI just
ignores it. Realistic-looking noise *looks* legitimate, so the AI trusts it
and gets actively misled. **A confident lie hurts more than an honest
blank.**

### What this proves, in one line

Real >> Mismatched >> Zeroed/Random is exactly the pattern you'd expect if
the AI is reading *specific* content — not just reacting to "something
being there." **This is the mechanism everything else in the project
stands on.**

### We didn't stop at one test — we tried to break it

We re-ran this same experiment everywhere we could: on compressed versions,
on a second model (Mistral), on a third model (Qwen3-8B), and on two
completely different team shapes (parallel agents instead of a chain; a
self-directed AI choosing its own delegation). Short version of what we
found:

- **Holds cleanly** on compressed configs, on Mistral (for most configs),
  and on both alternative team shapes — including two topologies where we
  now have the *complete* version of this test (real > mismatched > blank
  > noise, all proven).
- **Partially breaks** on Qwen3-8B (a newer, stronger model) and on
  Mistral specifically when layers are being dropped — the model seems
  able to partly shrug off losing or misdirecting content, staying
  defenseless only against pure noise. This is a real, honest nuance, not
  a flaw in the test — see the FAQ for how to talk about it.
- **A bonus discovery**, from the self-directed team-shape experiment: even
  when the AI never *sees* corrupted notes as readable text, its
  *behavior* — how long it keeps working, how often it gives up — tracks
  content quality in a way its final answer alone didn't fully reveal.

---

## Part F — Does it generalize across different AI models?

Same "drop ~29% of layers" operation, three different models:

| Model | What happened |
|---|---|
| Qwen2.5 (our main model) | Small cost, not yet statistically proven |
| Mistral-7B | **Severe collapse** — 30+ point drop, proven with near-certainty |
| Qwen3-8B (newer) | **No real cost** — proven statistically indistinguishable |

Same recipe, three different outcomes. **Layer redundancy depends on the
specific model's wiring, not on "how many layers you removed."** We also
checked whether a model's own internal "confidence" about which layers are
safe to drop predicts anything — it doesn't: Mistral's confidence signal
looks *more* decisive than Qwen's, yet Mistral is the one that collapses.

*(One model we tried and dropped: Phi-3.5-mini. It broke for a real,
pre-existing library limitation unrelated to our method — see the FAQ.)*

---

## Part G — Different team shapes

Everything above uses one straight-line chain: Reasoner → Verifier →
Finalizer. We tested two other shapes too:

- **Fan-in** — two agents read different passages in parallel, then one
  combines both. The causal-audit proof holds here too, and this is now the
  *cleanest* version of that proof in the whole project.
- **Self-directed delegation (RLM)** — instead of us scripting the
  structure, the AI decides on its own who to delegate to. The proof holds
  here too. One honest caveat: this particular setup isn't perfectly
  repeatable run-to-run (a numerical quirk in its long decision loop), so
  we measured that, reported it, and explained why our conclusions still
  hold anyway.

---

## Part H — An outside comparison: someone else's bridge

**C2C** is a published tool (not ours) that connects two *different* AI
models directly. We benchmarked it the same way as everything else: its
accuracy edge over running either model alone wasn't statistically proven,
and it cost twice the time. Its own causal-audit result has a genuinely
different shape from ours: it needs *some* real-looking content present,
but doesn't seem to care *which* question that content came from — a
shallower kind of "working" than our own method shows. A good honest
contrast, not a threat to our results.

---

## Part I — Full numbers, one place

| Config | Accuracy | F1 | Latency | Cache size |
|---|---|---|---|---|
| `single_agent` | 57.0% | 68.4% | 1.5s | — |
| `text_agent` | 54.0% | 68.3% | 6.7s | — |
| `A` (full KV relay) | 56.0% | 69.1% | 7.8s | 144 MB |
| `B_int8` | 57.0% | 70.5% | 8.0s | 72 MB |
| `B_int4` (broken) | 0.0% | 0.0% | 35.0s | 38 MB |
| `B_int4_kivi` (fixed) | 52.0% | 64.1% | — | 36 MB |
| `C` (layers only) | 43.0% | 57.8% | 10.6s | 104 MB |
| `D` (layers + compression) | 50.0% | 61.9% | 9.6s | 38 MB |
| `E` (offset correction) | 9.0% | 17.4% | 21.6s | ~40 MB |

**Headline p-values:** `A` vs `B_int8` p=1.0 (free) · `A` vs `text_agent`
p=0.81 (unproven) · `D` vs `E` p≈0.00003 (E fails) · `B_int4_kivi` vs `D`
p=0.86 (ties) · Mistral `A` vs `D` p=0.0001 (real collapse) · causal audit
real vs. fake, nearly everywhere: p<0.0001.

---

## Part J — Timeline

1. **Through Aug 31:** built the pipeline, pivoted to HotpotQA, got first
   baseline numbers.
2. **Sept 7:** started the publication push — added real statistical
   testing, a second model, fixed a generation bug.
3. **Mid-Sept:** root-caused and fixed the 4-bit collapse (Part C); built
   the causal audit (Part E); added a third model; tried and dropped a
   fourth (Phi-3.5).
4. **Mid-to-late Sept:** tested two new team shapes; benchmarked an
   external bridge (C2C); closed the remaining cross-model audit gaps.

---

## Part K — Why this is worth publishing

- **Timely:** multi-agent AI pipelines are a hot area, and today's normal
  practice (re-reading text between agents) is genuinely wasteful.
- **Rigorous:** every real claim is backed by a significance test — this
  already caught us over-claiming once, and we reported that instead of
  hiding it.
- **Mechanistic:** the causal audit is a reusable method on its own, not
  just a benchmark number.
- **Generalized:** tested across three models and three team shapes, with
  honest reporting of where it breaks, not just where it works.

---

## Part L — Conference target

| | |
|---|---|
| Target | **NAACL 2027** |
| Review system | ACL Rolling Review (ARR) |
| **Submission deadline** | **October 12, 2026** |
| Commit-to-NAACL deadline | December 23, 2026 |
| Conference | June 1-5, 2027, San Francisco |
| Fallback | COLING 2027 (same submission cycle) |

---

## Part M — Say this out loud

> "We let AI agents hand off their internal notes instead of re-writing
> everything as text. Compressing those notes to a quarter of their size
> costs almost nothing. We found and fixed a total collapse in our most
> aggressive compression method — from 0% to 52% accuracy, by tracing the
> failure to exactly how position information gets encoded and fixing the
> rounding scheme around it. Most importantly, we proved — not assumed —
> that the AI is genuinely using the specific content we hand it, not just
> benefiting from having *something* to lean on, and confirmed that across
> three models and three different team structures. We're targeting NAACL
> 2027, submission deadline October 12."

---

## Part N — Don't overclaim these

- `A` vs `text_agent`'s accuracy/latency edge (p = 0.81 — a wash)
- `A` vs `D` on Qwen alone (p = 0.34)
- The self-directed setup's KV-vs-text edge (not proven)
- The C2C bridge beating either model alone (not proven)
- The "does a model's confidence predict layer-drop safety" correlation
  (only 3 data points so far)
- Heterogeneous relay — one model's agent handing off to a *different*
  model's agent in one live pipeline — not attempted yet

---

## Part O — FAQ

**"Is this reproducible?"**
Mostly yes — the main pipeline gives bit-identical results on reruns. The
one documented exception is the self-directed (RLM) setup, and we measured
and reported that ourselves rather than let it slide.

**"Why only three models?"**
A fourth (Phi-3.5) was tried and genuinely broke for reasons outside our
control (a known limitation in how it handles long conversations) — we
confirmed this with a test using zero of our own code. Three models already
proves the main point: the same operation behaves very differently across
architectures.

**"What's the single strongest result?"**
The causal audit's real > mismatched > blank/noise ladder — replicated
10+ times across models, compressions, and team shapes.

**"What's the weakest part?"**
The original "KV relay is faster and just as accurate as text" pitch isn't
proven yet at current sample sizes. We say this openly rather than bury it.

**"Why is random worse than a blank cache?"**
A blank note is obviously empty, so it gets ignored. Realistic-looking
noise looks legitimate, so the AI trusts it and gets misled — a confident
lie is worse than an honest blank. We saw this same pattern twice, on two
completely different mechanisms.

**"What would you do with two more weeks?"**
Get a fourth model family running properly, write up the RLM
reproducibility finding on its own, and attempt handing a cache from one
model's agent to a *different* model's agent within a single pipeline.
