# LAKV — Complete Project Overview (for guide/panel presentation)

This document explains the whole project in plain language: what we set
out to do, what we actually did, what we found, and what we're honest
about not knowing yet. It's meant to be read start to finish by someone
who hasn't been following the day-to-day work, and to be a reference you
can come back to later without having to dig through code or logs.

---

## 1. The one-sentence version

When multiple AI language models work together on a task, passing
"notes" from one to the next, is it better to pass those notes as plain
text, or as the model's own internal memory state directly? We built a
system to test this rigorously, found real, carefully-verified answers,
and used those answers to catch several existing techniques making a
risky assumption without realizing it.

---

## 2. The problem, explained simply

Modern AI assistants increasingly work in teams: one model reasons about
a problem, a second model checks that reasoning, a third model writes
the final answer. Each model has to read everything the previous one
produced before it can start.

Every transformer-based language model, while it's "thinking" (generating
text token by token), builds up an internal memory of everything it has
read and written so far. This internal memory is called the **KV cache**
(key-value cache). Normally this memory is private to one model's one
conversation and gets thrown away afterward.

The idea this project tests: **what if, instead of the first model
writing out its answer as text for the second model to re-read from
scratch, we handed the second model the first model's KV cache
directly** — like handing someone your actual notebook instead of
dictating your notes to them over the phone? In principle this should be
faster (the second model doesn't have to re-process the same
information) and might even preserve more nuance than compressing
everything down into words.

This sounds promising, and several published research systems already
do versions of it. But it rests on an assumption nobody had directly
tested: **that this transferred memory is safe to shrink, reorganize, or
even guess at, without checking whether the specific content inside it
still matters.** This project's central contribution is building a
direct test for that assumption — and the answer turns out to be more
interesting, and more conditional, than a simple yes or no.

---

## 3. How we test it — the setup

**The models.** We use real, publicly available 7-8 billion parameter
language models (mainly Qwen2.5-7B-Instruct, plus Mistral-7B and
Qwen3-8B to check whether findings generalize across different model
architectures, not just one).

**The task.** Mainly HotpotQA, a well-established multi-hop
question-answering benchmark: each question requires combining
information from more than one of ten provided passages to answer
correctly (e.g. "What nationality was the director of the film that
won X award?" requires finding the film, then the director, then their
nationality). This forces genuine multi-step reasoning, which is exactly
the kind of task where relaying information between agents matters.

**The baseline architecture.** A three-agent pipeline: a *Reasoner*
reads the passages and thinks through the question, a *Verifier* checks
that reasoning, and a *Finalizer* writes the final answer. We compare:
- **`text_agent`**: each agent reads the previous agent's decoded text
  output, like a normal multi-agent chatbot pipeline.
- **`A`** and other KV-relay configs: each agent instead receives the
  previous agent's actual KV cache and continues directly from it,
  never re-reading anything as text.

**How we measure correctness.** Exact-match accuracy and F1 score
against the known correct answer, same metrics standard in QA research.

**How we measure whether a result is real.** Every comparison in this
project uses a paired statistical significance test (McNemar's test) and
a confidence interval, not just "config A got 56% and config B got 50%,
so A is better." This matters because with 50-100 test questions, a
difference of a few percentage points can easily be random noise. We
also compute *statistical power* — how much data would be needed to
trust a "no difference found" result — so we never confuse "we didn't
find a difference" with "we proved there is no difference."

---

## 4. The journey — what we actually did, in order

### 4.1 — Building the baseline and making the relayed memory smaller

First we confirmed the basic premise works: relaying KV cache directly
(`A`) matches the accuracy of a single model doing everything itself,
and is somewhat faster than the plain-text relay (`text_agent`) because
it skips re-processing the same content.

Then we asked: can we shrink the relayed memory to save bandwidth,
without losing accuracy? Two techniques:

- **Quantization** — storing each number in the cache with fewer bits of
  precision (like saving a photo as a slightly lower-quality JPEG
  instead of a raw file). Using 8-bit instead of full precision
  (`B_int8`) turned out to be essentially **free** — statistically
  indistinguishable from uncompressed, confirmed on two different models.
  Using 4 bits (`B_int4`) **completely broke** the system (0% accuracy,
  pure gibberish output).
- **Layer selection** — transformer models have many internal "layers,"
  and we tested whether some layers' memory could just be dropped
  entirely (`C`, `D`). This worked, but cost real accuracy (dropping the
  cache for \~29% of layers cost roughly 13 accuracy points) — not free,
  but a real, usable trade-off for extra compression.

**Diagnosing and fixing the 4-bit collapse.** This is one of the
project's cleanest pieces of detective work. We found that the reason
4-bit quantization broke the system wasn't "4 bits is just too
imprecise" — it was a specific interaction with *how* position
information is encoded in the model (RoPE). We confirmed this with two
targeted tests, then built a fix (`B_int4_kivi`, inspired by a published
technique called KIVI) that treats position-sensitive and
position-insensitive parts of the cache differently. Result: accuracy
recovered from 0% all the way to matching our best layer-selection
config, while being **more compressed than any other working config
in the project.**

### 4.2 — The central question: does the content actually matter?

This is the project's headline contribution. Every efficiency technique
above *assumes* the relayed memory can be safely approximated. We tested
this assumption directly instead of assuming it, using a method
called a **causal audit**: before the second agent uses the relayed
memory, we secretly swap it out for one of three things:

1. **Zeroed** — blank, empty memory (a sanity-check floor).
2. **Random** — artificial noise shaped to *look* statistically like
   real memory, but containing no real information.
3. **Mismatched** — another question's *real*, genuine memory — i.e.,
   real, well-formed content, just about the wrong topic.

If the specific content doesn't actually matter — if the model is just
reacting to "there is *something* in my memory" — accuracy should stay
roughly the same across all four conditions (real, zeroed, random,
mismatched). If the content specifically matters, accuracy should drop
in a clear, ordered way: **real > mismatched > zeroed/random.**

**This is exactly what we found, decisively, every time we tested it.**
Real content always wins; mismatched (wrong-but-real) content is better
than nothing; pure noise is worst of all — interestingly, often *worse*
than literally nothing, because realistic-looking noise actively
misleads the model's attention, while blank memory is just quietly
ignored. We confirmed this pattern across five different relay
configurations, multiple models, two different tasks (HotpotQA and
GSM8K math problems), and even on compressed/quantized caches — not
just the uncompressed baseline. This is the single most load-bearing
result in the whole project: it proves that every accuracy number
reported elsewhere reflects *preserved information*, not an accident of
having a non-empty memory.

### 4.3 — Does this hold on other models, not just one?

A result that only works on one specific AI model isn't very
trustworthy. We repeated the key experiments on **Mistral-7B** and
**Qwen3-8B**, two architecturally different models. The headline
causal-audit finding (content really matters) held up on both. But one
specific technique — dropping layers for compression — turned out to be
**architecture-dependent**: it cost Mistral roughly 30 accuracy points
(severe), while costing Qwen2.5 much less and costing Qwen3-8B
essentially nothing. This matters practically: a technique that looks
safe on one model family cannot be assumed safe on another without
testing — exactly the kind of hidden assumption this whole project is
built to catch.

We also found something subtle and a bit unsettling: the internal
"confidence" signal the system uses to decide which layers are safe to
drop was *more* confident on Mistral (the model that collapsed harder),
not less. In other words, the calibration signal doesn't predict safety
— and gets it backwards if anything.

### 4.4 — Does this hold in a different team structure, not just a chain?

So far, every test used a strict assembly-line structure (Reasoner →
Verifier → Finalizer). We built a second, structurally different setup:
a **fan-in** topology, where two independent "child" agents each read
half the passages in parallel, and a single "parent" agent combines both
of their memories to answer. This tests whether our findings are an
artifact of the assembly-line structure specifically.

Result: the same real > mismatched > zeroed/random pattern held — in
fact, this was the **cleanest, most statistically decisive version of
the result anywhere in the project.**

### 4.5 — The newest frontier: letting the model decide for itself

Every setup so far has the "who talks to whom and when" structure
decided in advance by us, the system designers. In late 2026, a new
technique called **Recursive Language Models (RLM)** became a major
talking point in the AI research community: instead of a fixed
structure, the model itself writes and runs its own small program,
deciding on the fly what to delegate to a helper call and when, inside
a live coding sandbox. The original technique always hands the helper's
answer back as plain decoded text.

We built something nobody else had published: the same self-directed
delegation idea, but with the helper's **KV cache** spliced in directly
instead of its decoded text — a genuine, novel combination.

At the native scale of our HotpotQA questions, the result was a clean
**draw**: KV relay and text relay were statistically indistinguishable
on accuracy. But we found KV relay was consistently a bit *slower*, and
traced this to the model needing more exploratory steps to work with
spliced memory than with plain readable text — the same mechanism
(needing more effort to use less "legible" content) that we later found
explains a much bigger effect at long context (next section). The causal
audit still held cleanly here too — even in this self-directed setting,
real content beat corrupted content, resolving one particular comparison
that raw accuracy alone couldn't settle, using how *often the model
finished within its turn budget* as a second, independent signal.

### 4.6 — Testing RLM+KV where it's actually supposed to matter

Here's an important piece of self-correction in our own process. The
whole motivation for the RLM technique, in the paper that introduced it,
is handling contexts *much longer* than a model's normal window —
specifically contexts far larger than our native \~2,000-3,000-token
HotpotQA questions. We realized our main experiment never actually
tested that regime, so we built a separate "long-context" test: the same
real HotpotQA passages, padded out with a large amount of irrelevant
filler text, up to 16,000 tokens.

**Result: at long context, decoded text decisively beats KV-cache
relay** — the opposite of the native-scale tie. We didn't stop at the
headline number; we dug into *why*:

- We first suspected the KV channel was failing because the helper
  delegations were too "chunky" (reading too many passages per
  delegation call at once, diluting the useful signal). We tested
  smaller delegation chunks directly. **This hypothesis was wrong** — KV
  relay's own accuracy stayed completely flat regardless of chunk size.
- We then found the real mechanism, for free, by reusing data we'd
  already collected: KV relay times out (fails to finish within its
  turn budget) *far* more often than text relay, at every configuration
  we tested, and this gets *worse*, not better, with smaller delegation
  chunks (because smaller chunks mean more total delegation rounds
  needed, and KV relay was already too slow to keep up). This is a very
  strong, statistically overwhelming result.
- We then asked the obvious follow-up: is this just a "not enough time"
  problem? We doubled the turn budget. **It didn't help at all** — the
  timeout rate stayed the same. Reading the actual failed attempts by
  hand, we found a clean, specific pattern: sessions either resolve
  quickly, or get stuck in one of two distinct failure modes (giving up
  on delegation entirely and resorting to reading passages one at a
  time, or endlessly repeating the exact same delegation request) —
  in one case, the system's own loop-detection safeguard literally told
  the model it was repeating itself, and it repeated itself anyway.

This is a genuinely new, carefully-verified finding: KV-cache relay's
efficiency promise does not automatically extend to the regime where
Recursive Language Models are supposed to shine, and we can now explain
precisely why, not just that it happens.

---

## 5. The results, summarized plainly

| Question | Answer |
|---|---|
| Does relaying KV cache work at all, accuracy-wise? | Yes — matches single-model accuracy, at a real (if modest) speed advantage over plain text. |
| Can you shrink it with 8-bit compression for free? | Yes, confirmed on two models — no measurable accuracy cost. |
| Can you shrink it with naive 4-bit compression? | No — total collapse, but we diagnosed and fixed the root cause. |
| Can you drop whole transformer layers to compress further? | Yes, but it costs real accuracy, and *how much* it costs depends heavily on which model you use — not a fixed, portable cost. |
| Does the specific content inside the relayed memory actually matter, or is "having some memory" enough? | It specifically matters — proven directly, not assumed, across five relay methods, three model families, two tasks, and three different team structures. |
| Does a published "offset correction" fix for memory drift work? | No — we found and fixed four real bugs trying to make it work fairly, and it still failed badly. A clean, honest negative result. |
| Does letting the model decide delegation for itself (RLM-style) change any of this? | The core "content matters" finding still holds. But KV relay does **not** show any accuracy or speed advantage over plain text in this setting, and actually loses decisively at long context, for a mechanism we've now identified and verified. |

---

## 6. What we're proud of — the main contributions

1. **A rigorous, repeatable method (the causal audit)** for testing
   whether any "relay the AI's internal memory" technique is really
   preserving meaningful information, rather than just assuming it —
   applicable well beyond this specific project.
2. **A real, root-caused bug diagnosis and fix** for 4-bit quantization
   collapse, going from total failure to matching our best
   compression-plus-accuracy trade-off, with the smallest memory
   footprint of any working configuration.
3. **Proof that compression safety is architecture-dependent**, not a
   property you can test once on one model and trust everywhere —
   including the counterintuitive finding that a calibration system's
   own confidence can be actively misleading.
4. **A documented, honest negative result** (the offset-correction
   technique) — showing our rigor applies even when the answer isn't
   what we hoped for.
5. **The first test, to our knowledge, of whether this kind of memory
   relay still matters once the "who delegates to whom" decision is
   made by the model itself**, rather than fixed in advance by the
   system designer — including a genuinely new combination (self-directed
   delegation plus real memory-cache relay) nobody else had built.
6. **A fully diagnosed explanation for why KV relay underperforms at
   long context** — not just an observed gap, but a specific,
   statistically confirmed mechanism (timeout/turn-budget behavior),
   with the obvious alternative explanation (not enough time) directly
   tested and ruled out.

---

## 7. What we're honest about — limitations

We've tried to be disciplined about not overclaiming, including flagging
our own uncertainty clearly rather than smoothing it over:

- Some comparisons (e.g., whether layer-dropping costs Qwen2.5 real
  accuracy) are not yet statistically significant at our current sample
  sizes — we report this explicitly rather than imply they're settled.
- A cross-architecture pattern we noticed (how "confident" a model's
  calibration is vs. how much accuracy it loses) is based on only three
  models — too few to treat as a general law yet.
- Everything runs on a single GPU in one program; we account for memory
  size accurately, but don't measure real network-transfer speedups a
  live multi-machine deployment would have.
- One specific aspect of our longest-context investigation (whether
  "confidently wrong" content makes the model stop searching too early)
  did not replicate on a repeat run — we reported that honestly as an
  open question rather than keep the more exciting original claim.
- The core causal-audit *method* (testing with zeroed/random/mismatched
  substitution) was proposed by another research group before us; our
  contribution is extending it to new settings (new team structures,
  self-directed delegation, cross-architecture checks) they hadn't
  tested, not inventing the method itself. We say this plainly rather
  than let it look otherwise.

---

## 8. Where things stand now

The core experimental work is complete and has been written up. The
remaining work is mostly about presentation and polish: finalizing the
write-up, making sure it fits within length requirements, and a final
proofreading/consistency pass — not open research questions. If asked
"what would you do with more time," the honest answer is: test more
model families to make the cross-architecture patterns statistically
stronger, and investigate *why* KV relay gets stuck in the specific
failure loops we found at long context, rather than only confirming
that it does.
