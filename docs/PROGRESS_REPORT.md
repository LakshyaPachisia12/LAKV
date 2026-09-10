# LAKV Progress Report (plain-language, day by day)

This is a running log, in simple words, of what actually happened on this
project and when — based on real commit dates, not guesses. Newest entries
at the top. This file gets a new entry added roughly once a day going
forward, whenever there's real progress to record.

---

## 2026-09-10 — Chasing the "RLM" idea, and catching a citation gap

- Figured out what "RLM" (Recursive Language Models) actually means and how
  it connects to this project: RLM is about splitting a huge chunk of text
  into pieces, having little "assistant" AI calls read each piece, and only
  passing short written summaries back to the "boss" AI — instead of
  dumping everything on one AI at once. This project's whole idea (pass the
  model's internal notes instead of re-processing text) is a natural fit to
  try on top of that.
- While digging into this, found two other very recent research papers
  doing something close to this project's central "prove the notes really
  matter" test (swapping an agent's notes with garbage/wrong-question notes
  to see if answers get worse). Checked them carefully — good news, the
  paper draft already cites and correctly handles one of them. Made a
  small, precise fix: added the second paper as a citation, and tightened
  one sentence that was claiming a bit more credit than it should have.
  Not a big rewrite — a few sentences, done.
- Confirmed that nobody has yet tried the specific idea this project wants
  to test: having those RLM-style "mini assistants" hand back their
  internal notes instead of a written summary. Checked the original RLM
  paper and the closest related paper directly — both only ever pass back
  plain text.
- Built a small working first version of that idea:
  - A new piece of code that splits a question's 10 reading passages into
    2 groups, has a "mini assistant" read each group, and then combines
    their work into a final answer two different ways — the normal way
    (written summaries) and the new way (internal notes).
  - The one tricky part: when you glue two independently-computed sets of
    notes together, the second set doesn't automatically know it's now
    "further along" in the combined document — it still thinks it starts
    at the beginning. Fixed this by re-numbering its position correctly,
    reusing a formula this project already built and trusted for something
    else.
  - Tested that re-numbering trick on its own, with small fake data on a
    laptop (no expensive GPU needed) — all 6 checks passed cleanly.
  - Wrote a small script to run 5 real example questions through both
    versions and print the results side by side, so it's easy to eyeball
    whether the new version produces sane answers or garbage before doing
    a full, expensive test.
- Not yet run on the real GPU — that's the next step, and it's a quick,
  cheap one (5 questions, a few minutes).
- Searched beyond formal papers — Twitter/X, Hacker News, blog write-ups —
  for what people are actually saying about RLM. Found real interest (a
  well-known AI infra company built a product around it) but also real
  pushback ("neat idea, but not a new idea" was a top Hacker News comment).
  Also found one paper (RecursiveMAS, from UIUC/Stanford/NVIDIA/MIT) that
  sounded at first like it might already be our exact idea — read it
  closely instead of panicking, and confirmed it's meaningfully different:
  it passes one small summary number-list in a round-robin loop between
  agents and needs the model to be retrained to do it; ours passes the
  model's full internal notes in a split-then-combine shape, with no
  retraining needed. Related neighborhood, not the same idea.
- You hit a real bug trying to run the 5-question script: `ModuleNotFoundError:
  No module named 'run'`. Cause: running a script from inside the
  `scripts/` folder doesn't automatically let Python see files sitting in
  the main project folder. Fixed with the same trick already used
  elsewhere in this project's test files. Confirmed fixed (the script's
  `--help` now runs cleanly instead of crashing).
- Added an honest paragraph into the actual paper draft (not just this
  report) pointing at both RLM and RecursiveMAS, explaining clearly what
  this new prototype does and doesn't share with each of them — so if
  this idea comes up again during review, the paper already has language
  ready instead of scrambling.

- Ran the fixed 5-question script for real. Good news: nothing broke — no
  garbled output, no crashes, the "internal notes" version produced normal
  readable English. That was the biggest engineering risk, and it held up.
- Bad news, but a useful, cheap-to-fix bad news: both versions (notes and
  summaries) got most questions wrong for the *same* reason — each
  "assistant" was told to say "nothing relevant here" if its own half of
  the reading material didn't look useful, but these are multi-hop
  questions where a fact only *becomes* useful once connected to a
  different fact possibly sitting in the *other* assistant's half. One
  assistant was literally looking at the correct answer's name in the text
  and still said "doesn't fit," because in isolation it couldn't tell.
  Because both versions hit this same wall, this run couldn't yet tell us
  whether internal notes beat summaries — that signal was drowned out by a
  bigger, shared problem.
- Fixed it: rewrote the instruction given to each "assistant" so it no
  longer tries to judge relevance at all — it just neutrally reports facts
  (names, dates, roles) and leaves the judging to the boss agent, who
  actually sees everything combined. Cheap, one-paragraph change, no
  architecture change needed.

**Where things stand:** the RLM idea is a real, promising, not-yet-tried
extension — but see the honest rating in this file's companion discussion
for why it's being treated as a nice-to-have stretch goal, not something
racing to make the Oct 12 deadline. The script is fixed, the first real
run's failure mode is diagnosed and fixed, and it's ready for another
5-question run to see if that was the main blocker.

---

## 2026-09-09 — The big push: writing the whole paper, fixing real bugs

The busiest single day of this project. In plain terms:

- Confirmed the "swap the notes with garbage" test still works even when
  the notes are compressed (not just the full, uncompressed version) — a
  stronger, more realistic result than before, and the moment the team
  called "decision gate passed": the paper's central idea officially holds
  up under real-world conditions, not just an idealized best case.
- Wrote up basically the entire research paper in one day: all 6 sections,
  a full bibliography, and the two main figures.
- Caught and fixed 2 real factual mistakes in citations before they made it
  into the paper.
- Found and fixed a real accounting bug: the code had been reporting
  compression savings without ever actually shrinking anything in storage —
  like advertising a suitcase as "half the size" without ever folding the
  clothes. Fixed it properly (built real compression), and the corrected
  numbers landed almost exactly back where they'd originally been claimed —
  confirming the original claim was right all along, just not backed by
  real code until now.
- Confirmed the two headline results also hold on a second, different task
  (GSM8K math problems), not just the main one (HotpotQA reading
  comprehension) — makes the findings more trustworthy.
- Started testing a third AI model for comparison.
- Packaged up the actual submission files.
- Wrote down a few good future-research ideas without starting them, on
  purpose, to protect the deadline.

---

## 2026-09-08 — Confirming the "the notes really matter" test, reframing the paper

- Ran the "swap the notes with garbage" test on more examples (50 instead
  of a smaller test batch) and confirmed it's solid: when an agent gets
  fed made-up or wrong notes, its accuracy collapses in a clear, measurable
  pattern — real proof the shared notes carry real information, not just
  "having something there at all."
- Reframed the whole paper around one central idea, nicknamed
  "non-exchangeability": you can't assume any one slice of the model's
  internal notes (which layer, which precision, whose question) can be
  swapped for another slice without actually checking — and this project
  tested that assumption three different ways and found it false every
  time.

---

## 2026-09-07 — Kicking off the push toward a real research paper

The day this publication effort started. In plain terms:

- Added tools to properly measure whether a result is "really different"
  or just random noise (statistical significance testing).
- Got the project ready to test a second AI model (Mistral) so results
  aren't just about one specific model.
- Built the "swap the notes with garbage/wrong-question notes" test
  described above, from scratch.
- Tried a fancy trick ("rotate the numbers") to shrink the notes down to
  4-bit precision (very compressed) — it broke badly, producing nonsense
  output. Spent the day diagnosing exactly why: the trick was interfering
  with how the model keeps track of word order. Found the right fix
  (a different, more careful compression method) and it worked well —
  went from 0% correct answers to over half correct, matching the
  project's best existing method while compressing even harder and
  needing less setup.

---

## Before 2026-09-07 — The original project (compressed summary)

From late March through early September, this project was built up from
scratch: a 3-agent pipeline (one AI reads and reasons, a second
double-checks, a third writes the final answer) that passes its internal
notes between agents instead of re-reading text from scratch. Along the
way: fixed many real bugs (wrong token positions causing garbled output,
wrong stop conditions causing run-on generation, broken compression, a
scoring/anchor system that needed several rounds of fixes), added a way to
drop less-important parts of the notes to shrink them further, and added a
text-based version of the pipeline as a fair comparison baseline. This is
the foundation everything above was built on top of.
