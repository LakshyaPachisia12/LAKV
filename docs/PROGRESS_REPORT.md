# LAKV Progress Report (plain-language, day by day)

This is a running log, in simple words, of what actually happened on this
project and when — based on real commit dates, not guesses. Newest entries
at the top. This file gets a new entry added roughly once a day going
forward, whenever there's real progress to record.

---

## 2026-09-18 (later) — Read the actual competing papers directly, and built a new measurement the log's summaries alone wouldn't have caught

- Asked for a proper outside-in audit of the whole project: what's
  actually novel, what's weak, and what to build next. Instead of just
  re-reading this project's own notes, went and read the real papers
  directly.
- Found something important: one of the papers already cited runs this
  project's own "swap the notes with wrong/blank/scrambled ones" test,
  on three other published systems instead of on our own setups, and
  finds the exact same kind of pattern this project independently found
  — whether the wrong notes matter depends on which system you're
  looking at. Good news: the paper draft already cites and correctly
  sets itself apart from that work; added one paragraph making the
  overlap between the two findings explicit instead of leaving it
  unsaid.
- Built something new in response: several of this project's own "real
  notes clearly beat wrong notes" comparisons aren't proven strongly
  enough yet by plain right/wrong scoring, at the current amount of
  data. Earlier this week, one specific setup (the free-form
  delegation one) got past that exact problem using a different kind of
  signal — not "was the final answer right," but "did the AI keep
  hesitating and re-checking, or commit confidently." That signal
  resolved a comparison plain right/wrong scoring couldn't. Today,
  built the general version of that idea so it works for the other two
  setups too, not just that one — measuring how CONFIDENT the AI's own
  next-word predictions are while it writes its answer, which can be
  computed the same way regardless of which setup is being tested.
- Checked it with real tests before considering it done (peaked vs.
  spread-out fake predictions give the expected high/low readings,
  averaging multiple answers works correctly, edge cases don't crash)
  — 6 new tests, all passing, full project test suite still 121/121
  with zero breakage from the changes needed to wire this in.
- Wired it into the two check scripts as an optional flag so it can
  actually be turned on and used on the next real run; deliberately left
  it out of the main three-agent pipeline's bigger, older command-line
  setup for now, since that setup's own comparisons are already
  well-proven and don't need this as urgently — noted clearly so it
  isn't forgotten later.
- Bottom line, asked honestly: this project is real and defensible, but
  its central "wrong notes hurt" story is no longer uniquely ours to
  claim — another paper got there first, on different systems. What's
  still genuinely unclaimed is testing that same idea across different
  SHAPES of multi-AI setups (fixed step-by-step vs. split-the-work vs.
  free-form) while holding the AI model and test itself fixed, plus this
  new confidence-based way of measuring it. That's the real, sharper
  story to build the rest of the paper around going forward.

---

## 2026-09-18 — Fixed a real flakiness bug, and built a check for a bias risk another paper flagged

- Earlier this week the free-form code-writing delegation setup was found
  to give slightly different answers on reruns, even with nothing changed
  — about 3 in 10 answers differed run to run, unlike the rest of this
  project, which reproduces exactly every time. Tried a real fix today
  (forcing the underlying math library into a strict, reproducible mode)
  rather than just living with the caveat.
- Checked it, not just assumed it worked: reran a slice of questions and
  compared word-for-word against an old run from before the fix,
  including the exact questions that used to come out different. All of
  them matched exactly this time. Real evidence the fix works, not a
  lucky sample.
- Also built a second check, prompted by a very similar outside paper
  that found its own version of this project's split-the-work setup can
  be biased by which order the pieces get combined in (not by their
  content, just their order — a real flaw in a similar published method).
  Added a way to flip that order in our own setup and compare, to test
  whether the same bias risk applies here. Built and tested the code
  itself; the actual comparison run is still to be done.
- Both pieces of work are ready to run at full scale; the determinism fix
  has only been checked on a smaller slice so far, and the order-bias
  check hasn't been run on the real model yet at all.

---

## 2026-09-17 — The other delegation setup's real-content test came back, and it's the cleanest one yet

- Ran the same "real notes vs. wrong notes vs. blank notes vs. scrambled
  notes" test on the other, more structured delegation setup from
  earlier this week (the split-the-work version, not the free-form
  code-writing one). This version doesn't let the AI peek at the
  original material directly on its own the way the other one did, so
  there's no escape hatch this time — and it shows in the numbers.
- Real, clean, decisive result — the best version of this test in the
  whole project so far. Real notes clearly and provably beat wrong
  notes, blank notes, and scrambled notes, all three, with the real-vs-
  wrong-notes comparison landing more decisively here than anywhere
  else this project has tested it.
- One small, honest nuance: telling "wrong notes" apart from "totally
  blank/scrambled notes" isn't quite proven by the strict pass/fail
  count (too few examples flip either way to say so confidently), but
  the softer, partial-credit scoring does point that way clearly. Said
  both things plainly instead of picking whichever sounded better.
- Double-checked by reading the actual raw answers, not just the score:
  blank notes produce garbled nonsense in about half the answers,
  scrambled notes produce garbled nonsense in nearly all of them, wrong-
  but-real notes are almost never garbled at all — a clean, sensible
  story that lines up with the numbers perfectly.
- Put together with the last two weeks of work, all three versions of
  this test — the fixed step-by-step handoff, the split-the-work
  version, and the free-form code-writing version — now all show real
  notes matter, properly proven with real statistics, not just assumed.
  That closes out every remaining gap of this kind in the project.
- Asked directly what else could make this deeper and more genuinely
  ours, rather than just adding more of the same. Went back through
  already-saved results from the free-form delegation setup — no new
  computer time needed — and found something real: it's not just
  whether the AI gets the final answer right that shows the notes
  matter, it's also how it BEHAVES along the way. With real notes, it
  wraps up quickly and confidently most of the time; with wrong notes,
  it's noticeably more hesitant; with blank or scrambled notes, it
  mostly runs out of time still searching, rarely committing to an
  answer at all. Checked this properly with real statistics, not just
  eyeballing it — and it's solid, including the exact comparison
  (wrong notes vs. blank/scrambled notes) that the plain right-or-wrong
  scoring couldn't quite prove. A genuinely new, complementary angle on
  the same finding, found for free by looking more carefully at data
  already sitting there.

---

## 2026-09-16 — The real test on the AI-bridge came back, clean and decisive

- Ran the proper "does this bridge actually care what it's told, or just
  that it's told something" test at a real sample size (50 questions, not
  10). The result is clean enough that there's no real doubt left, unlike
  most of the noisier checks earlier this week.
- Turning the bridge's signal into blank or into random static caused a
  total collapse — every single answer went from often-right to always
  wrong. So the bridge is definitely doing something real and necessary,
  not just decoration. Interesting bonus: the "random static" failures
  looked exactly like a very specific, unusual kind of garbage this
  project has seen before in a completely different part of the work
  (foreign characters and code fragments mixed in) — the same signature
  showing up twice, in two unrelated places, is a good sign it's a real,
  general pattern and not a fluke either time.
- But swapping in a completely different, wrong question's notes through
  the same bridge barely changed anything — statistically, not at all.
  So: the bridge clearly needs *some* real notes to work, but doesn't
  seem to care much *which* notes it gets. That's a genuinely different,
  and more precise, answer than the noisy 10-question version gave
  earlier this week — that one looked similar but wasn't confident enough
  to say so properly.
- Wrote all of this into the actual paper now, not just the internal
  notes — both halves (the plain accuracy check and this real content
  test) together, since the second one is what actually explains and
  justifies the first one's otherwise-unremarkable number.
- Also gave two honest, direct answers today that were hard to hear but
  needed saying plainly: a considered, reasoned rating on how likely
  this paper is to get accepted at the conferences being considered
  (with a specific, concrete fix identified — the "faster" pitch isn't
  actually backed by the paper's own numbers, and that's a bigger risk
  than any of the smaller add-on ideas discussed this week), and a
  similarly honest rating on continuing the earlier delegation/sub-agent
  detour, which was rated low and explained clearly why, while still
  helping move it forward since it's a personal call worth respecting
  even after an honest downside is laid out.
- Went back into that same delegation/sub-agent detour anyway, since you
  wanted to keep pushing on it despite the honest downsides. First real
  check came back with the notes-passing version clearly worse than the
  plain-text version — not just wrong answers, but genuinely broken,
  nonsense output in several cases. Read the actual broken output
  carefully instead of trusting the summary numbers, found a real,
  precise, fixable cause (the equivalent of stitching a photo into an
  album at the wrong page number, consistently off by however long each
  helper's own instructions were), and fixed it directly. Also caught
  that the existing safety-check test for this exact bug was written in
  a way that could never have caught it, and fixed that too.
- Reran the same check after the fix. The headline pass/fail number
  looked unchanged at first glance, which could easily be mistaken for
  "the fix did nothing" — but reading the actual output side by side
  told a different, real story: every single one of the previously
  broken, nonsense responses is now coherent and sensible. The fix
  worked exactly as diagnosed. The underlying task is still genuinely
  hard and still isn't being answered correctly yet, but that's now
  honestly a separate, harder problem, not this bug anymore. Also
  noticed something new and interesting along the way: in one case, a
  helper handed back the exact right answer, and the boss AI just kept
  investigating other things instead of using it until it ran out of
  time — a real, different, worth-remembering issue, not something to
  chase today. Wrote the whole honest before-and-after into the actual
  paper.
- You wanted to keep pushing on this specific piece today rather than
  stop there, so we did — properly. Found a real, specific reason the
  notes-passing version might not be noticing when a helper already
  handed back the right answer: the plain-text version literally sees
  the helper's written words each time, but the notes-passing version
  only gets a blank cue with no words at all — a real, structural gap
  between the two, not a guess. Added a small, careful nudge that
  doesn't cheat by secretly showing the actual words (that would defeat
  the whole point of testing notes vs. words) — it only prompts "you may
  have enough now, consider answering," leaving the real judgment
  entirely up to whatever the notes themselves convey.
- Small first check (10 questions) came back mixed — a little better
  overall, but not a clean win, so rather than declare victory or
  scrap it, agreed to test properly at a bigger size before deciding
  anything for real.
- Ran it at 25 questions. Real, clean result this time: the notes-passing
  version got 6 right versus the plain-text version's 2, and did so by
  strictly building on top of the plain-text version's own correct
  answers rather than trading wins for losses. Double-checked there was
  no hidden broken/garbled output hiding behind the better numbers at
  this bigger size — there wasn't. One honest cost that came with it: the
  notes-passing version also ran out of its turn budget noticeably more
  often, suggesting the new nudge sometimes pushes it to keep checking
  longer instead of answering sooner. Wrote the complete, honest
  before-and-after — the real win and the real cost together — into the
  actual paper.
- Did a proper, honest deep-dive today to check two things: is any of
  this RLM work actually going somewhere, and — importantly — is anyone
  else already doing exactly this, which would make it pointless to keep
  going? Checked carefully rather than assuming: the pieces individually
  (the delegating-AI mechanism, passing raw notes instead of text,
  testing whether real content matters) all already exist separately in
  outside work, but the specific combination of all three together, with
  a real "does it actually matter" test applied to it, doesn't appear to
  exist anywhere yet. Real, honest answer: yes, still worth doing.
- Ran the real content-matters test (the same kind of test proven
  throughout this whole project) on this specific setup for the first
  time, at a real size (50 questions). Result: genuinely exciting and
  clean. Feeding the AI blank or scrambled notes instead of real ones
  made it noticeably and provably worse — a real, statistically solid
  result, not a coincidence. This is the same core finding already
  proven twice elsewhere in this project, now shown a third time in a
  completely different, more free-form setup.
- One honest, important nuance caught while double-checking rather than
  just celebrating the good number: unlike everywhere else in this
  project, feeding blank/scrambled notes here didn't crash the AI's
  performance down to zero, only down partway. Figured out precisely
  why, not just noted it: in this specific setup, the AI can also just
  go read the original source material directly itself, completely
  separately from the notes being tested — so even ruined notes don't
  fully cut off its information. A real, honest limitation of this
  specific test, not a weaker version of the finding — explained clearly
  and written into the paper exactly that way rather than glossed over.
- Also caught and corrected an earlier read of our own: a smaller,
  25-question check earlier looked like a clean, one-sided win for the
  notes version over the plain-text version. At the full 50-question
  size, that specific comparison is no longer a real, provable
  difference — flagged this plainly ourselves and corrected the paper's
  own earlier wording rather than let the more flattering, smaller
  number stand uncorrected.
- Fixed a small, easy mix-up along the way: a command failed with a
  missing-package error because it was accidentally run from the wrong
  isolated setup (the one built specifically for the AI-bridge work
  earlier this week, not the project's main one) — pointed out the fix
  in one line and moved on.
- The other queued task from earlier today — checking the second AI
  model's real-content test on the two remaining configs — finished
  clean. One config confirmed the same real-content-matters pattern
  seen everywhere else in this project. The other one showed the exact
  same partial pattern already found once before on a different config a
  few days ago — not a new puzzle, a second, independent confirmation of
  the same explanation already worked out then (that specific config
  already performs worse on this second AI model to begin with, leaving
  less room to see the fuller effect). Between the two configs, a clean,
  consistent story emerged: the two techniques that involve dropping
  whole layers both show this partial pattern on the second AI model;
  the techniques that only compress without dropping layers both show
  the full, clean pattern — the same architecture-sensitivity finding
  from earlier in the project, showing up again through a completely
  different check. This closes out the last real gap in that part of
  the testing — every technique in the paper except the one already
  known to be broken has now been properly checked on both AI models.
- Finished the piece that was still missing from the delegating-AI real-
  content test: the "wrong but real notes" comparison. Result: the real
  finding now landed in full — real notes clearly beat wrong notes,
  clearly beat blank notes, and clearly beat scrambled notes, all three,
  properly proven. This is the same complete finding already proven
  twice elsewhere in this project, now shown a third time in the most
  free-form, least controlled setup yet — a genuinely stronger result
  here than the AI-bridge work from a few days ago, where the specific
  notes turned out not to matter at all. One honest loose end still
  open: telling "wrong but real" apart from "totally blank/scrambled"
  isn't quite proven yet at this size, even though it's pointing the
  right way — said so plainly rather than round up to a cleaner story
  than what was actually shown.
- Caught something important and genuinely surprising while double-
  checking this result against an earlier one: running the identical
  check twice, on the identical questions, with nothing changed, gave
  meaningfully different answers on almost a third of them. Chased this
  down properly instead of shrugging it off — this specific setup, being
  much more free-form and open-ended (the AI writing and running its own
  code turn by turn) than anything else in this project, turns out not
  to give the exact same answer every time the way the rest of this
  project reliably does. Confirmed this doesn't undermine today's real
  finding (that comparison was made fairly, within one single run), but
  it's a real, distinct, honestly-reported difference from how solid
  and repeatable this project's other results are — written into the
  paper plainly rather than glossed over.
- Also built the same missing piece for the OTHER delegating-AI setup
  from earlier this week (the more structured, split-the-work version)
  — same approach, properly tested — so both versions now have the
  ability to run this same complete check. Only one of the two has
  actually been run with it so far; the other is ready and queued.

---

## 2026-09-15 — Catching up, closing a leftover gap, and validating the AI-bridge direction

- A lot happened on this project since the last entry (a parallel effort
  wrote up several more real findings — a second topology test, a second
  model for the compression story, a third model breaking the pattern in
  an interesting way). Did a proper deep-search catch-up on all of it
  before deciding what to do next, instead of assuming nothing changed.
- Found real, independent validation of the direction we chose to keep
  pursuing: a separate write-up done the same week, after checking four
  brand-new outside papers on bridging two different AI models, landed
  on almost the exact same open question we'd already been chasing —
  confirmed it's genuinely not been answered yet by anyone else, not
  something we're duplicating. Also flagged honestly that this area has
  gotten busy very fast (four outside papers in a few weeks) — worth
  knowing, not a reason to stop.
- Found and fixed a real, small documentation mismatch: one summary file
  claimed a completed test (checking whether a second AI model's real
  notes causally matter, not just its compressed size) had already been
  written into the actual paper — it hadn't. The numbers were sitting
  correct and ready in the summary file since several days ago; the
  actual paper text was just never updated to match. Pulled the real
  numbers directly from the saved results to confirm they're solid (all
  five comparisons clearly significant), then wrote the missing paragraph
  into the paper properly and fixed the summary file's incorrect claim.
- Net effect: one more real, already-earned result now actually counts
  toward the paper instead of sitting finished-but-unwritten, at zero new
  computer time — and the bridge-between-two-different-AI-models work is
  queued up to actually run for the first time today.
- While that first real run was going, looked ahead to a bigger, more
  ambitious idea you'd floated a few turns back: could this project also
  build a cheap way to tell, before an AI even answers, whether the notes
  it just received from another AI look genuine or corrupted? Checked it
  carefully before building anything, and caught a real problem with the
  original plan: two of the three kinds of "corrupted" we'd need to
  detect are specifically built, by our own project's own design, to
  look statistically normal on the simple check that was planned — so
  the plan as originally described would have mostly detected the one
  trivially obvious case and quietly failed at the two that actually
  matter. Also found real outside evidence backing this concern up, not
  just our own reasoning: a very closely related published paper already
  tried almost exactly this and reported it working poorly; a separate,
  unrelated paper independently found the same kind of simple check
  specifically struggles exactly where ours would too. Rewrote the plan
  to be honestly smaller but actually correct: catch the "obviously
  garbage" cases with better checks, and state plainly, up front, that
  telling "wrong but plausible-looking" content apart from real content
  is likely a fundamentally harder, separate problem this simpler version
  isn't going to solve — logged all of this properly so nobody wastes
  time re-discovering it later.
- The first real run of the bridge-between-two-different-AI-models
  finished. Result: on 50 real questions, the AI using the bridge did
  marginally better than either AI working alone, but not by a
  meaningfully real amount — a proper statistical check says this could
  easily be chance, and using the bridge took roughly twice as long as
  just using the better of the two AIs by itself. So: it doesn't clearly
  help, and it clearly costs more time. Logged this plainly rather than
  reading a small, non-real bump as a win. The more important follow-up
  test — whether the bridge's answers actually depend on the SPECIFIC
  real notes it receives, versus just producing plausible-sounding
  answers regardless — is still queued to run next; today's result
  can't answer that question by itself.

---

## 2026-09-11 — A brutally honest gut-check, then three small real additions

- Gave a genuinely harsh answer to a direct question: do any of these
  small extra ideas actually change whether this paper gets accepted?
  Honest answer: no, not really — the paper's fate rides almost entirely
  on the work already done, and small additions mostly polish, not move
  the needle. Said so plainly instead of overselling.
- Dug into "C2C" (a real, published system for bridging two different
  AI models' internal notes) properly this time — found it needs real
  training to build from scratch, but also found something genuinely
  useful: a team already released a ready-to-use, pre-trained version
  for the exact two AI models already used in this project. Also found
  the closest existing evidence (from a paper already cited) suggests
  this kind of bridge probably won't show real information transfer —
  so going in, the honest expectation is a "no" result, not a
  discovery, and said that clearly before recommending anything.
- Built the one genuinely free item: a new figure plotting every
  config's size against its accuracy on one chart, using only numbers
  already reported and verified — nothing new asserted, just a much
  clearer picture, including a visual gut-punch showing that the
  broken 4-bit version, despite being small, is actually pointless
  next to the compressed-and-working version that's just as small.
  Checked it renders correctly, fixed a label-overlap issue, wired it
  into the actual LaTeX paper next to the table it summarizes, and
  double-checked the LaTeX itself is structurally sound (matching
  braces/figure blocks) since a full compiler isn't available here.
- You made the call to keep going on the RLM work over the coming week
  despite the honest downsides flagged — fair, your call to make. Agreed
  on real structure for it (scale up, check stability, real statistics,
  one capped round of fixes, write up whatever it shows) so it doesn't
  repeat the same reactive pattern, with a clear "stop and write it up"
  trigger decided in advance rather than left open-ended.
- Downloaded and inspected the real C2C codebase properly (not just
  summaries this time). Found the exact ready-to-use download for our
  two specific AI models, confirmed their layer/head counts really are
  mismatched (so this is a genuine test of the hard case, not a toy
  one), and caught a real, concrete problem before it caused trouble:
  C2C needs specific, older versions of two core libraries that would
  have broken this project's own setup if installed together — needs
  its own separate, isolated setup instead. Wrote and syntax-checked a
  small script that tests C2C on one real question from this project's
  own dataset instead of a generic example, so the very first read is
  meaningful. Ready for you to run — not yet run.
- Ran it. Hit three real, unglamorous setup problems in a row — the
  isolated setup didn't actually take the first time (my instructions
  used a tool, conda, that turned out not to even be installed), which
  briefly and silently broke this project's own main setup until
  caught and fully restored; then a missing small dependency; then the
  install pulling a CPU-only version of a core library by default.
  Fixed each one directly, confirmed the main project was fully back
  to working after the first issue (not just "probably fine"), and got
  a clean, isolated setup running.
- The actual check passed: C2C produced a coherent, correct-looking
  answer on a real question from this project's own dataset, using
  genuine cross-model communication between the two specific AI models
  already used in this project. Real, working signal — but flagged
  clearly why one example isn't enough to trust yet: this specific
  question is exactly the kind either AI could plausibly get right
  from its own general knowledge, whether or not anything useful was
  actually communicated between them. That's precisely why the real
  next test (swap in a wrong question's notes, see if the answer still
  holds up) matters more than this one passing example.
- Checked whether that real test is even buildable, straight from the
  tool's own source code rather than guessing: confirmed yes, the
  capability genuinely exists and is real, working code — not
  something we'd have to invent. But also found the one place that
  exact usage shows up in the whole toolkit is sitting inside code its
  own authors left switched off, which is a real signal it needs
  careful, unrushed work to use correctly, not a quick add-on.
  Stopped here on purpose, with a clear, well-understood next step
  written down — not a guess — instead of pushing into unfamiliar
  territory under time pressure.
- Read the tool's actual internal logic carefully (not just its
  instructions) and found the real reason its "different content per
  AI" feature exists: it's for the SAME conversation typed out
  differently for two AIs that don't share a vocabulary, not for
  genuinely different, unrelated content — feeding it two truly
  different, differently-sized things directly would break, since it
  lines the two up position-by-position underneath. Real, useful
  finding, and it points to the correct fix: keep the real question
  exactly as-is, and resize the "wrong" question's notes to match its
  length exactly, rather than feed it in raw. Built and syntax-checked
  the actual test this way — same real question asked both times, only
  the notes it's handed swapped between "its own real notes" and "a
  different question's notes." Ready to run — not yet run.
- Ran it: the real-notes answer and the wrong-notes answer came back
  word-for-word identical. Interesting, but caught the same trap this
  project's own main test always guards against before calling it a
  real finding — this specific question ("were these two people the
  same nationality") is exactly the kind either AI might already know
  from general knowledge, with or without any notes at all. An
  identical result doesn't yet tell us "the notes don't matter" versus
  "this question never needed notes in the first place." Added the
  missing third check: same question, but with note-passing switched
  off completely (not swapped for wrong notes — turned off entirely),
  to tell those two apart. Ready to run — not yet run.
- Ran it: real, clean, interpretable result. Turning the note-passing
  off completely DID change the answer (longer, more detailed) — proof
  the channel is genuinely doing something, not silently dead. But
  swapping in wrong notes gave the exact same short answer as the real
  notes did. Put together: something real is happening when notes get
  passed, but WHICH notes get passed doesn't seem to matter for this
  question — a genuine, properly-checked "no" on content mattering,
  not a broken test. This lines up with an existing published finding
  on this exact same setup, now independently double-checked on this
  project's own real reading-comprehension questions instead of just
  taken on faith.
- One example isn't enough to build anything on, so immediately scaled
  it up: built a version that runs this same three-way check across 10
  real questions instead of 1, scoring every answer properly using
  this project's own real scoring code (not a new, separate one) for
  full consistency with every other number in this project. Verified
  directly, not assumed, that borrowing this project's scoring code
  into the separate C2C setup actually works correctly. Ready to run
  — not yet run.
- Ran the 10-question version. Result was noisier than the 1-question
  version, not a clean repeat of it — real notes and wrong notes gave
  the exact same answer on 5 of the 10 questions, not all 10, and on
  the rest the wording shifted a little without the actual answer
  changing much. Also noticed the scoring numbers looked like a flat
  zero across the board and had to explain why: the scoring method
  wants a short, exact answer, but this tool's raw answers are full
  sentences, so the strict "exact match" number is misleading here —
  the more honest signal is whether wrong notes ever actually changed
  what the AI answered, which is the 5-out-of-10 finding above, not
  the zero.
- You asked, fairly, whether any of this actually leads anywhere or
  if it was all just going to become one citation sentence. Answer:
  no, there's a real next step, not just a citation. What we'd
  already validated (the two AIs really can talk to each other, and
  we know exactly what that channel does and doesn't carry) is enough
  to build an actual working piece of the project with it, not just
  observe it. Built that: a real new config, alongside this project's
  existing ones, where one AI's real notes drive the other AI's final
  answer through this bridge — scored properly against two honest
  baselines (each AI answering completely alone) using this project's
  own significance-testing code, not a new one-off script. Flagged
  plainly, in the script itself, the one real limitation this hits:
  the off-the-shelf bridge only supports a 2-AI handoff cleanly, not
  this project's usual 3-AI chain, without real extra engineering —
  said so upfront rather than quietly working around it. Ready to
  run — not yet run.

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

- Ran the fixed script for real. The prompt fix worked — the two "sub-
  assistants" now report real facts instead of bailing out. But with that
  problem out of the way, a smaller, more interesting one showed up: on
  the one question where both assistants clearly found the right fact
  (a book series called Animorphs), the "plain summary" version used it
  correctly, but the "internal notes" version lost it and said "None" —
  even though it was looking at the exact same underlying information.
- Worked out why, in plain terms: when the two assistants write things
  down as text, the boss AI reads both write-ups *together in one go* and
  can naturally connect them. When we hand over internal notes instead,
  each assistant's notes were built completely separately — neither ever
  "saw" the other — so gluing the notes together afterward doesn't let
  them retroactively connect, even once the page-numbering is fixed.
- Built a fix for that: gave the "notes" version an extra thinking step —
  before answering, it now has to write out loud how the two assistants'
  findings connect, using notes that genuinely can see both assistants at
  once (because those particular notes get created *after* the merge, not
  before it). This mirrors the same "reason first, answer after" pattern
  the main 3-agent pipeline already relies on.
- Upgraded the check script itself: it now runs all three versions
  (summary / notes / notes-with-thinking-step), scores every answer with
  the project's real scoring code instead of eyeballing, and prints a
  running tally at the end — so results are measured, not just read.
- Not yet run — that's the next step, with a bigger batch of questions
  this time (15 by default) to get a clearer read than 5 examples give.
- You asked directly: did we actually build real RLM, or just something
  inspired by it? Honest answer was: not the real thing — we'd been using
  a fixed, hard-coded way of splitting up the reading material, not
  letting the AI decide that itself, which is the actual core idea of
  RLM. Explained clearly why: the real version needs the AI to write and
  run its own small programs in a safe sandbox, which is a genuinely
  bigger piece of engineering, and there's a real chance our specific AI
  model (which was never specially trained for this, unlike the one used
  in the original research) might not do it reliably.
- You said to build the real version anyway. Did it in a deliberately
  staged way to manage that risk: built the actual sandbox where the AI
  can write and run real code, with a safe, limited set of allowed
  commands (no file access, no internet, nothing that could do real
  damage) — and, for now, kept the "notes vs summary" question out of it
  entirely, so this first version exactly matches how real RLM behaves
  (hands back plain text only). The reason: first prove the AI can even
  drive this kind of "write code, see what happens, write more code"
  loop at all, before adding our own twist back in on top.
- Tested the sandbox and the control loop thoroughly on a laptop first
  (10 checks, all passing) using a fake stand-in AI that follows a
  script — this proves the plumbing (running code safely, catching
  mistakes without crashing, stopping correctly once an answer is given)
  works correctly, completely separately from the open question of
  whether the REAL AI will use it sensibly.
- Wrote a small script for you to run on the GPU that shows, step by
  step, exactly what the AI writes and does — this is the one that
  actually answers the real open question. Not run yet.
- Ran it for real (n=3). Found a genuine, fixable bug, not a sign the AI
  is incapable: the AI *did* correctly ask its helper for information,
  but forgot to explicitly "print" the helper's answer — so the answer
  came back to it as a blank, and it ended up guessing with nothing to
  go on. Also found a visibility gap in the script itself: whenever the
  AI didn't write any code at all, the script was silently skipping that
  turn instead of showing what it said, so a couple of "went quiet"
  results couldn't be explained.
- Fixed both: made the sandbox automatically show the result of the last
  thing the AI's code did, even if it forgot to "print" it explicitly —
  the same way typing something into an interactive coding notebook
  shows you the answer without needing to ask for it. Also loosened a
  too-strict rule that only recognized code written in one exact
  format, in case that's why one example got no code at all. And made
  the script print everything now, including the previously-invisible
  quiet turns.
- Added 5 more small laptop-only checks (15 total, all passing) proving
  this fix specifically does what it's supposed to, before spending any
  more GPU time on it.
- Ready for another GPU run to see if this actually fixes the behavior.
- Ran it again. Good news: the "auto-show the answer" fix worked exactly
  as intended — the AI's helper calls now correctly show their answers.
- Found two more real, specific problems by reading the transcript
  closely:
  1. Once the AI had decided on an answer, it reliably stopped wrapping
     its "I'm done, here's my answer" command in the required code
     formatting — even though it used that formatting correctly earlier
     in the same run. So the command silently never ran, and it just
     repeated the same unrecognized line over and over until giving up.
  2. Bigger issue: the AI's helper calls were never actually including
     any of the real reading material — it was just asking bare
     questions like "who played this role?" with nothing attached for
     the helper to read. Since the helper has no access to the passages
     on its own, it was left guessing from general knowledge instead of
     reading anything real — which is exactly why one answer came back
     confidently wrong (a made-up actress and job title with zero
     connection to the actual source material).
- Fixed both: (1) the system now also recognizes the "I'm done" command
  even when the AI forgets the formatting, instead of requiring it every
  single time; (2) rewrote the instructions to explicitly show an
  example of correctly including real passage text in a helper call, and
  to spell out plainly that skipping this gets an unreliable guess, not
  a real answer.
- Added 2 more laptop-only checks (17 total, all passing) proving the
  new fallback works before spending more GPU time.
- Ready for another GPU run.
- Ran it again (n=3). Real, meaningful progress this time: on 2 of 3
  questions the AI correctly indexed into the real passage list itself
  instead of guessing blind. One question came out substantively
  correct (Shirley Temple / Chief of Protocol) — it only "failed" the
  strictest scoring because it added a couple of extra words to an
  otherwise right answer. On the third question, the old problem
  (asking its helper ungrounded questions with nothing real attached)
  still showed up — inconsistent, not solved everywhere yet.
- Added one more targeted nudge: if the AI calls its helper twice in a
  row without ever having looked at the real passages itself, it now
  gets a one-time reminder to go check the actual reading material
  first, instead of continuing to guess. Added 2 more laptop checks
  (19 total, all passing) proving this fires exactly once, only when
  actually needed.
- Next: a bigger run (8-10 questions) to see if this holds up beyond a
  handful of examples.
- Ran it (n=10). Confirmed both fixes from last round actually work now
  (the passages-nudge correctly fired this time; a stray True/False
  answer correctly turned into scoreable text). But it surfaced a
  related, deeper version of the very first bug: the "auto-show the
  result" fix only covered a helper call sitting alone on its own line —
  once the AI wrote more realistic code (a search loop checking several
  passages), a helper call buried inside that loop went right back to
  being invisible, for the same underlying reason as before. One
  question also showed the AI retrying the exact same failing search
  four times in a row without ever changing approach.
- Fixed both, more robustly this time: instead of trying to guess which
  lines deserve auto-showing, the helper itself now always announces
  its own answer, no matter where in the code it gets called from — so
  this whole category of "the AI asked but never saw the reply" bug is
  closed, not just the one shape of it we'd already seen. Also added a
  one-time nudge specifically for "you just tried the exact same thing
  and got the exact same (unhelpful) result — try something else."
- Added 2 more laptop checks (24 total, all passing).
- Ready for another GPU run.
- Ran it (n=10) — the best result yet: **2 of 10 questions came out
  completely, exactly correct**, plus two more that were substantively
  right (just scored strictly for exact wording). Confirmed both fixes
  from last round are working as intended, most clearly on the question
  about a K-pop album's label — the AI's search found the real answer
  inside a passage, and (unlike before the fix) actually saw it this
  time, extracted it correctly, and answered right.
- What's left now looks like genuine reasoning limits of a
  non-specially-trained AI, not plumbing bugs: on a couple of questions
  it only checked 2 of the 10 passages before giving up and admitting
  "unknown"; on one, it searched using the exact wording of the
  question instead of words likely to actually appear in the answer;
  on another it found the correct real fact but never went back to fix
  an earlier guess it had made up before finding it. One question also
  came out worse than an earlier attempt — expected and explained: every
  fix changes the exact wording the AI sees at every step for every
  question, not just the one that motivated the fix, so some individual
  questions shift in either direction even as the overall trend
  improves.
- Did some quick research on the remaining syntax slips (writing
  final_answer without quotes, or with a comma instead of parentheses).
  Confirmed this is a known, common issue specifically for AI models
  this size that haven't been specially trained for tool use, and the
  standard, well-supported fix is a single clear worked example shown
  up front — not more written instructions, which is what we'd been
  adding. Added one complete example to the instructions showing the
  correct format end to end. All 24 laptop checks still pass.
- Ready for another GPU run to see if this closes the remaining gap.
- Ran it (n=10) — same overall score as last time (2/10 exact matches)
  but genuinely useful new information, not a plateau:
  1. The worked example added last round backfired slightly: the AI
     started writing its own FAKE guess at what the result would be,
     right next to its code, instead of waiting for the real one —
     copying the example's layout too literally. Harmless in practice
     (the real result always overrides the fake one) but wasteful.
     Rewrote the example to explicitly say "the system reports back
     separately, never guess it yourself."
  2. Clear, direct proof the 6-turn limit was too short: one question
     had the AI correctly, methodically check six passages in a row,
     actually find both real facts needed to answer correctly — and
     then simply run out of turns right as it was about to answer.
     Raised the limit to 10.
  3. The single biggest pattern across the wrong answers: the AI
     checking only 1-3 of the 10 passages before giving up or guessing
     from a weak match, when the real answer was often still sitting
     unread. Added a check: if it tries to finalize an answer after
     barely looking around, it now has to confirm ("are you sure, or do
     you want to check more first?") instead of being allowed to stop
     immediately.
- Added 2 more laptop checks proving the new confirmation step fires
  exactly when it should (26 total, all passing) — updating the several
  existing checks that now correctly need one extra "confirm" step too.
- Ready for another GPU run.
- Ran it (n=10): **3 of 10 exact right this time**, up from 2. The
  "are you sure, or check more first?" fix is clearly doing its job —
  it fired on 6 of the 10 questions and, in several of them, genuinely
  drove the AI to keep looking and find the real answer it would have
  missed otherwise. Every remaining wrong answer now has an
  understandable, ordinary explanation (picked the wrong of two similar
  numbers in a passage it found correctly; answered a yes/no question
  with a fact instead of "yes"; made a reasonable-sounding but wrong
  guess) — not a plumbing failure.
- Caught a real mistake of my own while reading this run: last round I
  raised the turn limit from 6 to 10 inside the core logic, but the
  separate script you actually run had its OWN copy of that same
  number, which I forgot to update — so every run since then was
  silently still capped at 6. Confirmed directly: two questions hit
  exactly that old limit while still making real progress. Fixed the
  script's own copy to match.
- Stepped back and re-explained the big picture and an honest "is this
  working" verdict, since it's easy to lose the thread after this many
  rounds of bug-fixing: yes, it's working — exact-match has gone
  0 → 0 → 2 → 3 out of 10 across real runs, and the kind of failure
  has shifted entirely from mechanical bugs to ordinary reasoning
  limits. Also clarified: everything built so far is still just the
  scaffolding (the "hand back a written summary" version, matching real
  RLM). The actual new idea this project is testing — handing back
  internal notes instead — hasn't been built yet.
- Decision: one more confirmation run with the turn-limit bug actually
  fixed, then start building the real internal-notes version on top of
  this now-solid scaffolding.
- Ran the confirmation (n=10). Score held at 3/10, but the fixed turn
  limit clearly did its job underneath that number: one question now
  genuinely used all 10 turns and actually found the exact right
  passage on turn 9 — but then, instead of answering, it just kept
  going and checked one more (irrelevant) passage out of habit, running
  out of turns right as it had the answer in hand. A brand new, very
  specific, cheap-to-fix problem: the AI doesn't recognize "I found it"
  as a signal to stop looking. A second question that failed outright
  last time now got substantively closer (missing only two words of the
  full correct phrase). Also noticed one problem repeating consistently
  across every single run tonight: yes/no questions keep getting
  answered with a restated fact instead of the word "yes" or "no."
- Added two small, targeted instructions to address both directly:
  stop searching the moment a passage answers the question, and answer
  yes/no questions with exactly "yes" or "no." Both are just wording
  changes, no logic changes — confirmed nothing else broke (24 laptop
  checks still pass).
- This is very likely the last round of this kind of tuning before
  moving on to the real build (the internal-notes swap) — returns are
  visibly getting smaller and more specific with each round now.
- Ran it once more (n=10): **4 of 10 exact right — the best result
  yet, and both fixes are directly, visibly responsible.** The yes/no
  question that had been wrong in every single earlier run tonight
  finally came out correct. And on another question, the AI switched
  from blindly checking passages one by one to writing a real keyword
  search ("check each passage for the word 'Aladin'") — a genuinely
  smarter strategy, not just a lucky guess. The real trend across every
  run tonight: 0 → 0 → 2 → 3 → 4 out of 10, each step tied to a specific,
  understood fix. Everything still wrong now is an ordinary reasoning
  limit (picking the wrong of two true numbers/facts sitting in a
  correctly-found passage, or a logical guess that sounds reasonable but
  isn't) — not a mechanical problem anymore.
- **Calling Stage 1 done here.** Next step is the real build: making the
  helper hand back its internal notes instead of written text, on top
  of this now-solid foundation.
- **Built Stage 2.** This needed one real architectural change first: the
  "boss" AI's own thinking had been getting fully re-read from scratch,
  as text, on every single turn — there was nowhere to attach a helper's
  internal notes even if we wanted to. Rebuilt the boss's own loop so it
  now genuinely carries its own notes forward turn to turn (the same
  trick the main project already uses), instead of restarting from
  scratch each time.
- With that in place, built the actual comparison: a "summary" version
  (matches everything from Stage 1, now just running on the upgraded
  note-carrying boss) and a real "internal notes" version, where a
  helper's answer gets spliced directly onto the boss's own notes
  instead of ever being turned back into words. The helper's answer is
  still available as normal text to the AI's own code (so its
  decision-making logic keeps working exactly as before) — only what
  the boss AI itself "reads" afterward changes.
- Kept every one of Stage 1's working fixes (the nudges for grounding,
  repeated code, checking enough passages, yes/no formatting, stopping
  once found) carried over unchanged, so this is a fair, like-for-like
  comparison, not a step back to an earlier, rougher version.
- Tested the one genuinely new piece of math on a laptop first (5
  checks, all passing) before touching the GPU, same approach as every
  other new mechanism tonight.
- Not yet run on a real question — that's next.
- Ran the first real comparison (n=5). Score: summary version 2/5, notes
  version 1/5 — but reading the actual transcripts found two distinct,
  real, fixable problems, not "the idea doesn't work":
  1. Splicing a helper's notes in whole included the helper's OWN
     private instructions ("read this and respond concisely") along
     with its actual answer — confusing the boss AI, which produced
     blank responses and once literally wrote the word "assistant" into
     its own code. Fixed by keeping only the helper's generated answer,
     dropping its private framing before splicing.
  2. A second, separate problem showed up in BOTH versions equally (so
     it wasn't about notes vs. summaries at all): severe repetition
     loops, the exact same failure already documented elsewhere in this
     project. Root cause: the new note-carrying "boss" loop checks for
     repeated text using a much shorter memory window than the old
     version did, since it no longer re-reads everything from scratch
     each turn. Fixed by tracking repeated text across the WHOLE
     conversation instead of just the current turn, reusing the same
     approach the main pipeline already relies on for this exact reason.
  3. Both fixes verified with dedicated laptop checks before touching
     the GPU again (7 checks total for this file, all passing).
  This is the expected shape of a first real round on a brand-new
  mechanism — same pattern as every round of Stage 1's early debugging.
- Ran it again (n=5). Both earlier fixes visibly helped (no more massive
  repeated-text spam like before) but a bigger, more fundamental problem
  was still there in both versions: the AI would write its real action,
  then just keep going and MAKE UP a fake version of what the system
  would say back — a fabricated "here's what happened" — and then a
  fake next action, all in one breath, before the real system ever got
  a chance to respond. Proved this directly rather than assuming it:
  fed the AI's own malformed code through the real execution path and
  confirmed it should have produced an error message — completely
  different from the friendly, made-up text the AI had written for
  itself.
- This mattered a lot because of how this new loop works: whatever gets
  generated becomes part of the AI's permanent notes going forward. So
  the AI's own fabricated story about what happened was getting baked
  in as if it were real, right alongside (or sometimes instead of) the
  actual system response — meaning some "correct" answers were arrived
  at by trusting its own fiction, not real information.
- Fixed by cutting generation off the instant one real action is
  complete, instead of letting the AI keep going. Added a test that
  checks this cutoff happens at exactly the right point — not before
  the action is finished, and never so late that any made-up content
  gets through.
- 8 checks total for this file now, all passing. Ready for another
  real run — this is likely the fix that actually lets a fair
  notes-vs-summary comparison happen for the first time.

## 2026-09-10 (continued) — A reality check, and getting back to what matters

- Stepped back and asked directly: is this RLM detour turning into
  overengineering, given the deadline? Honest answer: yes. Rated it
  plainly — the code itself is genuinely solid (multiple real bugs found
  and fixed correctly, well tested throughout), but pouring a whole
  session into a side experiment a month from deadline, while two
  already-flagged, more important paper checks sat untouched the entire
  time, was the wrong call on prioritization. Owned that directly rather
  than deflecting.
- Agreed on a clean close: one last already-built GPU run gets the final
  word on the RLM work, no more new engineering after it either way, and
  attention goes back to the actual paper.
- Picked back up the two related-work checks that got dropped earlier
  tonight (Q-KVComm and RelayCaching — both real, relevant papers found
  during the original RLM research pass). Checked both carefully:
  neither threatens anything already established. RelayCaching solves a
  different problem (reusing notes only when two agents are reading
  near-identical material, not a full relay like this project does).
  Q-KVComm does something close to this project's own layer-selection
  idea, but only ever checks end-to-end accuracy — never the "did the
  notes actually matter" test this project's whole causal-audit
  contribution is built on. Added honest, precise citations for both to
  the actual paper — small, careful additions, not rewrites.
- **Ran the agreed final RLM check. Closing the thread here, as
  promised — no more work on this tonight regardless of what it
  showed.** Result: worse than the previous run on both versions, and
  one question broke down into genuine gibberish in both — a new way of
  failing, not one of the ones already fixed. Honest reading: the
  system remains too fragile at this AI's size, without special
  training, to trust any comparison from it yet. If anything, this
  cleanest run leans toward the "internal notes" version doing *worse*
  than the "summary" version, matching the very first, much simpler
  test from earlier tonight — but five questions is nowhere near enough
  to call that a real finding either way.
- Wrote an honest, complete paragraph for the paper describing exactly
  this: what was built, the real bugs found and fixed along the way,
  and the honest state it's left in — a real piece of engineering and a
  partial hint of a negative result, not a validated finding. Nothing
  overclaimed, nothing hidden.
- Did a real reality check, asked for directly: is the RLM detour
  overengineering given the deadline? Answer: yes. Rated it honestly,
  agreed to close it out with one last run and no further debugging
  regardless of outcome, and refocused on the actual paper — checked
  the two related-work papers that had been sitting untouched all
  session (both fine, added precise citations for both), and confirmed
  the paper itself is essentially complete already.
- Pointed to the two real remaining tasks: verifying the nibble-packing
  fix end-to-end, and extending the causal audit to two more configs.
  Got the nibble-packing check back first: **exact match on every
  number** (accuracy, F1, size, compression ratio) between the original
  report and a fresh, real GPU run — confirming that fix is solid, not
  just unit-tested. Updated the paper to reflect this closed, no longer
  open, caveat.
- Ran it. **Clean, complete success — the best possible outcome.** Both
  new configs (`C`, and the fixed `B_int4_kivi`) show the exact same
  pattern already established for the other three: real accuracy holds
  up, swapping in garbage notes collapses accuracy to literally zero,
  and swapping in a different real question's notes lands in between —
  every single one of 12 statistical comparisons came back significant.
  This closes the exact gap the paper's own Limitations section named:
  the causal-audit finding now holds across every real relay condition
  in the whole paper (no compression, layer-selection alone, 8-bit,
  layer-selection plus compression, and the most aggressive 4-bit
  version) — not just three of five. Updated both the Results section
  and Limitations to reflect this. This is now a genuinely stronger
  paper than it was this morning, from real work on the real central
  claim — the kind of hour this project actually needed.

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
