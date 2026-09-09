# LaTeX submission package

Self-contained ACL/ARR submission directory. `main.tex` is the real paper,
converted from `docs/naacl2027_paper_draft.md` — that markdown file is the
working draft going forward; edit there, then re-sync changes into
`main.tex` (they will drift if edited independently).

## Contents

- `main.tex` — the paper (`\documentclass[11pt]{article}`, `acl.sty` in
  `review` mode for anonymous ARR submission).
- `acl.sty`, `acl_natbib.bst` — official ACL style files, downloaded
  verbatim from `github.com/acl-org/acl-style-files` (2026-09-09). Do not
  hand-edit these; re-download if a newer version is announced.
- `references.bib` — copy of `docs/references.bib`. Keep these in sync;
  the copy here is what `main.tex` actually compiles against.
- `figures/fig1_causal_audit.png`, `figures/fig2_b_int4_journey.png` —
  copies of `docs/figures/*.png`, regenerate via
  `scripts/make_paper_figures.py` and re-copy if the underlying results
  change.

## Compiling

No LaTeX toolchain was available in the environment this was written in,
so this has been checked statically (brace balance, every `\citep`/`\citet`
key resolves against `references.bib`, no unescaped `%`/`&`/`_`, table
column count matches header count, all `\label`/`\ref` pairs resolve) but
**not actually compiled**. Do one real compile pass before trusting it
fully — static checking catches most but not all LaTeX failures (package
interactions, font substitution, overfull boxes).

**Easiest: Overleaf.** Create a new blank project, upload every file in
this directory (preserving the `figures/` subfolder), set the main
document to `main.tex`, compile with pdfLaTeX. No local install needed.

**Local, if you have a LaTeX distribution** (TeX Live, MiKTeX):
```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
(Two `pdflatex` passes after `bibtex` are needed to resolve citation
numbers/labels correctly — this is normal LaTeX/BibTeX behavior, not a
sign anything is wrong.)

## Before actually submitting

- Switch `\usepackage[review]{acl}` to `\usepackage{acl}` (drops
  anonymization + line numbers) only for the camera-ready version, after
  acceptance — ARR submission itself must stay anonymous.
- Re-check the current 8-page limit and whether Limitations/references
  still count as exempt (they did as of the 2026-09-09 check documented in
  the paper's Limitations section, via the ARR author checklist).
- Fill out the Responsible NLP Research checklist on the ARR submission
  platform itself — it's a separate form, not something this .tex file
  produces.
