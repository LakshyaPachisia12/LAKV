# Reference audit for main_v8.tex (references_v8.bib)

Generated 2026-10-12 from `references_v8.bib` and the citations in `main_v8.tex`.
Nothing in this file was checked online; fields are what the bib file contains.
"Unverified" = the entry you said could not be opened by hand today.

## Mechanical checks
- 29 bib entries, 28 distinct keys cited, no cited key missing from the bib, no duplicate keys.
- **Uncited entry:** `rethinkredundancy2026` (Kim et al., "Rethinking Layer Redundancy", arXiv 2604.24938). BibTeX
  ignores it, so it does not appear in the reference list. Not removed; tell me if you want it deleted.
- `\ref{tab:causal_audit_full}` resolves to the label inside the `\input` table file; all other refs resolve.
- Same-surname pairs: kivi2024 and liu2024lost both render "Liu et al. 2024" and natbib will add a/b;
  rlm2026 "Zhang et al." vs zhangemu2026 "Zhang and Emu" are distinguishable.
- **"and 3 others" for Wolf et al.:** the bib already has all 22 authors. The ACL style file (`acl_natbib.bst`,
  lines 573-606) truncates any list over 20 names to 19 names plus "and N others". This is the official style's
  behaviour, not a bib gap. I did not touch the .bst.

## Table 1: every entry
Legend: OK = field present. Author lists are complete unless marked.

| key | authors | title (as in bib) | venue | year | pages / ID | link | flags |
|---|---|---|---|---|---|---|---|
| transformers2020 | Wolf et al. (22, complete) | Transformers: State-of-the-Art NLP | EMNLP 2020 System Demonstrations | 2020 | 38-45 | ACL Anthology | rendered "and 3 others" by the .bst |
| rlm2026 | Zhang, Kraska, Khattab | Recursive Language Models | arXiv preprint | 2026 | 2512.24601 | arXiv | **check for a published version**; year 2026 vs Dec-2025 arXiv ID (title page year is yours to confirm) |
| canonicalmerge2026 | Baquero, Brito | Cache Merging as a Convergent Replicated State for Multi-Agent Latent Reasoning | arXiv | 2026 | 2607.01308 | arXiv (html URL) | verified by you |
| zhangemu2026 | Zhang, Emu | Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication | arXiv | 2026 | 2607.26773 | arXiv | **UNVERIFIED** |
| shikvcomm2025 | Shi, Chiesa, Maguire Jr., Kostic | KVComm: Enabling Efficient LLM Communication through Selective KV Sharing | ICLR 2026 | 2026 | arXiv 2510.03346 | arXiv | key says 2025, venue year 2026 (harmless); no pages/OpenReview link |
| kvcomm2025 | Ye et al. (11) | KVCOMM: Online Cross-context KV-cache Communication... | NeurIPS 2025 | 2025 | arXiv 2510.12872 | arXiv | OK |
| latentmas2026 | Zou et al. (13) | Latent Collaboration in Multi-Agent Systems | ICML 2026 | 2026 | arXiv 2511.20639 | arXiv | OK; no pages/PMLR link |
| turboquant2026 | Zandieh, Daliri, Hadian, Mirrokni | TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate | ICLR 2026 | 2026 | arXiv 2504.19874 | arXiv | OK |
| polarquant2026 | Han, Kacham, Karbasi, Mirrokni, Zandieh | PolarQuant: Vector Quantization with Polar Transformation | AISTATS 2026, PMLR 300 | 2026 | 5113-5121 | arXiv | **title and pages changed in v8** per your instruction |
| kvtc2026 | Staniszewski, Lancucki | KV Cache Transform Coding for Compact Storage in LLM Inference | ICLR 2026 | 2026 | arXiv 2511.01815 | arXiv | verified by you |
| kivi2024 | Liu et al. (8) | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache | ICML 2024, PMLR 235 | 2024 | 32332-32344 | PMLR | OK |
| rotatekv2025 | Su et al. (7) | RotateKV: ... Outlier-Aware Adaptive Rotations | IJCAI 2025 | 2025 | 6200-6208 | IJCAI | OK |
| kvquant2024 | Hooper et al. (7) | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization | NeurIPS 2024 | 2024 | arXiv 2401.18079 | arXiv | no pages |
| kvreusefails2026 | Liang et al. (6) | When KV Cache Reuse Fails in Multi-Agent Systems... | arXiv | 2026 | 2601.08343 | arXiv | verified by you |
| latentcommpay2026 | Cheng, Das, Ramnath | When Does Latent Communication Pay? A Causal Audit of Relayed KV Caches in Multi-Agent LLMs | arXiv | 2026 | 2608.04893 | arXiv | **UNVERIFIED**; cited 4 times, including three substantive claims about its findings |
| nofreeswap2026 | Garcia | No Free Swap: Protocol-Dependent Layer Redundancy in Transformers | arXiv | 2026 | 2605.16234 | arXiv | verified by you |
| rethinkredundancy2026 | Kim et al. (6) | Rethinking Layer Redundancy... | arXiv | 2026 | 2604.24938 | arXiv | **uncited** |
| adak2026 | Shao | AdaK: Adaptive KV Cache Budget Estimation Framework... | Frontiers in AI 9 | 2026 | DOI 10.3389/frai.2026.1892739 | DOI + URL | verified by you; single author |
| pitfallskv2026 | Chen, Geh, Grover, Van den Broeck, Israel | The Pitfalls of KV Cache Compression | ACL 2026 | 2026 | arXiv 2510.00231 | ACL Anthology | verified by you |
| latentagentslie2026 | Brito, Baquero | When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration | arXiv | 2026 | 2606.28958 | arXiv | verified by you |
| orthobackfill2026 | Li, An, Du | When Less Latent Leads to Better Relay: ... | arXiv | 2026 | 2604.13349 | arXiv | verified by you |
| hotpotqa2018 | Yang et al. (7, complete) | HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA | EMNLP 2018 | 2018 | 2369-2380 | ACL Anthology | OK |
| gsm8k2021 | Cobbe et al. (12, complete) | Training Verifiers to Solve Math Word Problems | arXiv | 2021 | 2110.14168 | arXiv | preprint only (no refereed version known to me) |
| qwen2p5_2024 | "Qwen Team" | Qwen2.5 Technical Report | arXiv | 2024 | 2412.15115 | arXiv | **organisation author, not a full name list** (see decisions) |
| mistral7b2023 | Jiang et al. (18, complete) | Mistral 7B | arXiv | 2023 | 2310.06825 | arXiv | preprint only |
| qwen3_2025 | **"Yang, An and others"** | Qwen3 Technical Report | arXiv | 2025 | 2505.09388 | arXiv (added in v8) | **truncated author list**, I did not invent one |
| shi2023distracted | Shi et al. (8) | Large Language Models Can Be Easily Distracted by Irrelevant Context | ICML 2023 | 2023 | arXiv 2302.00093 | none besides arXiv ID | no pages / PMLR link in bib |
| liu2024lost | Liu et al. (7) | Lost in the Middle: How Language Models Use Long Contexts | TACL 12 | 2024 | 157-173 | **no URL or DOI** | add an Anthology/DOI link if you want one |
| xie2024knowledgeconflicts | Xie et al. (5) | Adaptive Chameleon or Stubborn Sloth... | ICLR 2024 | 2024 | arXiv 2305.13300 | none besides arXiv ID | no OpenReview link |

arXiv style: the three entries that used `journal = {arXiv preprint arXiv:ID}` (rlm2026, canonicalmerge2026,
qwen3_2025) are now `@misc` with `eprint`/`archivePrefix`, so every preprint renders the same way ("Preprint, arXiv:ID").

Preprint-only entries that may have a refereed version (flagged, **not changed**): rlm2026, zhangemu2026,
latentcommpay2026, canonicalmerge2026, latentagentslie2026, orthobackfill2026, kvreusefails2026, nofreeswap2026,
rethinkredundancy2026 (uncited), and the model reports (qwen2p5_2024, qwen3_2025, mistral7b2023, gsm8k2021).

## Table 2: each citation, the sentence it supports, plausibility (for your spot-check; no sentence edited)

| key | sentence (shortened) | does the cited work plausibly say this? |
|---|---|---|
| kvcomm2025, latentmas2026 | "pass each other the KV cache instead of decoded text, so a receiving agent need not re-encode what its sender already read" | Yes for both (KV-cache communication between agents). |
| kvcomm2025 | "KVCOMM corrects KV-cache offset drift when reusing cache across prefixes" | Plausible: KVCOMM is about reusing cache across diverging prefixes with offset correction. |
| latentmas2026 | "agents exchange latent thoughts through KV working memory" | Plausible per its latent-collaboration framing. |
| latentmas2026 | "LatentMAS's layer-wise KV concatenation" (Scope detail, appendix) | Plausible but I cannot confirm "layer-wise concatenation" from the bib alone; hand-check. |
| shikvcomm2025 | "KVComm is a distinct system with a near-identical name" | Yes: different authors/paper from KVCOMM. |
| rlm2026 | "a root model delegates through a sandboxed REPL rather than a fixed topology" | Yes, matches the RLM paper's setup. |
| canonicalmerge2026 | "studies merging parallel caches, reporting downstream accuracy only"; Qwen3-1.7B/4B, HotpotQA bridge, merge-order question | You verified this source today. |
| latentcommpay2026 | "swap the relayed cache for zeroed / moment-matched random / mismatched caches"; "audit three published relay systems"; "no-communication control is a zeroed cache rather than an empty one"; "ceiling on LatentMAS, partial on KVComm, no advantage on C2C" | **UNVERIFIED. Highest priority to hand-check.** Four specific claims rest on it. |
| zhangemu2026 | "adds a no-message baseline, defines an other-message effect, Qwen3-4B/8B, GSM8K/ARC-Challenge/MATH-500, mixed signs reversing between sizes" | **UNVERIFIED. Hand-check**, including that its no-message baseline is what our dropped condition follows. |
| latentagentslie2026 | "audit cache integrity under an adversarial agent" | You verified this source today. |
| kivi2024 | "KIVI quantizes keys per-channel and values per-token" | Yes. |
| kivi2024 | "...motivated by RoPE leaving channel-wise magnitude more consistent across positions than per-head ranges assume" (App. C) | **Doubtful.** As I recall, KIVI motivates per-channel keys by fixed outlier channels in the key cache, not by RoPE. Please check the paper; this sentence may over-attribute. |
| kivi2024, rotatekv2025, kvquant2024 | "identify the difficulty of quantizing post-RoPE keys" | RotateKV (pre-RoPE rotation) and KVQuant (pre-RoPE key quantization) fit. **KIVI is the weaker fit** for the specific RoPE claim. |
| rotatekv2025 | "rotates before RoPE via Pre-RoPE Grouped-Head Rotation" | Plausible from the title/abstract of RotateKV. |
| kvquant2024 | "post-RoPE key quantization is harder than value quantization" | Plausible (KVQuant introduces pre-RoPE key quantization). |
| turboquant2026, polarquant2026 | "rotation-based scheme in the style of TurboQuant and PolarQuant"; "PolarQuant rotate-then-quantize; TurboQuant separate QJL term; two papers sharing two authors" | Author overlap checks out against the bib (Zandieh, Mirrokni). The QJL/rotate descriptions are from memory of the papers; hand-check. |
| kvtc2026 | "transform coding (PCA decorrelation, adaptive quantization, entropy coding) for up to 20x compression" | You verified this source today. |
| orthobackfill2026 | "injects a low-rank residual of discarded content orthogonal to what is retained; its published formula requires attention weights" | Source verified by you, but the "requires attention weights" detail is our own reading; hand-check. |
| nofreeswap2026 | "layer redundancy depends on model and protocol" | You verified this source today. |
| kvreusefails2026 | "reuse strategies effective for generation agents can silently corrupt an LLM judge's cross-candidate comparison" | You verified this source today. |
| adak2026 | "adaptive KV-budget estimation generalizing across Qwen3 (4B/8B) and Mistral-7B-Instruct-v0.2" | You verified this source today. |
| pitfallskv2026 | "compression methods optimize aggregate metrics while equating token retention with functional preservation" | You verified this source today. |
| shi2023distracted | "models are distracted by irrelevant context"; "in line with the distractor literature" | Yes (title). |
| liu2024lost | "use information unevenly across a long context" | Yes (lost-in-the-middle). |
| xie2024knowledgeconflicts | "respond to conflicting evidence in ways that depend on how coherent it looks" | Plausible: the paper reports models accept coherent, convincing conflicting evidence. |
| hotpotqa2018, gsm8k2021 | dataset citations | Yes. |
| qwen2p5_2024, qwen3_2025, mistral7b2023 | model citations | Yes. |
| transformers2020 | "built directly on transformers" | Yes. |

## Task 7: dated evidence for "specified, with predicted outcomes, before they were run"
- File: `docs/PREDICTIONS_BEFORE_RUNS_2026-10-11.md`. Windows creation time **2026-10-11 11:25:22**; last write
  2026-10-12 10:11:53 (outcomes appended later, as the file says it would be).
- Result folders for the runs it covers: Run 1 (`results/run_20261011_122408`): folder name time 12:24, first partial
  results file written 12:27:56 on 2026-10-11, i.e. about 1 hour after the file was created. Run 3
  (`results/recursive_poc_check_withpass/run_20261011_132223`): started 13:22:23, about 2 hours after. The remaining
  result folders for Runs 4-6 on 10-11 all have names timestamped 14:10 or later; none starts before 11:25.
- The earlier runs that the predictions are compared against (`run_20261009_113254`, standard chain n=100) predate the
  file, as expected: they are the baselines, not predicted runs.
- Run 2 (wrong-text control) has a decision rule but no directional prediction, and the fan-in text-relay row (Run 3b)
  is labelled exploratory in the file; the paper already calls the text comparison exploratory.
- **Problem you must decide on:** the file's own header says "Written and committed BEFORE any of the runs", but it is
  **untracked in git** (`git status` shows `??`; `git log` for it is empty). So there is no commit timestamp. The only
  evidence is filesystem times, which are editable and do not prove the content at 11:25 matched what is there now. If a
  reviewer asked, "specified before they were run" would rest on file times alone. Options: (a) say "recorded in advance
  in a dated notes file" without "committed", or (b) commit the file now with an honest message (it would then prove
  nothing about 10-11, so do not claim it does).
