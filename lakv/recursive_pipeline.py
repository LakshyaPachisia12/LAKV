"""
LAKV research-extensions prototype: RLM-style recursive decomposition
combined with LAKV's KV-relay mechanism, applied to HotpotQA's distractor
passages.

Motivation (see docs/naacl2027_paper_draft.md for the full related-work
reconciliation): Recursive Language Models (Zhang, Kraska, Khattab, arXiv
2512.24601) decompose a long prompt into sub-calls and relay ONLY a
token-limited text answer back to the parent, by design, to bound context
growth -- discarding the sub-call's full computed representation. The
closest adjacent work ("Recursive Models for Long-Horizon Reasoning", arXiv
2603.02112) explicitly considers KV-cache-based return values, but only as
an unimplemented assumption for a theoretical speedup calculation; its real
experiments discard a child call's reasoning entirely, same as RLM. Neither
tests whether relaying the child's KV cache instead of re-tokenized text
changes accuracy or cost -- this module is a minimal testbed for exactly
that question, transplanting this project's existing `A`-vs-`text_agent`
comparison onto a decompose-solve-aggregate topology instead of a fixed
three-role chain.

Proof-of-concept scope (deliberately narrow -- see docs for the staged plan):
  - Fixed, static passage partition into `n_children` groups. NOT RLM's
    dynamic, LM-driven decomposition via a Python REPL -- that is out of
    scope here. This is a real simplification, named as a limitation.
  - Depth-1 recursion only: one aggregator, `n_children` leaf sub-calls,
    no further recursion (matches the reproduction literature's finding
    that RLM recursion beyond depth-1 tends to degrade -- see "Think, But
    Don't Overthink: Reproducing Recursive Language Models", arXiv
    2603.02615).
  - Three return-channel conditions to compare:
      "kv"   -- each child's raw KV cache is relayed directly into the
                aggregator via a fan-in merge (see merge_child_kv), which
                then answers immediately with no intermediate reasoning.
      "text" -- each child's decoded text is concatenated into the
                aggregator's prompt as plain text (RLM's actual mechanism).
      "kv_synthesis" -- same fan-in merge as "kv", but the aggregator first
                generates an explicit reasoning pass over the merged cache
                before answering (see _run_aggregator_reasoning). Added
                after an n=5 spot check (2026-09-10) found plain "kv" losing
                a fact both children clearly reported, plausibly because
                each child's cache was computed in total isolation from the
                other -- unlike "text", where the aggregator's single joint
                forward pass over both children's write-ups lets it
                cross-reference them for free. "kv_synthesis" tries to
                recover that by giving the aggregator its own generation
                step over the merged cache first: those newly generated
                reasoning tokens, unlike the frozen child caches, ARE
                computed attending over both children at once.
  - No compression, no layer selection, no offset-correction/causal-audit
    hooks yet -- deliberately minimal, to validate the multi-source KV
    fan-in position mechanics in isolation before layering anything else
    from lakv/kv_compressor.py or lakv/layer_selector.py on top.
  - Greedy decoding only (do_sample=False); sampling is not implemented.

The RoPE position-shift needed to fan multiple independently-computed child
caches into one combined cache reuses AnchorTable._rope_shift_k
(lakv/anchor_table.py) -- the same constant-position-shift primitive already
validated and in production for config E's offset correction -- applied here
for a new purpose (re-basing an independent child's cache into the
aggregator's combined position range) rather than reimplementing RoPE-shift
math from scratch.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

from lakv.anchor_table import AnchorTable, question_key
from lakv.causal_audit import apply_causal_audit, KVAuditPool


CHILD_SYSTEM_PROMPT = (
    "You are given a subset of the context passages related to a larger "
    "question, along with the full question. Read only the passages given "
    "to you. Do not judge whether these passages are relevant or "
    "sufficient to answer the question -- this is a multi-hop question, "
    "so the answer may depend on connecting a fact in these passages to a "
    "fact in passages you were not given, meaning a passage can matter "
    "even if it does not look directly relevant on its own. Instead, "
    "objectively summarize the key facts in these passages: names, dates, "
    "titles, roles, and relationships, quoting the exact sentence(s) they "
    "come from. Do not attempt to answer the full question yourself -- "
    "only report what these specific passages say."
)

AGGREGATOR_REASONING_SYSTEM_PROMPT = (
    "You have findings from multiple sub-agents, each of whom reviewed a "
    "different subset of the context passages, plus the original question. "
    "This is a multi-hop question -- a fact reported by one sub-agent may "
    "only become useful once connected to a fact reported by a different "
    "sub-agent. Explicitly connect the sub-agents' findings to each other "
    "and to the question, reasoning step by step, before drawing any "
    "conclusion."
)

AGGREGATOR_SYSTEM_PROMPT = (
    "You are the final answer agent. You have findings from multiple "
    "sub-agents, each of whom reviewed a different subset of the context "
    "passages, plus the original question. Combine their findings to "
    "answer the question. Output exactly one line in this format: "
    "The answer is: <answer>. The answer should be a short word or phrase, "
    "not a full sentence. Do not add explanation or any other text."
)


def format_passage(title: str, sentences: List[str]) -> str:
    return f"[{title}] " + " ".join(sentences)


def split_passages(passages: List[str], n_children: int) -> List[List[str]]:
    """Fixed, static, content-agnostic partition -- deliberately not RLM's
    dynamic LM-driven decomposition (see module docstring). Round-robin
    rather than contiguous blocks, so no single child is guaranteed to get
    an all-distractor or all-gold group purely from HotpotQA's own passage
    ordering."""
    groups: List[List[str]] = [[] for _ in range(n_children)]
    for i, p in enumerate(passages):
        groups[i % n_children].append(p)
    return groups


def load_hotpotqa_structured(split: str = "validation", n: Optional[int] = None) -> List[dict]:
    """Like run.py's load_hotpotqa, but keeps passages as a List[str]
    instead of flattening them into one opaque question string -- the
    recursive pipeline needs passage-level granularity to split across
    children. Deliberately a separate loader (not a modification of
    run.py's existing one) to keep this prototype isolated from the
    established pipeline's data path."""
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    data = []
    for item in ds[split]:
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        passages = [format_passage(t, s) for t, s in zip(titles, sentences)]
        data.append({
            "question": item["question"],
            "answer": item["answer"],
            "passages": passages,
        })
    if n is not None:
        data = data[:n]
    return data


@dataclass
class RecursivePipelineConfig:
    n_children: int = 2
    return_channel: str = "kv"       # "kv" | "text" | "kv_synthesis"
    child_max_new_tokens: int = 256
    aggregator_max_new_tokens: int = 128
    aggregator_reasoning_max_new_tokens: int = 200
    child_system_prompt: str = CHILD_SYSTEM_PROMPT
    aggregator_system_prompt: str = AGGREGATOR_SYSTEM_PROMPT
    aggregator_reasoning_system_prompt: str = AGGREGATOR_REASONING_SYSTEM_PROMPT
    generation_kwargs: Dict[str, object] = field(default_factory=lambda: {
        "do_sample": False,
        "repetition_penalty": 1.05,
    })
    # Causal-audit substitution, applied to ONE child's KV before
    # merge_child_kv -- mirrors lakv/causal_audit.py's three-tier test
    # (mode "none"/"zeroed"/"random"/"mismatched") applied to this
    # topology's fan-in merge point instead of the sequential pipeline's
    # hop-to-hop handoff. Only meaningful for return_channel in
    # ("kv", "kv_synthesis") -- ignored for "text" (no KV is merged
    # there). causal_audit_child_idx selects WHICH child's KV gets
    # substituted (default: the last child, matching how the sequential
    # pipeline's audit substitutes the incoming hop closest to the
    # receiver); real content in every other child stays untouched, so a
    # degraded answer can be attributed to that one substitution.
    causal_audit_mode: str = "none"
    causal_audit_child_idx: int = -1
    audit_pool: Optional["KVAuditPool"] = None


@dataclass
class RecursiveHopStat:
    child_idx: int
    seq_len: int


@dataclass
class RecursiveRunResult:
    answer: str
    child_texts: List[str]
    hop_stats: List[RecursiveHopStat]
    aggregator_reasoning_text: str = ""
    audit_log: Optional[dict] = None  # None if causal_audit_mode == "none"


class RecursiveKVPipeline:
    """Minimal depth-1 RLM-style decomposition pipeline, with a KV-relay vs
    text-relay switch on the child->aggregator return channel. Proof-of-
    concept: static partition, no compression, no offset correction, greedy
    decoding only."""

    def __init__(self, model, tokenizer, config: RecursivePipelineConfig,
                 device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self._eos_ids = self._get_stop_token_ids()

    def _get_stop_token_ids(self) -> set:
        ids = set()
        gen_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if gen_eos is not None:
            ids.update(gen_eos if isinstance(gen_eos, (list, tuple, set)) else [gen_eos])
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids

    def _build_prompt_ids(self, system_prompt: str, user_content: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

    def _sample_next_token(self, logits: torch.Tensor, generated_ids: Optional[List[int]] = None) -> torch.Tensor:
        kwargs = self.config.generation_kwargs
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
        if repetition_penalty != 1.0 and generated_ids:
            unique_ids = torch.tensor(sorted(set(generated_ids)), device=logits.device, dtype=torch.long)
            seen = logits[0, unique_ids]
            seen = torch.where(seen < 0, seen * repetition_penalty, seen / repetition_penalty)
            logits = logits.clone()
            logits[0, unique_ids] = seen
        if kwargs.get("do_sample", False):
            raise NotImplementedError(
                "Sampling not implemented in this proof-of-concept -- greedy only."
            )
        return logits.argmax(-1, keepdim=True)

    @staticmethod
    def _to_tuple(pkv) -> tuple:
        if isinstance(pkv, tuple):
            return pkv
        if isinstance(pkv, DynamicCache) and hasattr(pkv, "to_legacy_cache"):
            return pkv.to_legacy_cache()
        layers = []
        for layer in pkv:
            if isinstance(layer, tuple):
                layers.append((layer[0], layer[1]))
            else:
                layers.append((layer.keys, layer.values))
        return tuple(layers)

    @staticmethod
    def _to_dynamic_cache(kv_tuple: tuple) -> DynamicCache:
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_tuple):
            cache.update(k, v, layer_idx)
        return cache

    @staticmethod
    def merge_child_kv(child_kvs: List[tuple], rope_theta: float) -> tuple:
        """Fan-in merge: concatenate N independently-computed child KV
        tuples into one combined cache the aggregator can inject as if it
        were a single upstream sender.

        Child 0's KV is used as-is -- it was computed at local positions
        0..len_0-1, which is also where it needs to sit in the merged
        sequence. Every subsequent child's KV was ALSO computed starting
        from local position 0 (each child ran as an independent forward
        pass with no knowledge of the others), so before concatenating, its
        K tensors need every position shifted by the cumulative length of
        all children already placed before it -- otherwise its RoPE-encoded
        keys would misrepresent their position in the merged sequence (the
        same "position drift" failure mode already characterized in this
        project for config E; here it's corrected for up front instead of
        allowed to happen). V tensors need no shift: RoPE encodes Q/K only.

        Reuses AnchorTable._rope_shift_k -- the same constant-position-shift
        primitive already in production for offset correction -- for a new
        purpose (re-basing an independent child's cache) rather than
        reimplementing RoPE-shift math from scratch. Correctness of the
        underlying algebraic assumption (shifting an already-rotated key by
        delta == rotating the original key directly by position+delta) is
        checked in tests/test_recursive_kv_merge.py, not just asserted here.
        """
        if not child_kvs:
            raise ValueError("merge_child_kv requires at least one child KV tuple")

        n_layers = len(child_kvs[0])
        shifted_per_child: List[tuple] = []
        cumulative_len = 0
        for child_idx, kv in enumerate(child_kvs):
            if child_idx == 0:
                shifted_per_child.append(kv)
            else:
                shifted = tuple(
                    (AnchorTable._rope_shift_k(k, shift=cumulative_len, theta=rope_theta), v)
                    for (k, v) in kv
                )
                shifted_per_child.append(shifted)
            cumulative_len += int(kv[0][0].shape[2])

        merged_layers = []
        for layer_idx in range(n_layers):
            ks = [shifted_per_child[c][layer_idx][0] for c in range(len(child_kvs))]
            vs = [shifted_per_child[c][layer_idx][1] for c in range(len(child_kvs))]
            merged_layers.append((torch.cat(ks, dim=2), torch.cat(vs, dim=2)))
        return tuple(merged_layers)

    def _get_rope_theta(self) -> float:
        if hasattr(self.model.config, 'rope_theta'):
            return self.model.config.rope_theta
        if hasattr(self.model.config, 'rope_parameters'):
            return self.model.config.rope_parameters.get('rope_theta', 1_000_000.0)
        return 1_000_000.0

    def _run_child(self, question: str, passage_group: List[str]) -> Tuple[tuple, str, int]:
        user_content = (
            "Passages:\n" + "\n\n".join(passage_group) + f"\n\nQuestion: {question}"
        )
        input_ids = self._build_prompt_ids(self.config.child_system_prompt, user_content)
        with torch.no_grad():
            gen_out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.config.child_max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                **self.config.generation_kwargs,
            )
        new_tokens = gen_out.sequences[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        kv_tuple = self._to_tuple(gen_out.past_key_values)
        seq_len = int(kv_tuple[0][0].shape[2])
        return kv_tuple, text, seq_len

    def _generate_with_injected_kv(self, question: str, merged_kv: tuple) -> str:
        user_content = f"Question: {question}"
        input_ids = self._build_prompt_ids(self.config.aggregator_system_prompt, user_content)
        cache = self._to_dynamic_cache(merged_kv)
        cache_seq_len = int(merged_kv[0][0].shape[2])
        prompt_len = input_ids.shape[1]
        position_ids = torch.arange(
            cache_seq_len, cache_seq_len + prompt_len, device=self.device
        ).unsqueeze(0)
        attention_mask = torch.ones(
            (1, cache_seq_len + prompt_len), dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        running_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]
        cur_pos = cache_seq_len + prompt_len

        first_token = self._sample_next_token(next_logits)
        first_tok_id = first_token.item()
        if first_tok_id in self._eos_ids:
            return ""
        handoff_mask = torch.ones((1, cur_pos + 1), dtype=torch.long, device=self.device)
        with torch.no_grad():
            gen_out = self.model.generate(
                input_ids=first_token,
                past_key_values=running_cache,
                attention_mask=handoff_mask,
                max_new_tokens=max(self.config.aggregator_max_new_tokens - 1, 1),
                use_cache=True,
                return_dict_in_generate=True,
                **self.config.generation_kwargs,
            )
        return self.tokenizer.decode(gen_out.sequences[0].tolist(), skip_special_tokens=True)

    def _run_aggregator_reasoning(self, question: str, merged_kv: tuple) -> Tuple[tuple, str]:
        """Intermediate synthesis step, mirroring this project's existing
        Reasoner->Verifier pattern: generate real reasoning tokens while
        attending over the merged children cache, BEFORE answering.

        Why this helps (see module docstring): the children's caches were
        each computed in isolation and never attended to each other, so
        concatenating them (even position-corrected) doesn't let them
        cross-reference -- only the aggregator's OWN new tokens, generated
        after the merge, get to attend over both at once. Explicitly
        generating a reasoning pass here (instead of jumping straight to a
        one-line answer) gives the model tokens whose K/V genuinely reflect
        both children's content jointly, extending the cache the final
        answer step reads from -- an attempt to recover, via real generated
        tokens, some of what a fresh joint text read gets for free.
        """
        user_content = f"Question: {question}"
        input_ids = self._build_prompt_ids(self.config.aggregator_reasoning_system_prompt, user_content)
        cache = self._to_dynamic_cache(merged_kv)
        cache_seq_len = int(merged_kv[0][0].shape[2])
        prompt_len = input_ids.shape[1]
        position_ids = torch.arange(
            cache_seq_len, cache_seq_len + prompt_len, device=self.device
        ).unsqueeze(0)
        attention_mask = torch.ones(
            (1, cache_seq_len + prompt_len), dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        running_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]
        cur_pos = cache_seq_len + prompt_len

        first_token = self._sample_next_token(next_logits)
        first_tok_id = first_token.item()
        if first_tok_id in self._eos_ids:
            return self._to_tuple(running_cache), ""
        handoff_mask = torch.ones((1, cur_pos + 1), dtype=torch.long, device=self.device)
        with torch.no_grad():
            gen_out = self.model.generate(
                input_ids=first_token,
                past_key_values=running_cache,
                attention_mask=handoff_mask,
                max_new_tokens=max(self.config.aggregator_reasoning_max_new_tokens - 1, 1),
                use_cache=True,
                return_dict_in_generate=True,
                **self.config.generation_kwargs,
            )
        text = self.tokenizer.decode(gen_out.sequences[0].tolist(), skip_special_tokens=True)
        return self._to_tuple(gen_out.past_key_values), text

    def _generate_from_text(self, question: str, child_texts: List[str]) -> str:
        findings = "\n\n".join(f"Sub-agent {i}: {t}" for i, t in enumerate(child_texts))
        user_content = f"{findings}\n\nQuestion: {question}"
        input_ids = self._build_prompt_ids(self.config.aggregator_system_prompt, user_content)
        with torch.no_grad():
            gen_out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.config.aggregator_max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                **self.config.generation_kwargs,
            )
        new_tokens = gen_out.sequences[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def run(self, question: str, passages: List[str]) -> RecursiveRunResult:
        groups = split_passages(passages, self.config.n_children)
        child_kvs: List[tuple] = []
        child_texts: List[str] = []
        hop_stats: List[RecursiveHopStat] = []

        for i, group in enumerate(groups):
            if not group:
                continue
            kv, text, seq_len = self._run_child(question, group)
            child_kvs.append(kv)
            child_texts.append(text)
            hop_stats.append(RecursiveHopStat(child_idx=i, seq_len=seq_len))

        audit_log = None
        if self.config.causal_audit_mode != "none" and self.config.return_channel in ("kv", "kv_synthesis"):
            target_idx = self.config.causal_audit_child_idx
            if not (-len(child_kvs) <= target_idx < len(child_kvs)):
                raise ValueError(
                    f"causal_audit_child_idx={target_idx} out of range for "
                    f"{len(child_kvs)} children"
                )
            substituted_kv, audit_log = apply_causal_audit(
                self.config.causal_audit_mode, child_kvs[target_idx],
                agent_idx=target_idx, question_key=question_key(question),
                pool=self.config.audit_pool,
            )
            child_kvs = list(child_kvs)
            child_kvs[target_idx] = substituted_kv

        reasoning_text = ""
        if self.config.return_channel == "kv":
            merged_kv = self.merge_child_kv(child_kvs, rope_theta=self._get_rope_theta())
            answer = self._generate_with_injected_kv(question, merged_kv)
        elif self.config.return_channel == "kv_synthesis":
            merged_kv = self.merge_child_kv(child_kvs, rope_theta=self._get_rope_theta())
            extended_kv, reasoning_text = self._run_aggregator_reasoning(question, merged_kv)
            answer = self._generate_with_injected_kv(question, extended_kv)
        elif self.config.return_channel == "text":
            answer = self._generate_from_text(question, child_texts)
        else:
            raise ValueError(f"Unknown return_channel: {self.config.return_channel!r}")

        return RecursiveRunResult(
            answer=answer, child_texts=child_texts, hop_stats=hop_stats,
            aggregator_reasoning_text=reasoning_text, audit_log=audit_log,
        )
