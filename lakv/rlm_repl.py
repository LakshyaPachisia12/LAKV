"""
Stage 1 of a real (not fixed-split) Recursive Language Model implementation.

lakv/recursive_pipeline.py builds a FIXED, model-independent passage split
specifically to isolate a different question (KV-relay vs text-relay at a
combine step) from decomposition quality. This module builds the actual
RLM decomposition mechanism instead: the root model decides, via Python
code it writes and executes itself, how to inspect the given context and
when to delegate a piece of it to a sub-model call -- following the real
mechanism in Zhang, Kraska, and Khattab, "Recursive Language Models"
(arXiv 2512.24601): the context lives as a variable in a persistent REPL,
and the model can call other LM instances from within that REPL.

Staged on purpose (see docs/PROGRESS_REPORT.md): this module uses ONLY a
text-returning llm_query() -- matching RLM's real, published mechanism
exactly -- so Stage 1 answers one open question in isolation: can
Qwen2.5-7B-Instruct (never fine-tuned for this, unlike the original paper's
RLM-Qwen3-8B, which needed 1,000 rejection-sampled trajectories of
fine-tuning before it drove this loop reliably) drive a real code-writing
decomposition loop at all? Only once that's confirmed on real examples
does Stage 2 make sense: adding a KV-relaying alternative to llm_query()
and re-asking this project's real research question (KV vs text) on top
of genuine model-driven decomposition instead of a fixed split.

Sandboxing note -- a real, named scoping decision, not an oversight: this
uses Python's exec() in a restricted namespace with a conservative
builtins allowlist (see SAFE_BUILTINS), not process- or container-level
isolation (e.g. Docker, which alexzhang13/rlm's reference implementation
uses). Adequate for a local research prototype, run by its own author,
against a model under their own control, on their own machine -- NOT
adequate for running untrusted code from an adversarial or external
source. If this ever moves beyond a local research prototype, replace
CodeSandbox's exec()-based isolation with real process/container
sandboxing before running anything you don't trust.
"""

import ast
import builtins
import re
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch


# Conservative allowlist: enough for real "inspect/slice/search the
# context" behavior (the actual point of RLM) without exposing anything
# that reads/writes the filesystem, network, or process state. Built from
# the real `builtins` module rather than the ambient `__builtins__` name,
# which is a plain dict in some execution contexts and a module in others
# -- using `builtins` directly sidesteps that inconsistency entirely.
_SAFE_BUILTIN_NAMES = (
    "len", "range", "enumerate", "zip", "map", "filter", "sorted",
    "reversed", "sum", "min", "max", "abs", "round", "any", "all",
    "str", "int", "float", "bool", "list", "dict", "tuple", "set",
    "print", "repr", "isinstance", "type",
)
SAFE_BUILTINS = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}

MAX_STDOUT_CHARS = 800   # RLM's own design: only constant-size metadata
                          # about stdout goes back into the model's history,
                          # not the full raw output -- prevents a single
                          # verbose print() from blowing up context growth
                          # the same way naively dumping the whole prompt
                          # would.

# Case-insensitive, and the "python"/"py" language label is optional --
# real-model testing (2026-09-10) found the model sometimes writes a bare
# ``` fence with no label, which the original (label-required,
# case-sensitive) version of this pattern silently failed to match at all,
# making the model's code invisible to the loop for an entire example.
CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class SandboxError(Exception):
    pass


@dataclass
class SandboxTurn:
    """Record of one execute() call, for logging/debugging a run."""
    code: str
    stdout: str
    stdout_truncated: bool
    error: Optional[str] = None
    final_answer: Optional[str] = None


class CodeSandbox:
    """Restricted exec() environment holding the question/context as
    variables, plus an injected llm_query() the model's code can call.

    A real RLM implementation lets the model call itself/sub-models
    directly from code it writes -- this class is that mechanism, scoped
    down to a single-process, single-machine, trusted-author setting (see
    module docstring for exactly what that does and doesn't cover).
    """

    def __init__(self, question: str, passages: List[str],
                 llm_query: Callable[[str], str]):
        self.done = False
        self.answer: Optional[str] = None
        self._llm_query_calls = 0

        def _llm_query(text: str) -> str:
            self._llm_query_calls += 1
            result = llm_query(text)
            # Always echo the result as a side effect, regardless of
            # where in the code this call sits -- real-model testing
            # (2026-09-10) found the earlier "auto-print the last bare
            # expression" fix only covers a call sitting alone on the
            # final line; a call buried inside a for-loop or if-block
            # (exactly what realistic search code looks like) still
            # vanished silently, since Python itself doesn't display
            # values from nested statements. This makes the result
            # visible unconditionally instead of depending on where the
            # call happens to sit in the code's structure.
            print(f"[llm_query result] {result}")
            return result

        def _final_answer(text) -> None:
            # str() rather than assuming text is already a string: real-
            # model testing (2026-09-10) found the model sometimes calls
            # final_answer(False) or similar non-string values (following
            # the literal wording of a yes/no question rather than the
            # system prompt's instruction to answer with a short phrase).
            # Coercing here means that still produces a scoreable answer
            # ("False") instead of silently breaking extract_qa_answer
            # downstream, which expects a string.
            self.done = True
            self.answer = str(text)

        self.namespace = {
            "__builtins__": SAFE_BUILTINS,
            "question": question,
            "passages": list(passages),
            "llm_query": _llm_query,
            "final_answer": _final_answer,
        }

    @property
    def llm_query_calls(self) -> int:
        return self._llm_query_calls

    def execute(self, code: str) -> SandboxTurn:
        import io
        import contextlib

        stdout_buf = io.StringIO()
        error: Optional[str] = None
        try:
            tree = ast.parse(code)
            body = tree.body
            with contextlib.redirect_stdout(stdout_buf):
                if body and isinstance(body[-1], ast.Expr):
                    # Jupyter/IPython-style auto-display: if the code's
                    # last top-level statement is a bare expression (e.g.
                    # `llm_query("...")` on its own line, not assigned to
                    # a variable), run everything before it normally, then
                    # evaluate that last expression separately and print
                    # its value if it isn't None. Real-model testing
                    # (2026-09-10) found the model frequently calls
                    # llm_query() and never explicitly print()s the
                    # result -- without this, that result goes nowhere and
                    # the model ends up guessing blind, having never
                    # actually seen the answer it asked for. A plain
                    # exec() alone (the original version of this method)
                    # has no equivalent of a REPL's "show me what that
                    # line evaluated to," which is exactly the behavior a
                    # model used to interactive/notebook-style tool use
                    # would expect by default.
                    *init_stmts, last_expr = body
                    init_module = ast.Module(body=init_stmts, type_ignores=[])
                    ast.fix_missing_locations(init_module)
                    exec(compile(init_module, "<rlm_repl>", "exec"), self.namespace)
                    last_expression = ast.Expression(body=last_expr.value)
                    ast.fix_missing_locations(last_expression)
                    result = eval(compile(last_expression, "<rlm_repl>", "eval"), self.namespace)
                    if result is not None:
                        print(repr(result))
                else:
                    exec(code, self.namespace)
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any
            # error in model-generated code must be caught and fed back as
            # an observation, not crash the whole session.
            error = f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=2)

        raw_stdout = stdout_buf.getvalue()
        truncated = len(raw_stdout) > MAX_STDOUT_CHARS
        stdout = raw_stdout[:MAX_STDOUT_CHARS]
        if truncated:
            stdout += f"\n...[truncated, {len(raw_stdout)} total chars]"

        return SandboxTurn(
            code=code, stdout=stdout, stdout_truncated=truncated,
            error=error, final_answer=self.answer if self.done else None,
        )


RLM_SYSTEM_PROMPT = (
    "You are solving a question using a Python REPL. Two variables are "
    "already defined: `question` (str) and `passages` (a list of context "
    "passage strings -- there may be many, and they may not all be "
    "relevant). You do not need to read every passage yourself.\n\n"
    "You have two tools, callable from Python code:\n"
    "  llm_query(text: str) -> str -- asks a fresh sub-assistant to read "
    "`text` and respond. It has NO memory of this conversation and NO "
    "ACCESS to `passages` or anything else here -- it can only see "
    "exactly the string you pass it. You MUST include the actual passage "
    "text you want it to read inside that string, for example:\n"
    "```python\n"
    "llm_query(passages[0] + '\\n\\nQuestion: ' + question)\n"
    "```\n"
    "Calling it with only a bare question and no passage text attached "
    "will get you an unreliable guess, not a grounded answer -- it has "
    "nothing to read.\n"
    "  final_answer(text: str) -- submits your final answer and ends the "
    "session. Call this only when you are ready to answer.\n\n"
    "EVERY action -- including a single final_answer(...) call -- MUST be "
    "written inside a fenced code block like:\n"
    "```python\n"
    "# your code here\n"
    "```\n"
    "Do not write final_answer(...) or llm_query(...) as bare text "
    "outside a code block; it will not run. Only the FIRST such block in "
    "your response is executed. Anything you print() -- or the value of "
    "the last line, even without print() -- will be shown back to you "
    "(truncated if long). You may take multiple turns: inspect passages, "
    "call llm_query with the actual text of the ones that look relevant, "
    "reason about what comes back, and call final_answer(...) once you "
    "know the answer. Keep your final answer short -- a word or phrase, "
    "not a full sentence. If the question is phrased as yes/no (starts "
    "with \"Are\", \"Were\", \"Is\", \"Was\", \"Did\", etc.), your final "
    "answer must be exactly \"yes\" or \"no\", not a restated fact. "
    "IMPORTANT: as soon as a passage you've read directly answers the "
    "question, call final_answer(...) immediately -- do not keep "
    "checking further passages out of habit once you already have the "
    "answer.\n\n"
    "Here is a complete worked example of the expected format, using an "
    "unrelated question (do not reuse its content). The system runs your "
    "code and reports back what happened AS A SEPARATE MESSAGE, AFTER "
    "you write it -- never write your own guess at what that report will "
    "say. Only write the code; wait for the real result:\n\n"
    "You write, in turn 1:\n"
    "```python\n"
    "passages[2]\n"
    "```\n"
    "The system then reports back to you (you do not write this part "
    "yourself): [stdout]\\n\"[Example Bridge] The Example Bridge was "
    "completed in 1932 and is located in Example City.\"\n\n"
    "You write, in turn 2:\n"
    "```python\n"
    "final_answer(\"1932\")\n"
    "```\n\n"
    "Notice the final answer is a short, QUOTED string inside "
    "parentheses -- not a bare word with no quotes, not separated from "
    "final_answer by a comma, and not a full sentence. Also notice each "
    "turn contains ONLY code -- never a guess at what the system will "
    "report back."
)

# Fallback for when the model writes a recognizable call to one of the two
# tools as bare text with no code fence at all -- observed directly in
# real-model testing (2026-09-10): once the model had "decided" on an
# answer, it reliably stopped wrapping final_answer(...) in a fence, even
# though it had used one correctly earlier in the same run. Rather than
# require the model to remember the exact format every time, recognize an
# unfenced call and run it anyway. Only handles the common single-quoted-
# string-argument shape; anything more complex still needs a real fence.
BARE_CALL_RE = re.compile(r'(final_answer|llm_query)\s*\(\s*(["\'])(.*?)\2\s*\)', re.DOTALL)


def _references_passages_variable(code: str) -> bool:
    """True only if `passages` is referenced as an actual variable (e.g.
    `passages[0]`, `for p in passages`), not merely mentioned as a word
    inside a string literal. A naive substring check on the word
    "passages" was found, in real-model testing (2026-09-10), to
    false-positive on the model's own natural-language strings like
    "I should check the passages for clues" -- text ABOUT checking
    passages, with no actual passages[...] access -- which silently
    disabled the grounding nudge below for the rest of that example, the
    exact reason one example's nudge never fired despite the model never
    having read a real passage."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Name) and node.id == "passages"
        for node in ast.walk(tree)
    )


@dataclass
class RLMRunResult:
    answer: Optional[str]
    turns: List[SandboxTurn] = field(default_factory=list)
    hit_max_turns: bool = False
    llm_query_calls: int = 0
    # Full role/content message history, INCLUDING turns where the model
    # wrote no code at all (those are silently nudged and skipped from
    # `turns`, which only records turns that reached sandbox.execute()).
    # Added after real-model testing (2026-09-10) went quiet for several
    # turns with no visibility into what the model was actually saying --
    # this is what makes that diagnosable instead of a black box.
    transcript: List[dict] = field(default_factory=list)


class RLMSession:
    """Drives the root model through the write-code / execute / observe
    loop until it calls final_answer() or a turn budget is exhausted.

    generate_fn: Callable[[List[dict]], str] -- given a chat-style message
    list (role/content dicts), returns the root model's next response as
    decoded text. Kept as an injected function (not a hardcoded model
    call) so this class can be unit-tested with a scripted stub, without a
    GPU or a real model -- see tests/test_rlm_repl.py.
    """

    def __init__(self, generate_fn: Callable[[List[dict]], str],
                 llm_query: Callable[[str], str], max_turns: int = 10):
        self.generate_fn = generate_fn
        self.llm_query = llm_query
        self.max_turns = max_turns

    def run(self, question: str, passages: List[str]) -> RLMRunResult:
        sandbox = CodeSandbox(question, passages, self.llm_query)
        messages = [
            {"role": "system", "content": RLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        turns: List[SandboxTurn] = []
        touched_passages = False
        nudged_about_grounding = False
        nudged_about_repetition = False
        low_exploration_confirmed = False
        last_code: Optional[str] = None
        # Distinct turns whose code referenced `passages` -- a light-
        # weight proxy for "how much of the real material has this
        # actually looked at," not a count of distinct passages (a loop
        # touching several passages in one turn only counts once here,
        # but that's fine: the point is distinguishing "barely looked"
        # from "made a real effort," not precise accounting).
        passage_touch_turns = 0

        for _ in range(self.max_turns):
            response = self.generate_fn(messages)
            messages.append({"role": "assistant", "content": response})

            match = CODE_FENCE_RE.search(response)
            if match is not None:
                code = match.group(1)
            else:
                bare_match = BARE_CALL_RE.search(response)
                if bare_match is not None:
                    # Reconstruct just the recognized call as code, rather
                    # than trying to exec() the surrounding prose (which
                    # usually isn't valid Python at all).
                    func_name, quote, arg = bare_match.groups()
                    code = f"{func_name}({quote}{arg}{quote})"
                else:
                    # No code this turn -- nudge it instead of silently
                    # stalling, so a model that "talks itself out" of
                    # writing code doesn't just burn turns with no
                    # observation to learn from.
                    messages.append({
                        "role": "user",
                        "content": (
                            "No ```python code block found in your last "
                            "response. Write code in a fenced ```python "
                            "block to continue, or call final_answer(...) "
                            "inside one to finish."
                        ),
                    })
                    continue
            if _references_passages_variable(code):
                touched_passages = True
                passage_touch_turns += 1

            code_repeated = (last_code is not None and code.strip() == last_code.strip())
            last_code = code

            turn = sandbox.execute(code)
            turns.append(turn)

            if sandbox.done:
                # Real-model testing (2026-09-10) found the dominant
                # failure pattern across wrong answers was giving up
                # after checking only 1-3 of the (often 10) passages,
                # either guessing from a barely-related one or answering
                # "unknown"/"cannot determine" -- when the real answer
                # was often still sitting unread. Give one chance to
                # reconsider before accepting a low-exploration answer,
                # rather than ending the session immediately; if it
                # confirms (calls final_answer again, even with the same
                # text), accept it -- this is a nudge, not a hard block.
                if passage_touch_turns < 3 and not low_exploration_confirmed:
                    low_exploration_confirmed = True
                    tentative_answer = sandbox.answer
                    sandbox.done = False
                    sandbox.answer = None
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Before finalizing: you have only directly "
                            f"looked at {passage_touch_turns} of "
                            f"{len(passages)} passages so far, and your "
                            f"answer was going to be "
                            f"{tentative_answer!r}. If you're confident "
                            "that's correct, call final_answer(...) "
                            "again to confirm. Otherwise, check a few "
                            "more passages first -- the answer may "
                            "still be sitting in one you haven't read."
                        ),
                    })
                    continue

                return RLMRunResult(
                    answer=sandbox.answer, turns=turns, hit_max_turns=False,
                    llm_query_calls=sandbox.llm_query_calls, transcript=messages,
                )

            observation = (
                f"[stdout]\n{turn.stdout}" if not turn.error
                else f"[error]\n{turn.error}"
            )
            messages.append({"role": "user", "content": observation})

            # Real-model testing (2026-09-10) found the model sometimes
            # writes the EXACT same code again after getting an empty or
            # unhelpful result, repeatedly, with no adaptation -- e.g. a
            # search loop whose filter string doesn't match anything,
            # rerun unchanged four times in a row. Since the observation
            # will necessarily be identical too, there is nothing new for
            # the model to react to on its own; nudge it to change
            # approach instead of burning the whole turn budget on
            # repeats. Fires once per run, on the first repeat.
            if code_repeated and not nudged_about_repetition:
                nudged_about_repetition = True
                messages.append({
                    "role": "user",
                    "content": (
                        "Note: that is the exact same code as your last "
                        "attempt, and it will produce the same result "
                        "again. Try a different approach -- e.g. a "
                        "different search term, or just printing each "
                        "passage directly to see what's actually there, "
                        "instead of re-running the same filter."
                    ),
                })

            # Real-model testing (2026-09-10) found the model sometimes
            # never touches `passages` at all -- it calls llm_query
            # repeatedly with bare, ungrounded questions and gets
            # confident-sounding but fabricated answers back, since the
            # helper has nothing real to read either. Fire once (not
            # every turn, to avoid nagging) if it's called llm_query at
            # least twice with no passage access in between.
            if (not touched_passages and not nudged_about_grounding
                    and sandbox.llm_query_calls >= 2):
                nudged_about_grounding = True
                messages.append({
                    "role": "user",
                    "content": (
                        "Note: you have called llm_query more than once "
                        "without ever reading any of the `passages` list "
                        "(e.g. passages[i]) yourself. llm_query has no "
                        "access to `passages` either, so open-ended "
                        "questions with no passage text attached will "
                        "keep getting you unreliable guesses. Try "
                        "inspecting the actual passages directly before "
                        "asking further questions -- the answer may "
                        "already be sitting in one of them."
                    ),
                })

        return RLMRunResult(
            answer=None, turns=turns, hit_max_turns=True,
            llm_query_calls=sandbox.llm_query_calls, transcript=messages,
        )


def make_model_generate_fn(model, tokenizer, device: str = "cuda",
                            max_new_tokens: int = 300) -> Callable[[List[dict]], str]:
    """Wraps a real HF model+tokenizer into the generate_fn signature
    RLMSession expects. Plain model.generate() -- no KV injection, no
    manual decode loop -- Stage 1 deliberately does not touch this
    project's KV-relay machinery yet (see module docstring)."""

    def _generate(messages: List[dict]) -> str:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _generate


def make_child_llm_query_fn(model, tokenizer, device: str = "cuda",
                             max_new_tokens: int = 200) -> Callable[[str], str]:
    """The Stage 1 (text-only) sub-call: a fresh one-shot generation with
    no memory of the root's conversation, matching RLM's real mechanism.
    Stage 2 will add a KV-relaying alternative to this same call site."""

    child_system_prompt = (
        "Read the given text and respond to the request in it concisely. "
        "You have no memory of any other conversation."
    )

    def _llm_query(text: str) -> str:
        messages = [
            {"role": "system", "content": child_system_prompt},
            {"role": "user", "content": text},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
            )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _llm_query
