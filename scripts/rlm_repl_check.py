"""
Real-model check for lakv/rlm_repl.py -- Stage 1 of the actual RLM
mechanism (code-writing decomposition via a real Python REPL sandbox),
as opposed to lakv/recursive_pipeline.py's fixed-split prototype.

This is the one that answers the real open question: can
Qwen2.5-7B-Instruct (never fine-tuned for this) reliably write working
decomposition code and drive itself to a final_answer() call at all? The
CPU-only loop/sandbox mechanics are already verified in
tests/test_rlm_repl.py -- this script is the first real-model step, and
deliberately small (n=3 by default) and turn-by-turn verbose, since the
point right now is reading whether the model's OWN generated code and
reasoning are coherent, not measuring accuracy yet. Expect this to need
a few rounds of system-prompt iteration -- that's the normal, expected
shape of getting a non-fine-tuned model to drive an agentic code loop.

Usage:
    python scripts/rlm_repl_check.py --n 3
    python scripts/rlm_repl_check.py --n 3 --max_turns 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import load_model
from lakv.qa_scoring import extract_qa_answer, exact_match_score, f1_score
from lakv.recursive_pipeline import load_hotpotqa_structured
from lakv.rlm_repl import RLMSession, make_model_generate_fn, make_child_llm_query_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=3)
    # Was 6, matching RLMSession's old default -- when that default was
    # raised to 10 (2026-09-10, after real-model testing showed a
    # question run out of turns while making genuine progress toward a
    # correct answer), this script's OWN separate argparse default was
    # missed, so every run since then was silently still capped at 6
    # regardless of that fix. Confirmed directly: two questions in the
    # very next real run hit exactly "max_turns=6" while mid-progress.
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, device="cuda")
    data = load_hotpotqa_structured(split=args.split, n=args.n)

    generate_fn = make_model_generate_fn(model, tokenizer, device="cuda")
    llm_query_fn = make_child_llm_query_fn(model, tokenizer, device="cuda")
    session = RLMSession(generate_fn=generate_fn, llm_query=llm_query_fn, max_turns=args.max_turns)

    n_correct = 0
    for i, item in enumerate(data):
        print(f"\n{'=' * 70}\nExample {i}\n{'=' * 70}")
        print(f"Question: {item['question']}")
        print(f"Gold answer: {item['answer']}")
        print(f"(passages given: {len(item['passages'])})")

        result = session.run(item["question"], item["passages"])

        # Print the FULL transcript, not just turns where code executed --
        # a turn where the model wrote no code at all (silently nudged,
        # previously invisible) is exactly the kind of thing that needs
        # to be visible to diagnose a run that goes quiet.
        for msg in result.transcript[2:]:  # skip the fixed system+question preamble
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                print(f"\n--- model response ---\n{content.strip()}")
            else:
                print(f"\n--- observation shown to model ---\n{content.strip()[:600]}")

        if result.hit_max_turns:
            print(f"\n[DID NOT FINISH] hit max_turns={args.max_turns} without calling final_answer()")
            em, f1 = False, 0.0
        else:
            pred = extract_qa_answer(result.answer or "")
            em = exact_match_score(pred, item["answer"])
            f1 = f1_score(pred, item["answer"])
            print(f"\nFinal answer: {result.answer!r} -> extracted: {pred!r} | EM={em} F1={f1:.2f}")

        print(f"llm_query() calls made: {result.llm_query_calls}")
        n_correct += int(em)

    print(f"\n{'=' * 70}\nSUMMARY (n={len(data)}, NOT statistically powered)\n{'=' * 70}")
    print(f"  {n_correct}/{len(data)} finished with an exact-match answer")


if __name__ == "__main__":
    main()
