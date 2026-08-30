"""
QA scoring for free-text-answer datasets (HotpotQA), as opposed to GSM8K's
numeric-answer scoring in evaluator.py's extract_answer(). Kept separate
because the two answer formats need entirely different extraction and
comparison logic (short text span + EM/F1, vs. a final number).

normalize_answer/exact_match_score/f1_score follow the standard SQuAD/
HotpotQA evaluation convention.
"""

import re
import string
from typing import Optional


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, remove articles, collapse whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_qa_answer(text: str) -> Optional[str]:
    """Pull the answer span out of a finalizer's raw output.

    Priority: an explicit "The answer is: <span>" / "answer is/=/: <span>"
    marker (take the rest of that line) > the last non-empty line of the
    text. Mirrors extract_answer()'s cascading-marker style in evaluator.py,
    but text-based instead of numeric-regex-based.
    """
    if not text:
        return None

    match = re.search(r"(?i)answer\s*(?:is|=|:)?\s*[:\-]?\s*(.+)", text)
    if match:
        span = match.group(1).strip().strip(".").strip()
        if span:
            return span

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


def exact_match_score(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    return normalize_answer(pred) == normalize_answer(gold)


def f1_score(pred: Optional[str], gold: str) -> float:
    if pred is None:
        return 0.0
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    num_same = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in set(pred_tokens))
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
