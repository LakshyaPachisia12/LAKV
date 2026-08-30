"""
LAKV-v2 Module: Parsing Utilities
Extracts numeric solutions smoothly and safely for GSM8K.
"""

import re
from typing import Optional

def extract_answer(text: str) -> Optional[str]:
    """
    Robust extraction for `#### [number]` format.
    Ignores trailing periods, commas, and dollar signs securely.
    """
    if not text:
        return None

    cleaned = re.sub(r"(?<=\d)[^\d,\.\-]+(?=\d)", "", text)

    for candidate in (text, cleaned):
        match = re.search(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", candidate)
        if match:
            return match.group(1).replace(",", "").rstrip(".")

    for candidate in (text, cleaned):
        match = re.search(r"(?im)^\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\.?\s*$", candidate)
        if match:
            return match.group(1).replace(",", "").rstrip(".")

    for candidate in (text, cleaned):
        match = re.search(r"(?i)answer\s*(?:is|=|:)?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", candidate)
        if match:
            return match.group(1).replace(",", "").rstrip(".")

    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", cleaned)
    return numbers[-1].replace(",", "").rstrip(".") if numbers else None
