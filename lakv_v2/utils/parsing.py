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
    # 1. Primary path: try to find explicit #### delim
    match = re.search(r"####\s*\$?(-?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
        
    # 2. Fallback path: last found number
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None
