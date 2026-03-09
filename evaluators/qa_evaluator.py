from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class QAEvalResult:
    correct: bool
    pred: str
    gold: str


def _normalize_choice(text: str) -> str:
    s = str(text).strip().upper()

    # handle plain A/B/C/D
    m = re.search(r'\b([A-D])\b', s)
    if m:
        return m.group(1)

    # handle formats like "(A)" or "Answer: B"
    m = re.search(r'ANSWER\s*[:\-]?\s*([A-D])', s)
    if m:
        return m.group(1)

    m = re.search(r'\(([A-D])\)', s)
    if m:
        return m.group(1)

    return s


def evaluate_qa(output: str, gold: str) -> QAEvalResult:
    pred = _normalize_choice(output)
    gold_norm = _normalize_choice(gold)

    return QAEvalResult(
        correct=(pred == gold_norm),
        pred=pred,
        gold=gold_norm,
    )
