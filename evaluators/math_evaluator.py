from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class MathEvalResult:
    correct: bool
    pred: str
    gold: str


def _normalize_num(text: str) -> str:
    t = str(text).strip()
    t = t.replace(",", "")
    t = t.replace("$", "")
    t = t.strip()

    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        try:
            n = float(t)
            if n.is_integer():
                return str(int(n))
            return str(n).rstrip("0").rstrip(".")
        except Exception:
            pass

    return t


def extract_math_answer(text: str) -> str:
    s = str(text).strip()

    # GSM8K format: #### 42
    matches = re.findall(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)", s)
    if matches:
        return _normalize_num(matches[-1])

    # fallback: last number in output
    matches = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", s)
    if matches:
        return _normalize_num(matches[-1])

    return _normalize_num(s)


def evaluate_math(output: str, gold: str) -> MathEvalResult:
    pred = extract_math_answer(output)
    gold_norm = extract_math_answer(gold)

    return MathEvalResult(
        correct=(pred == gold_norm),
        pred=pred,
        gold=gold_norm,
    )
