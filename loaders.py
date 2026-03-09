from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional
import re

from datasets import load_dataset


@dataclass
class Example:
    id: str
    prompt: str
    answer: Optional[str] = None
    test: Optional[str] = None
    entry_point: Optional[str] = None
    canonical_solution: Optional[str] = None


def _gsm8k_extract_final(answer_text: Optional[str]) -> Optional[str]:
    if not answer_text:
        return None
    m = re.search(r"####\s*(-?\d+)", answer_text)
    if m:
        return m.group(1)
    return answer_text.strip().splitlines()[-1].strip()


def load_gsm8k(split: str = "test", limit: Optional[int] = 25) -> Iterator[Example]:
    ds = load_dataset("gsm8k", "main", split=split, cache_dir="data/gsm8k")

    n = 0
    for row in ds:
        q = row["question"].strip()
        gold = _gsm8k_extract_final(row.get("answer"))

        yield Example(
            id=str(n),
            prompt=(
                "Solve this math word problem. "
                "Give ONLY the final numeric answer.\n\n"
                f"{q}\n\nAnswer:"
            ),
            answer=gold,
        )

        n += 1
        if limit is not None and n >= limit:
            break


def load_truthfulqa(limit: Optional[int] = 25) -> Iterator[Example]:
    ds = load_dataset(
        "truthful_qa",
        "multiple_choice",
        split="validation",
        cache_dir="data/truthfulqa",
    )

    n = 0
    for row in ds:
        question = row["question"].strip()
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        gold = chr(65 + labels.index(1))

        choice_block = "\n".join(
            f"{chr(65+i)}. {choice.strip()}" for i, choice in enumerate(choices)
        )

        yield Example(
            id=str(n),
            prompt=(
                "Answer the multiple choice question. "
                "Return ONLY the letter of the best answer.\n\n"
                f"{question}\n\n{choice_block}\n\nAnswer:"
            ),
            answer=gold,
        )

        n += 1
        if limit is not None and n >= limit:
            break


def load_humaneval(limit: Optional[int] = 25) -> Iterator[Example]:
    ds = load_dataset(
        "openai_humaneval",
        split="test",
        cache_dir="data/humaneval",
    )

    n = 0
    for row in ds:
        prompt = row["prompt"]

        yield Example(
            id=row["task_id"],
            prompt=(
                "Complete the following Python function.\n"
                "Return ONLY the function body as executable Python code.\n"
                "Do NOT include the function signature.\n"
                "Do NOT include markdown, explanations, comments, code fences, or triple-quoted text.\n"
                "Do NOT wrap the answer in ``` fences, ''' quotes, or \"\"\" quotes.\n"
                "Formatting rules:\n"
                "- Start the first line at column 0.\n"
                "- Do NOT indent the entire answer.\n"
                "- Use 4 spaces only for lines inside nested blocks like if/for/while/try.\n"
                "- Return only the body statements.\n\n"
                f"{prompt}"
            ),
            test=row["test"],
            entry_point=row["entry_point"],
            canonical_solution=row["canonical_solution"],
        )

        n += 1
        if limit is not None and n >= limit:
            break


def load_arc(limit: Optional[int] = 25) -> Iterator[Example]:
    ds = load_dataset(
        "ai2_arc",
        "ARC-Challenge",
        split="test",
        cache_dir="data/arc_challenge",
    )

    n = 0
    for row in ds:
        question = row["question"]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]

        options = "\n".join(
            f"{label}. {text}" for label, text in zip(labels, choices)
        )

        yield Example(
            id=str(n),
            prompt=(
                "Answer the multiple choice question. "
                "Return ONLY the letter of the best answer.\n\n"
                f"{question}\n\n"
                f"{options}\n\n"
                "Answer:"
            ),
            answer=row["answerKey"],
        )

        n += 1
        if limit is not None and n >= limit:
            break
