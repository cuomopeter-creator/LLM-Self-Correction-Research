from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional

from datasets import load_dataset


@dataclass
class Example:
    id: str
    prompt: str
    answer: Optional[str] = None


def load_gsm8k(split: str = "test", limit: Optional[int] = 25) -> Iterator[Example]:
    ds = load_dataset("gsm8k", "main", split=split)

    n = 0
    for row in ds:
        q = row["question"].strip()
        a = row.get("answer").split("####")[-1].strip()

        yield Example(
            id=str(n),
            prompt=f"Solve the math problem. Return ONLY the final number. No words, no units, no explanation.\n\n{q}\n\nAnswer:",
            answer=a,
        )

        n += 1
        if limit is not None and n >= limit:
            break
