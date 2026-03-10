from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class OracleFeedbackResult:
    initial_output: str
    final_output: str
    corrected: bool
    rounds_used: int
    feedback: Optional[str]
    initial_raw: Any
    final_raw: Any


def _default_revision_prompt(original_prompt: str, previous_output: str, feedback: str) -> str:
    return (
        f"{original_prompt}\n\n"
        f"Your previous answer was:\n{previous_output}\n\n"
        f"Oracle feedback: {feedback}\n\n"
        "Revise your answer accordingly.\n"
        "Return only the final answer."
    )


def run_oracle_feedback(
    model: Any,
    prompt: str,
    evaluator: Callable[[str], bool],
    *,
    max_output_tokens: int = 256,
    max_new_tokens: int = 256,
    revision_prompt_builder: Optional[Callable[[str, str, str], str]] = None,
) -> OracleFeedbackResult:
    gen_kwargs = {}
    if hasattr(model, "cfg") and hasattr(model.cfg, "model"):
        gen_kwargs["max_output_tokens"] = max_output_tokens
    else:
        gen_kwargs["max_new_tokens"] = max_new_tokens

    initial_raw = model.generate(prompt, **gen_kwargs)
    initial_output = initial_raw["text"].strip() if isinstance(initial_raw, dict) else str(initial_raw).strip()

    if evaluator(initial_output):
        return OracleFeedbackResult(
            initial_output=initial_output,
            final_output=initial_output,
            corrected=False,
            rounds_used=0,
            feedback=None,
            initial_raw=initial_raw,
            final_raw=initial_raw,
        )

    feedback = "Your previous answer is incorrect. Please try again and ensure you follow the original formatting instructions."
    builder = revision_prompt_builder or _default_revision_prompt
    revision_prompt = builder(prompt, initial_output, feedback)

    revised_raw = model.generate(revision_prompt, **gen_kwargs)
    revised_output = revised_raw["text"].strip() if isinstance(revised_raw, dict) else str(revised_raw).strip()

    return OracleFeedbackResult(
        initial_output=initial_output,
        final_output=revised_output,
        corrected=revised_output != initial_output,
        rounds_used=1,
        feedback=feedback,
        initial_raw=initial_raw,
        final_raw=revised_raw,
    )
