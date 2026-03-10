from __future__ import annotations
from typing import Any, Dict, List


def _generate_once(
    *,
    model,
    prompt: str,
    model_cfg: Dict[str, Any],
    temperature_override=None,
):
    provider = model_cfg.get("provider")
    temperature = model_cfg.get("temperature", 0.0)
    if temperature_override is not None:
        temperature = temperature_override

    if provider == "openai":
        return model.generate(
            prompt,
            max_output_tokens=model_cfg.get("max_output_tokens", 256),
            temperature=temperature,
        )
    elif provider in {"anthropic", "fireworks"}:
        return model.generate(
            prompt,
            max_tokens=model_cfg.get("max_tokens", 256),
            temperature=temperature,
        )
    else:
        return model.generate(
            prompt,
            max_new_tokens=model_cfg.get("max_new_tokens", 256),
            temperature=temperature,
        )


def _unwrap_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        return str(result.get("text", "")).strip()

    return str(result).strip()


def _build_selection_prompt(original_prompt: str, candidates: List[str]) -> str:
    parts = [
        "You are selecting the best answer candidate for the original task.",
        "",
        "Your goal is to choose the candidate that BEST follows the instructions of the ORIGINAL TASK.",
        "",
        "Important rules:",
        "1. If the task requests ONLY a final numeric answer, prefer the candidate that outputs ONLY the number with no explanation.",
        "2. Prefer answers that strictly follow the required output format.",
        "3. Prefer the candidate that is most likely to be correct.",
        "4. If multiple candidates appear equally correct, choose the one that follows the format most exactly.",
        "",
        "Return ONLY the number of the best candidate.",
        "",
        "ORIGINAL TASK:",
        original_prompt,
        "",
        "CANDIDATES:",
    ]

    for i, cand in enumerate(candidates, start=1):
        parts.append(f"[{i}]")
        parts.append(cand)
        parts.append("")

    parts.append("BEST CANDIDATE NUMBER:")
    return "\n".join(parts)


def _parse_choice(text: str, n: int) -> int:
    text = text.strip()

    for token in text.replace("[", " ").replace("]", " ").split():
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= n:
                return idx - 1

    for i in range(1, n + 1):
        if str(i) in text:
            return i - 1

    return 0


def run_best_of_n(
    *,
    model,
    prompt: str,
    model_cfg: Dict[str, Any],
    n: int = 3,
) -> Dict[str, Any]:
    """
    Generate N candidates at higher temperature, then ask the model
    to choose the best one deterministically.
    """

    candidates: List[str] = []
    raw_candidates: List[Any] = []

    for _ in range(n):
        raw = _generate_once(
            model=model,
            prompt=prompt,
            model_cfg=model_cfg,
            temperature_override=0.7,
        )
        raw_candidates.append(raw)
        candidates.append(_unwrap_text(raw))

    selection_prompt = _build_selection_prompt(prompt, candidates)
    judge_raw = _generate_once(
        model=model,
        prompt=selection_prompt,
        model_cfg=model_cfg,
        temperature_override=0.0,
    )
    judge_text = _unwrap_text(judge_raw)

    selected_idx = _parse_choice(judge_text, n)
    final_output = raw_candidates[selected_idx]

    return {
        "final_output": final_output,
        "all_outputs": raw_candidates,
        "intermediate_steps": [
            {
                "step": "candidate_generation",
                "candidates": candidates,
                "generation_temperature": 0.7,
            },
            {
                "step": "self_selection",
                "selection_prompt": selection_prompt,
                "judge_output": judge_text,
                "judge_temperature": 0.0,
                "selected_index": selected_idx,
                "selected_candidate": candidates[selected_idx],
            },
        ],
        "strategy_meta": {
            "strategy_name": "best_of_n",
            "n_generations": n,
            "n_refinement_steps": 1,
        },
    }
