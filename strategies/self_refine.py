from __future__ import annotations
from typing import Any, Dict


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


def _build_critique_prompt(original_prompt: str, draft: str) -> str:
    return "\n".join([
        "You are reviewing a draft answer for the ORIGINAL TASK.",
        "",
        "Your job is to detect errors or instruction violations.",
        "",
        "IMPORTANT RULES:",
        "- Do NOT solve the task.",
        "- Do NOT provide step-by-step reasoning.",
        "- Keep the explanation to ONE sentence.",
        "",
        "FORMAT YOUR RESPONSE EXACTLY AS:",
        "ERROR_TYPE: <format | reasoning | none>",
        "EXPLANATION: <one short sentence>",
        "",
        "ORIGINAL TASK:",
        original_prompt,
        "",
        "DRAFT ANSWER:",
        draft,
        "",
        "CRITIQUE:",
    ])


def _build_refine_prompt(original_prompt: str, draft: str, critique: str) -> str:
    return "\n".join([
        "Revise the draft answer using the critique.",
        "",
        "If the draft already appears correct and follows the required output format,",
        "return the draft unchanged.",
        "",
        "Your revised answer must follow the ORIGINAL TASK exactly.",
        "Return ONLY the revised final answer.",
        "",
        "ORIGINAL TASK:",
        original_prompt,
        "",
        "DRAFT ANSWER:",
        draft,
        "",
        "CRITIQUE:",
        critique,
        "",
        "REVISED ANSWER:",
    ])


def run_self_refine(
    *,
    model,
    prompt: str,
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    raw_draft = _generate_once(
        model=model,
        prompt=prompt,
        model_cfg=model_cfg,
        temperature_override=0.7,
    )
    draft = _unwrap_text(raw_draft)

    critique_prompt = _build_critique_prompt(prompt, draft)
    raw_critique = _generate_once(
        model=model,
        prompt=critique_prompt,
        model_cfg=model_cfg,
        temperature_override=0.0,
    )
    critique = _unwrap_text(raw_critique)

    refine_prompt = _build_refine_prompt(prompt, draft, critique)
    raw_final = _generate_once(
        model=model,
        prompt=refine_prompt,
        model_cfg=model_cfg,
        temperature_override=0.0,
    )

    return {
        "final_output": raw_final,
        "all_outputs": [raw_draft, raw_critique, raw_final],
        "intermediate_steps": [
            {
                "step": "initial_draft",
                "draft": draft,
                "generation_temperature": 0.7,
            },
            {
                "step": "self_critique",
                "critique_prompt": critique_prompt,
                "critique": critique,
                "critique_temperature": 0.0,
            },
            {
                "step": "self_refine",
                "refine_prompt": refine_prompt,
                "refine_temperature": 0.0,
            },
        ],
        "strategy_meta": {
            "strategy_name": "self_refine",
            "n_generations": 3,
            "n_refinement_steps": 1,
        },
    }
