from __future__ import annotations
from typing import Any, Dict


def run_single_pass(
    *,
    model,
    prompt: str,
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    S0 baseline:
    Generate exactly one answer with no refinement.
    """

    provider = model_cfg.get("provider")

    if provider == "openai":
        output = model.generate(
            prompt,
            max_output_tokens=model_cfg.get("max_output_tokens", 256),
        )
    elif provider in {"anthropic", "fireworks"}:
        output = model.generate(
            prompt,
            max_tokens=model_cfg.get("max_tokens", 256),
        )
    else:
        output = model.generate(
            prompt,
            max_new_tokens=model_cfg.get("max_new_tokens", 256),
        )

    return {
        "final_output": output,
        "all_outputs": [output],
        "intermediate_steps": [],
        "strategy_meta": {
            "strategy_name": "single_pass",
            "n_generations": 1,
            "n_refinement_steps": 0,
        },
    }
