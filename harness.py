from __future__ import annotations
import argparse
import time
import yaml
from typing import Any, Tuple

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from logger import JSONLLogger, RunMeta, make_run_id
from data.loaders import load_gsm8k, load_truthfulqa, load_humaneval, load_arc
from evaluators.code_evaluator import evaluate_humaneval
from evaluators.math_evaluator import evaluate_math, oracle_math_correct
from evaluators.qa_evaluator import evaluate_qa

from models.huggingface_model import HuggingFaceModel, HFModelConfig
from models.openai_model import OpenAIModel, OpenAIModelConfig
from models.anthropic_model import AnthropicModel, AnthropicModelConfig
from models.fireworks_model import FireworksModel, FireworksModelConfig
from strategies.single_pass import run_single_pass
from strategies.best_of_n import run_best_of_n
from strategies.self_refine import run_self_refine
from strategies.oracle_feedback import run_oracle_feedback


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model(model_cfg: dict):
    provider = model_cfg.get("provider")
    if provider == "huggingface":
        return HuggingFaceModel(
            HFModelConfig(
                model_id=model_cfg["model_id"],
                dtype=model_cfg.get("dtype", "float16"),
                device_map=model_cfg.get("device_map", "auto"),
                max_new_tokens=model_cfg.get("max_new_tokens", 128),
                temperature=model_cfg.get("temperature", 0.0),
                top_p=model_cfg.get("top_p", 1.0),
            )
        )

    if provider == "openai":
        return OpenAIModel(
            OpenAIModelConfig(
                model=model_cfg["model"],
                api_key_env=model_cfg.get("api_key_env", "OPENAI_API_KEY"),
                max_output_tokens=model_cfg.get("max_output_tokens", 128),
                temperature=model_cfg.get("temperature", 0.0),
            )
        )

    if provider == "anthropic":
        return AnthropicModel(
            AnthropicModelConfig(
                model=model_cfg["model"],
                api_key_env=model_cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
                max_tokens=model_cfg.get("max_tokens", 128),
                temperature=model_cfg.get("temperature", 0.0),
            )
        )

    if provider == "fireworks":
        return FireworksModel(
            FireworksModelConfig(
                model=model_cfg["model"],
                api_key_env=model_cfg.get("api_key_env", "FIREWORKS_API_KEY"),
                max_tokens=model_cfg.get("max_tokens", model_cfg.get("max_output_tokens", 256)),
                temperature=model_cfg.get("temperature", 0.0),
            )
        )

    raise ValueError(f"Unknown provider: {provider}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run evaluation harness.")

    p.add_argument(
        "--model",
        default="llama",
        help="Model key from configs/models.yaml under `models:`",
    )

    p.add_argument(
        "--task",
        default="gsm8k",
        choices=["gsm8k", "truthfulqa", "humaneval", "arc"],
        help="Task to run.",
    )

    p.add_argument(
        "--strategy",
        default="single_pass",
        choices=["single_pass", "best_of_n", "self_refine", "oracle"],
        help="Strategy to use.",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of examples to run.",
    )

    return p.parse_args()


def unwrap_generation_result(result: Any) -> Tuple[str, dict]:
    if isinstance(result, str):
        return result, {}

    if isinstance(result, dict):
        text = str(result.get("text", "")).strip()
        usage = result.get("usage", {}) or {}
        return text, usage

    return str(result).strip(), {}


def sum_usage(outputs: list[Any]) -> dict:
    total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    for item in outputs:
        if isinstance(item, dict):
            usage = item.get("usage", {}) or {}
            total["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            total["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            total["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

    return total


def load_examples(task: str, limit: int):
    if task == "gsm8k":
        return load_gsm8k(split="test", limit=limit)
    if task == "truthfulqa":
        return load_truthfulqa(limit=limit)
    if task == "humaneval":
        return load_humaneval(limit=limit)
    if task == "arc":
        return load_arc(limit=limit)
    raise ValueError(f"Unknown task: {task}")


def generate_with_model(model, model_cfg: dict, prompt: str):
    provider = model_cfg.get("provider")
    if provider == "openai":
        return model.generate(
            prompt,
            max_output_tokens=model_cfg.get("max_output_tokens", 128),
        )
    if provider == "anthropic":
        return model.generate(
            prompt,
            max_tokens=model_cfg.get("max_tokens", 128),
        )
    if provider == "fireworks":
        return model.generate(
            prompt,
            max_tokens=model_cfg.get("max_tokens", model_cfg.get("max_output_tokens", 256)),
        )
    return model.generate(
        prompt,
        max_new_tokens=model_cfg.get("max_new_tokens", 128),
    )


def main():
    args = parse_args()

    cfg = load_yaml("configs/models.yaml")
    model_key = args.model
    model_cfg = cfg["models"][model_key]

    model = build_model(model_cfg)

    run_id = make_run_id()
    run_dir = f"runs/{run_id}"
    logger = JSONLLogger(
        run_dir=run_dir,
        meta=RunMeta(
            run_id=run_id,
            created_at_utc=time.time(),
            model_name=model_key,
            model_cfg=model_cfg,
            strategy_name=args.strategy,
            task_name=args.task,
        ),
    )

    for ex in load_examples(args.task, args.limit):
        t0 = time.time()
        if args.strategy == "single_pass":
            result = run_single_pass(
                model=model,
                prompt=ex.prompt,
                model_cfg=model_cfg,
            )
        elif args.strategy == "best_of_n":
            result = run_best_of_n(
                model=model,
                prompt=ex.prompt,
                model_cfg=model_cfg,
                n=3,
            )
        elif args.strategy == "self_refine":
            result = run_self_refine(
                model=model,
                prompt=ex.prompt,
                model_cfg=model_cfg,
            )
        elif args.strategy == "oracle":
            if args.task == "gsm8k":
                evaluator = lambda output, gold=ex.answer: oracle_math_correct(output, gold)
            elif args.task == "arc":
                evaluator = lambda output, gold=ex.answer: evaluate_qa(output, gold).correct
            elif args.task == "truthfulqa":
                evaluator = lambda output, gold=ex.answer: evaluate_qa(output, gold).correct
            else:
                raise ValueError(f"Oracle strategy not yet supported for task: {args.task}")

            oracle_res = run_oracle_feedback(
                model=model,
                prompt=ex.prompt,
                evaluator=evaluator,
                max_output_tokens=model_cfg.get("max_output_tokens", 128),
                max_new_tokens=model_cfg.get("max_new_tokens", 128),
            )

            result = {
                "final_output": oracle_res.final_output,
                "all_outputs": [
                    oracle_res.initial_raw,
                    oracle_res.final_raw,
                ],
                "intermediate_steps": [
                    {
                        "initial_output": oracle_res.initial_output,
                        "feedback": oracle_res.feedback,
                        "rounds_used": oracle_res.rounds_used,
                    }
                ],
                "strategy_meta": {
                    "corrected": oracle_res.corrected,
                    "rounds_used": oracle_res.rounds_used,
                },
            }
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
        latency = time.time() - t0

        out, usage = unwrap_generation_result(result["final_output"])
        usage_total = sum_usage(result.get("all_outputs", []))
        score = None
        eval_stdout = ""
        eval_stderr = ""
        pred = ""
        gold_norm = ""

        if args.task == "humaneval":
            try:
                res = evaluate_humaneval(
                    prompt=ex.prompt,
                    completion=out,
                    test_code=ex.test,
                    entry_point=ex.entry_point,
                )
                score = res.passed
                eval_stdout = res.stdout
                eval_stderr = res.stderr
            except Exception as e:
                score = False
                eval_stderr = str(e)

        elif args.task == "gsm8k":
            try:
                res = evaluate_math(out, ex.answer)
                score = res.correct
                pred = res.pred
                gold_norm = res.gold
            except Exception as e:
                score = False
                eval_stderr = str(e)

        elif args.task in ("arc", "truthfulqa"):
            try:
                res = evaluate_qa(out, ex.answer)
                score = res.correct
                pred = res.pred
                gold_norm = res.gold
            except Exception as e:
                score = False
                eval_stderr = str(e)

        print(f"ID: {ex.id}")
        print(f"OUTPUT: {out}")
        if usage:
            print(f"USAGE: {usage}")
        if args.task in ("gsm8k", "arc", "truthfulqa"):
            print(f"PRED: {pred}")
            print(f"GOLD: {gold_norm}")
            print(f"CORRECT: {score}")
        print("-" * 60)

        logger.log(
            example_id=ex.id,
            prompt=ex.prompt,
            output=out,
            latency_s=latency,
            score=score,
            extra={
                "gold": ex.answer,
                "gold_norm": gold_norm,
                "pred": pred,
                "usage": usage,
                "usage_total": usage_total,
                "entry_point": getattr(ex, "entry_point", None),
                "eval_stdout": eval_stdout,
                "eval_stderr": eval_stderr,
                "all_outputs": result.get("all_outputs", []),
                "intermediate_steps": result.get("intermediate_steps", []),
                "strategy_meta": result.get("strategy_meta", {}),
            },
        )

    print(f"Logged to: {run_dir}/results.jsonl")


if __name__ == "__main__":
    main()


