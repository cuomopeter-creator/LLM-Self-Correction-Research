from __future__ import annotations
import argparse
import time
import yaml
from typing import Any, Tuple

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from logger import JSONLLogger, RunMeta, make_run_id
from data.loaders import load_gsm8k

from models.huggingface_model import HuggingFaceModel, HFModelConfig
from models.openai_model import OpenAIModel, OpenAIModelConfig
from models.anthropic_model import AnthropicModel, AnthropicModelConfig
from models.fireworks_model import FireworksModel, FireworksModelConfig


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
    p = argparse.ArgumentParser(description="Run GSM8K evaluation harness.")
    p.add_argument(
        "--model",
        default="llama3_1_8b_instruct",
        help="Model key from configs/models.yaml under `models:`",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of GSM8K examples to run.",
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
            strategy_name="single_pass",
            task_name="gsm8k_test",
        ),
    )

    for ex in load_gsm8k(split="test", limit=args.limit):
        t0 = time.time()

        provider = model_cfg.get("provider")
        if provider == "openai":
            raw = model.generate(
                ex.prompt,
                max_output_tokens=model_cfg.get("max_output_tokens", 128),
            )
        elif provider == "anthropic":
            raw = model.generate(
                ex.prompt,
                max_tokens=model_cfg.get("max_tokens", 128),
            )
        elif provider == "fireworks":
            raw = model.generate(
                ex.prompt,
                max_tokens=model_cfg.get("max_tokens", model_cfg.get("max_output_tokens", 256)),
            )
        else:
            raw = model.generate(
                ex.prompt,
                max_new_tokens=model_cfg.get("max_new_tokens", 128),
            )

        latency = time.time() - t0
        out, usage = unwrap_generation_result(raw)

        print(f"ID: {ex.id}")
        print(f"OUTPUT: {out}")
        if usage:
            print(f"USAGE: {usage}")
        print("-" * 60)

        logger.log(
            example_id=ex.id,
            prompt=ex.prompt,
            output=out,
            latency_s=latency,
            extra={
                "gold": ex.answer,
                "usage": usage,
            },
        )

    print(f"Logged to: {run_dir}/results.jsonl")


if __name__ == "__main__":
    main()
