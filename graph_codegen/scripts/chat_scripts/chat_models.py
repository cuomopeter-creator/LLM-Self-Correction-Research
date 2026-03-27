from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from dotenv import load_dotenv

from models.huggingface_model import HuggingFaceModel, HFModelConfig
from models.openai_model import OpenAIModel, OpenAIModelConfig
from models.anthropic_model import AnthropicModel, AnthropicModelConfig
from models.fireworks_model import FireworksModel, FireworksModelConfig

load_dotenv(PROJECT_ROOT / ".env", override=True)


def load_yaml(path: str | Path) -> dict:
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
                max_new_tokens=model_cfg.get("max_new_tokens", 512),
                temperature=model_cfg.get("temperature", 0.0),
                top_p=model_cfg.get("top_p", 1.0),
                adapter_path=model_cfg.get("adapter_path"),
            )
        )

    if provider == "openai":
        return OpenAIModel(
            OpenAIModelConfig(
                model=model_cfg["model"],
                api_key_env=model_cfg.get("api_key_env", "OPENAI_API_KEY"),
                max_output_tokens=model_cfg.get("max_output_tokens", 512),
                temperature=model_cfg.get("temperature", 0.0),
            )
        )

    if provider == "anthropic":
        return AnthropicModel(
            AnthropicModelConfig(
                model=model_cfg["model"],
                api_key_env=model_cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
                max_tokens=model_cfg.get("max_tokens", 512),
                temperature=model_cfg.get("temperature", 0.0),
            )
        )

    if provider == "fireworks":
        return FireworksModel(
            FireworksModelConfig(
                model=model_cfg["model"],
                api_key_env=model_cfg.get("api_key_env", "FIREWORKS_API_KEY"),
                max_tokens=model_cfg.get("max_tokens", model_cfg.get("max_output_tokens", 512)),
                temperature=model_cfg.get("temperature", 0.0),
            )
        )

    raise ValueError(f"Unknown provider: {provider}")


def generate_with_model(model, model_cfg: dict, prompt: str) -> dict:
    provider = model_cfg.get("provider")

    if provider == "openai":
        return model.generate(prompt, max_output_tokens=model_cfg.get("max_output_tokens", 512))

    if provider == "anthropic":
        return model.generate(prompt, max_tokens=model_cfg.get("max_tokens", 512))

    if provider == "fireworks":
        return model.generate(
            prompt,
            max_tokens=model_cfg.get("max_tokens", model_cfg.get("max_output_tokens", 512)),
        )

    return model.generate(prompt, max_new_tokens=model_cfg.get("max_new_tokens", 512))
