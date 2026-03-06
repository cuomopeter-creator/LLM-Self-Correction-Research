from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import os

from openai import OpenAI
from models.base import BaseModel


@dataclass
class OpenAIModelConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_output_tokens: int = 256
    temperature: float = 0.0


class OpenAIModel(BaseModel):
    def __init__(self, cfg: OpenAIModelConfig):
        self.cfg = cfg
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env var: {cfg.api_key_env}")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, **kwargs: Any) -> dict:
        max_tokens = int(kwargs.get("max_output_tokens", self.cfg.max_output_tokens))
        temperature = float(kwargs.get("temperature", self.cfg.temperature))

        params: dict[str, Any] = {
            "model": self.cfg.model,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }

        # gpt-5 family rejects temperature, so don't send it there
        unsupported_sampling_prefixes = ("o1", "o3", "o4", "gpt-5")
        if not self.cfg.model.startswith(unsupported_sampling_prefixes):
            params["temperature"] = temperature

        resp = self.client.responses.create(**params)

        text = (resp.output_text or "").strip()
        if not text:
            status = getattr(resp, "status", None)
            out_types = [getattr(o, "type", "unknown") for o in getattr(resp, "output", [])]
            text = f"[EMPTY_OUTPUT status={status} output_types={out_types}]"

        usage_raw = getattr(resp, "usage", None)

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        if usage_raw is not None:
            input_tokens = int(
                getattr(usage_raw, "input_tokens", 0)
                or (usage_raw.get("input_tokens", 0) if isinstance(usage_raw, dict) else 0)
            )
            output_tokens = int(
                getattr(usage_raw, "output_tokens", 0)
                or (usage_raw.get("output_tokens", 0) if isinstance(usage_raw, dict) else 0)
            )
            total_tokens = int(
                getattr(usage_raw, "total_tokens", 0)
                or (usage_raw.get("total_tokens", 0) if isinstance(usage_raw, dict) else 0)
            )

            if total_tokens == 0:
                total_tokens = input_tokens + output_tokens

        return {
            "text": text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }
