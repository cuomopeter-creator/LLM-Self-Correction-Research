from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import os

from anthropic import Anthropic

from models.base import BaseModel


@dataclass
class AnthropicModelConfig:
    model: str
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 256
    temperature: float = 0.0


class AnthropicModel(BaseModel):
    def __init__(self, cfg: AnthropicModelConfig):
        self.cfg = cfg
        api_key = os.environ.get(cfg.api_key_env)

        if not api_key:
            raise RuntimeError(f"Missing API key in env var: {cfg.api_key_env}")

        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, **kwargs: Any) -> dict:
        max_tokens = int(kwargs.get("max_tokens", self.cfg.max_tokens))
        temperature = float(kwargs.get("temperature", self.cfg.temperature))

        resp = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        text = resp.content[0].text.strip()

        usage_raw = getattr(resp, "usage", None)

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        if usage_raw is not None:
            input_tokens = getattr(usage_raw, "input_tokens", 0)
            output_tokens = getattr(usage_raw, "output_tokens", 0)
            total_tokens = input_tokens + output_tokens

        return {
            "text": text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }
