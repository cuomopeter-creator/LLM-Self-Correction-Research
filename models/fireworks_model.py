from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any

from fireworks.client import Fireworks
from models.base import BaseModel


@dataclass
class FireworksModelConfig:
    model: str
    api_key_env: str = "FIREWORKS_API_KEY"
    max_tokens: int = 256
    temperature: float = 0.0


class FireworksModel(BaseModel):
    def __init__(self, cfg: FireworksModelConfig):
        self.cfg = cfg
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env var: {cfg.api_key_env}")

        self.client = Fireworks(api_key=api_key)

    def generate(self, prompt: str, **kwargs: Any) -> dict:
        max_tokens = int(kwargs.get("max_tokens", self.cfg.max_tokens))
        temperature = float(kwargs.get("temperature", self.cfg.temperature))

        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        msg = resp.choices[0].message
        text = (msg.content or msg.reasoning_content or "").strip()

        usage_raw = getattr(resp, "usage", None)

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        if usage_raw is not None:
            input_tokens = getattr(usage_raw, "prompt_tokens", 0)
            output_tokens = getattr(usage_raw, "completion_tokens", 0)
            total_tokens = getattr(usage_raw, "total_tokens", input_tokens + output_tokens)

        return {
            "text": text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }
