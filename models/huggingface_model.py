from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.base import BaseModel


@dataclass
class HFModelConfig:
    model_id: str
    dtype: str = "float16"     # "float16" | "bfloat16" | "float32"
    device_map: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    adapter_path: str | None = None


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class HuggingFaceModel(BaseModel):
    def __init__(self, cfg: HFModelConfig):
        self.cfg = cfg
        if cfg.dtype not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {cfg.dtype}. Choose from {list(_DTYPE_MAP)}")

        tok_source = cfg.adapter_path if cfg.adapter_path else cfg.model_id
        self.tok = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=_DTYPE_MAP[cfg.dtype],
            device_map=cfg.device_map,
        )

        if cfg.adapter_path:
            self.model = PeftModel.from_pretrained(self.model, cfg.adapter_path)

        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, **kwargs: Any) -> str:
        max_new_tokens = int(kwargs.get("max_new_tokens", self.cfg.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.cfg.temperature))
        top_p = float(kwargs.get("top_p", self.cfg.top_p))

        messages = [{"role": "user", "content": prompt}]
        text_in = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tok(text_in, return_tensors="pt", padding=True)

        if hasattr(self.model, "device") and self.model.device is not None:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "device"):
            inputs = {k: v.to(self.model.base_model.device) for k, v in inputs.items()}

        gen = self.model.generate(
            use_cache=True,
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        input_len = int(inputs["input_ids"].shape[1])
        new_tokens = gen[0][input_len:]
        output_tokens = int(new_tokens.shape[0])
        text = self.tok.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "text": text,
            "usage": {
                "input_tokens": input_len,
                "output_tokens": output_tokens,
                "total_tokens": int(input_len + output_tokens),
            },
        }
