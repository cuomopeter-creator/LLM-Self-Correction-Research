from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from models.base import BaseModel

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

@dataclass
class HFModelConfig:
    model_id: str
    dtype: str = "float16"
    device_map: str | dict[str, Any] = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    offload_folder: Optional[str] = None
    load_in_4bit: bool = False


class HuggingFaceModel(BaseModel):
    def __init__(self, cfg: HFModelConfig):
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

        kwargs: dict[str, Any] = {
            "device_map": cfg.device_map,
        }

        if cfg.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_DTYPE_MAP.get(cfg.dtype, torch.float16),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            kwargs["dtype"] = _DTYPE_MAP.get(cfg.dtype, torch.float16)

        if cfg.offload_folder:
            kwargs["offload_folder"] = cfg.offload_folder

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        max_new = int(kwargs.get("max_new_tokens", self.cfg.max_new_tokens))

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
        )

        text = self.tokenizer.decode(gen[0], skip_special_tokens=True)
        return text[len(prompt):].strip()
