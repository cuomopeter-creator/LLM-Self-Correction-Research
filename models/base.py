from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseModel(ABC):
    """Minimal interface every model (HF/API/etc.) must implement."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError
