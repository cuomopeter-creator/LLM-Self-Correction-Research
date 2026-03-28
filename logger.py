from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class RunMeta:
    run_id: str
    created_at_utc: float
    model_name: str
    model_cfg: Dict[str, Any]
    strategy_name: str = "single_pass"
    task_name: str = "smoke_test"


class JSONLLogger:
    def __init__(self, run_dir: str, meta: RunMeta):
        self.run_path = Path(run_dir)
        self.run_path.mkdir(parents=True, exist_ok=True)

        self.meta = meta
        (self.run_path / "meta.json").write_text(
            json.dumps(asdict(meta), indent=2),
            encoding="utf-8",
        )

        self.results_path = self.run_path / "results.jsonl"

    def log(
        self,
        example_id: str,
        prompt: str,
        output: str,
        *,
        latency_s: float,
        score: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec: Dict[str, Any] = {
            "ts_utc": time.time(),
            "example_id": example_id,
            "prompt": prompt,
            "output": output,
            "latency_s": latency_s,
            "score": score,
            "extra": _json_safe(extra or {}),
        }
        with self.results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_run_id(prefix: str = "run") -> str:
    # e.g., run_20260227_142300
    return time.strftime(f"{prefix}_%Y%m%d_%H%M%S", time.gmtime())
