from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_results_jsonl_flat(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num}: {e}") from e

    records: list[dict[str, Any]] = []

    for row_idx, r in enumerate(rows):
        extra = r.get("extra", {}) or {}
        usage = extra.get("usage", {}) or {}
        usage_total = extra.get("usage_total", {}) or {}
        strategy_meta = extra.get("strategy_meta", {}) or {}
        intermediate_steps = extra.get("intermediate_steps", []) or []
        all_outputs = extra.get("all_outputs", []) or []

        records.append({
            "source_file": str(path),
            "row_idx": row_idx,
            "ts_utc": r.get("ts_utc"),
            "example_id": r.get("example_id"),
            "prompt": r.get("prompt"),
            "output": r.get("output"),
            "latency_s": r.get("latency_s"),
            "score": r.get("score"),
            "score_int": int(r["score"]) if isinstance(r.get("score"), bool) else pd.NA,

            "gold": extra.get("gold"),
            "gold_norm": extra.get("gold_norm"),
            "pred": extra.get("pred"),
            "entry_point": extra.get("entry_point"),

            "strategy_name": strategy_meta.get("strategy_name"),
            "n_generations": strategy_meta.get("n_generations"),
            "n_refinement_steps": strategy_meta.get("n_refinement_steps"),

            "usage_input_tokens": usage.get("input_tokens"),
            "usage_output_tokens": usage.get("output_tokens"),
            "usage_total_tokens": usage.get("total_tokens"),

            "usage_total_input_tokens": usage_total.get("input_tokens"),
            "usage_total_output_tokens": usage_total.get("output_tokens"),
            "usage_total_total_tokens": usage_total.get("total_tokens"),

            "n_intermediate_steps_logged": len(intermediate_steps),
            "n_all_outputs_logged": len(all_outputs),

            # optional: keep raw nested data in case you need it later
            "intermediate_steps_json": json.dumps(intermediate_steps, ensure_ascii=False),
            "all_outputs_json": json.dumps(all_outputs, ensure_ascii=False),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_results_jsonl_flat("results.jsonl")
    print(df.head())
    #print("\nAccuracy:", df["score_int"].mean())
    df.to_csv("results_flat.csv", index=False)
