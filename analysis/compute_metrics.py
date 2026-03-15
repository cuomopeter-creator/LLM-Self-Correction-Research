from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.load_results import load_results_jsonl_flat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_meta_get(meta: dict[str, Any], *keys: str, default=None):
    cur: Any = meta
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _extract_run_metadata(run_dir: Path) -> dict[str, Any]:
    meta = _read_json(run_dir / "meta.json")

    model = (
        _safe_meta_get(meta, "model")
        or _safe_meta_get(meta, "model_name")
        or _safe_meta_get(meta, "config", "model")
        or _safe_meta_get(meta, "run", "model")
    )

    task = (
        _safe_meta_get(meta, "task")
        or _safe_meta_get(meta, "task_name")
        or _safe_meta_get(meta, "config", "task")
        or _safe_meta_get(meta, "run", "task")
    )

    strategy = (
        _safe_meta_get(meta, "strategy")
        or _safe_meta_get(meta, "strategy_name")
        or _safe_meta_get(meta, "config", "strategy")
        or _safe_meta_get(meta, "run", "strategy")
    )

    limit = (
        _safe_meta_get(meta, "limit")
        or _safe_meta_get(meta, "config", "limit")
        or _safe_meta_get(meta, "run", "limit")
    )

    return {
        "run_dir": run_dir.name,
        "meta_model": model,
        "meta_task": task,
        "meta_strategy": strategy,
        "meta_limit": limit,
    }


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_all_examples(
    runs_dir: str | Path = RUNS_DIR,
    min_results: int = 1,
    run_names: list[str] | None = None,
) -> pd.DataFrame:
    runs_dir = Path(runs_dir)
    requested_runs = set(run_names) if run_names else None

    all_dfs: list[pd.DataFrame] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        if not run_dir.name.startswith("run"):
            continue

        if requested_runs is not None and run_dir.name not in requested_runs:
            continue
            
        results_path = run_dir / "results.jsonl"
        if not results_path.exists():
            continue

        try:
            n_results = _count_jsonl_rows(results_path)
        except Exception as e:
            print(f"Skipping {run_dir.name}: failed counting rows ({e})")
            continue

        if n_results < min_results:
            print(
                f"Skipping {run_dir.name}: only {n_results} results "
                f"(minimum required: {min_results})"
            )
            continue

        try:
            df = load_results_jsonl_flat(results_path)
        except Exception as e:
            print(f"Skipping {run_dir.name}: failed to load results.jsonl ({e})")
            continue

        meta_info = _extract_run_metadata(run_dir)

        for key, value in meta_info.items():
            df[key] = value

        if "model" not in df.columns:
            df["model"] = pd.NA
        if "task" not in df.columns:
            df["task"] = pd.NA
        if "strategy" not in df.columns:
            df["strategy"] = pd.NA

        df["model"] = df["model"].fillna(df["meta_model"])
        df["task"] = df["task"].fillna(df["meta_task"])
        df["strategy"] = df["strategy"].fillna(df["meta_strategy"])

        if "strategy_name" in df.columns:
            df["strategy"] = df["strategy"].fillna(df["strategy_name"])

        df["n_results_in_run"] = n_results

        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No qualifying results.jsonl files found.")

    return pd.concat(all_dfs, ignore_index=True)


def compute_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    numeric_cols = [
        "score_int",
        "latency_s",
        "usage_total_total_tokens",
        "usage_total_input_tokens",
        "usage_total_output_tokens",
        "usage_total_tokens",
        "usage_input_tokens",
        "usage_output_tokens",
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    token_col = None
    for candidate in [
        "usage_total_total_tokens",
        "usage_total_tokens",
        "usage_input_tokens",
    ]:
        if candidate in work.columns:
            token_col = candidate
            break

    if token_col is None:
        work["token_usage_for_metrics"] = pd.NA
    else:
        work["token_usage_for_metrics"] = work[token_col]

    group_cols = ["model", "strategy", "task"]

    summary = (
        work.groupby(group_cols, dropna=False)
        .agg(
            n_examples=("example_id", "count"),
            n_runs=("run_dir", "nunique"),
            accuracy_mean=("score_int", "mean"),
            accuracy_median=("score_int", "median"),
            latency_mean_s=("latency_s", "mean"),
            latency_median_s=("latency_s", "median"),
            token_usage_mean=("token_usage_for_metrics", "mean"),
            token_usage_median=("token_usage_for_metrics", "median"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        by=["model", "task", "strategy"],
        na_position="last",
    ).reset_index(drop=True)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute summary metrics across run results."
    )
    parser.add_argument(
        "--min-results",
        type=int,
        default=1,
        help="Minimum number of JSONL rows required for a run to be included.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="One or more specific run folder names to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_examples = load_all_examples(
        RUNS_DIR,
        min_results=args.min_results,
        run_names=args.runs,
    )
    df_metrics = compute_metrics_table(df_examples)

    out_name = "metrics_summary.csv"
    if args.runs:
        out_name = "metrics_summary_selected_runs.csv"
    elif args.min_results > 1:
        out_name = f"metrics_summary_minresults_{args.min_results}.csv"

    out_path = PROJECT_ROOT / "analysis" / out_name
    df_metrics.to_csv(out_path, index=False)

    print("\nMetrics summary:")
    print(df_metrics.to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
