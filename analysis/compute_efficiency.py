from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.compute_metrics import load_all_examples


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "analysis" / "run_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "compute_efficiency.csv"


def _latest_manifest_runs(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path).copy()
    manifest["run_dir"] = manifest["run_dir"].astype(str)
    manifest["n_examples"] = pd.to_numeric(manifest["n_examples"], errors="coerce")
    latest = (
        manifest.sort_values(["model", "task", "strategy", "n_examples", "run_dir"])
        .groupby(["model", "task", "strategy", "n_examples"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def _prepare_examples(manifest_path: Path) -> pd.DataFrame:
    latest = _latest_manifest_runs(manifest_path)
    run_names = latest["run_dir"].astype(str).tolist()
    df = load_all_examples(run_names=run_names).copy()

    numeric_cols = [
        "score_int",
        "usage_total_total_tokens",
        "usage_total_tokens",
        "usage_input_tokens",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    token_col = None
    for candidate in [
        "usage_total_total_tokens",
        "usage_total_tokens",
        "usage_input_tokens",
    ]:
        if candidate in df.columns:
            token_col = candidate
            break
    if token_col is None:
        raise ValueError("No token usage column found in results.")

    df["token_usage_for_metrics"] = df[token_col]
    return df


def _summarize_group(group: pd.DataFrame) -> dict[str, float | int]:
    total_examples = int(group["example_id"].count())
    total_correct = float(group["score_int"].sum())
    total_tokens = float(group["token_usage_for_metrics"].sum())
    accuracy_mean = float(group["score_int"].mean())
    token_usage_mean = float(group["token_usage_for_metrics"].mean())

    return {
        "n_examples": total_examples,
        "total_correct": total_correct,
        "total_tokens": total_tokens,
        "accuracy_mean": accuracy_mean,
        "token_usage_mean": token_usage_mean,
        "tokens_per_correct": (total_tokens / total_correct) if total_correct else pd.NA,
        "accuracy_per_1000_tokens": (total_correct / total_tokens) * 1000
        if total_tokens
        else pd.NA,
    }


def compute_efficiency_table(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_specs = [
        ("model_task", ["model", "task", "strategy"]),
        ("model", ["model", "strategy"]),
    ]

    for group_level, group_cols in group_specs:
        for keys, group in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)

            record: dict[str, object] = {"group_level": group_level, "task": pd.NA}
            for col, value in zip(group_cols, keys):
                record[col] = value

            record.update(_summarize_group(group))
            records.append(record)

    out = pd.DataFrame(records)
    out = out.sort_values(["group_level", "model", "task", "strategy"]).reset_index(
        drop=True
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute efficiency metrics from the current run manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to run_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _prepare_examples(args.manifest)
    out = compute_efficiency_table(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote compute efficiency summary to: {args.output}")


if __name__ == "__main__":
    main()
