from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.load_results import load_results_jsonl_flat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_MANIFEST = PROJECT_ROOT / "analysis" / "run_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "bootstrap_accuracy_cis.csv"

BASELINE_STRATEGY = "single_pass"
COMPARISON_STRATEGIES = ["best_of_n", "self_refine", "oracle"]


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


def _load_run_scores(run_dir: str) -> pd.DataFrame:
    df = load_results_jsonl_flat(RUNS_DIR / run_dir / "results.jsonl").copy()
    df["example_id"] = df["example_id"].astype(str)
    df["score_int"] = pd.to_numeric(df["score_int"], errors="coerce")
    return df[["example_id", "score_int"]]


def build_pairwise_rows(manifest_path: Path) -> pd.DataFrame:
    latest = _latest_manifest_runs(manifest_path)
    rows: list[pd.DataFrame] = []

    baseline_rows = latest[latest["strategy"] == BASELINE_STRATEGY]
    for _, base in baseline_rows.iterrows():
        for comparison_strategy in COMPARISON_STRATEGIES:
            comp = latest[
                (latest["model"] == base["model"])
                & (latest["task"] == base["task"])
                & (latest["n_examples"] == base["n_examples"])
                & (latest["strategy"] == comparison_strategy)
            ]
            if comp.empty:
                continue

            comp = comp.iloc[0]
            base_scores = _load_run_scores(str(base["run_dir"])).rename(
                columns={"score_int": "baseline_score"}
            )
            comp_scores = _load_run_scores(str(comp["run_dir"])).rename(
                columns={"score_int": "comparison_score"}
            )
            merged = base_scores.merge(comp_scores, on="example_id", how="inner")
            if merged.empty:
                continue

            merged["comparison"] = f"{comparison_strategy}_vs_{BASELINE_STRATEGY}"
            merged["baseline_strategy"] = BASELINE_STRATEGY
            merged["comparison_strategy"] = comparison_strategy
            merged["model"] = base["model"]
            merged["task"] = base["task"]
            merged["cohort_n_examples"] = int(base["n_examples"])
            merged["baseline_run_dir"] = base["run_dir"]
            merged["comparison_run_dir"] = comp["run_dir"]
            rows.append(merged)

    if not rows:
        raise ValueError("No comparable baseline/comparison runs found in manifest.")

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(
        ["comparison", "model", "task", "cohort_n_examples", "example_id"]
    ).reset_index(drop=True)
    return out


def _bootstrap_accuracy_diff(
    df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    baseline = df["baseline_score"].to_numpy(dtype=float)
    comparison = df["comparison_score"].to_numpy(dtype=float)
    n = len(df)
    rng = np.random.default_rng(seed)

    observed_baseline = float(baseline.mean())
    observed_comparison = float(comparison.mean())
    observed_diff = observed_comparison - observed_baseline

    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = comparison[idx].mean() - baseline[idx].mean()

    summary = {
        "n_examples": int(n),
        "baseline_accuracy": observed_baseline,
        "comparison_accuracy": observed_comparison,
        "accuracy_diff": observed_diff,
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
    }
    return summary, diffs


def summarize_bootstrap_cis(
    pairwise_df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    sample_records: list[pd.DataFrame] = []

    group_specs = [
        ("model_task", ["comparison", "model", "task"]),
        ("model", ["comparison", "model"]),
    ]

    for group_level, group_cols in group_specs:
        for keys, group in pairwise_df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record = {"group_level": group_level}
            for col, value in zip(group_cols, keys):
                record[col] = value
            if "task" not in record:
                record["task"] = pd.NA

            stats, diffs = _bootstrap_accuracy_diff(
                group,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            record.update(stats)
            record["ci_excludes_zero"] = bool(
                record["ci_lower"] > 0 or record["ci_upper"] < 0
            )
            records.append(record)

            sample_df = pd.DataFrame(
                {
                    "group_level": group_level,
                    "comparison": record["comparison"],
                    "model": record["model"],
                    "task": record["task"],
                    "sample_idx": np.arange(len(diffs)),
                    "accuracy_diff": diffs,
                }
            )
            sample_records.append(sample_df)

    out = pd.DataFrame(records)
    out = out.sort_values(["comparison", "group_level", "model", "task"]).reset_index(
        drop=True
    )
    samples = pd.concat(sample_records, ignore_index=True).sort_values(
        ["comparison", "group_level", "model", "task", "sample_idx"]
    ).reset_index(drop=True)
    return out, samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap 95% CIs for accuracy differences vs single-pass."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to run manifest CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "bootstrap_accuracy_ci_samples.csv",
        help="Path to output bootstrap sample CSV.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairwise_df = build_pairwise_rows(args.manifest)
    summary_df, samples_df = summarize_bootstrap_cis(
        pairwise_df,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    samples_df.to_csv(args.samples_output, index=False)
    print(f"Wrote bootstrap CI summary to: {args.output}")
    print(f"Wrote bootstrap CI samples to: {args.samples_output}")


if __name__ == "__main__":
    main()
