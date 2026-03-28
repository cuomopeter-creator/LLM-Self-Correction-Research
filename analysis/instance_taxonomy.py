from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
MANIFEST_PATH = ANALYSIS_DIR / "run_manifest.csv"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.compute_metrics import load_all_examples


def _load_manifest(path: str | Path = MANIFEST_PATH) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"model", "task", "strategy", "run_dir"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return manifest


def _load_canonical_examples(manifest: pd.DataFrame) -> pd.DataFrame:
    selected_runs = manifest["run_dir"].dropna().astype(str).tolist()
    df = load_all_examples(run_names=selected_runs, min_results=1)

    keep_cols = [
        "run_dir",
        "model",
        "task",
        "strategy",
        "example_id",
        "score_int",
        "output",
        "pred",
        "gold",
        "latency_s",
        "usage_total_total_tokens",
        "usage_total_tokens",
        "usage_input_tokens",
        "usage_output_tokens",
        "all_outputs_json",
        "intermediate_steps_json",
    ]
    present = [c for c in keep_cols if c in df.columns]
    work = df[present].copy()
    work["example_id"] = work["example_id"].astype(str)
    work["score_int"] = pd.to_numeric(work["score_int"], errors="coerce")
    return work


def _label_transition(
    baseline_correct: float | int | None,
    comparison_correct: float | int | None,
) -> str:
    if pd.isna(baseline_correct) or pd.isna(comparison_correct):
        return "missing"

    b = int(baseline_correct)
    c = int(comparison_correct)
    if b == 1 and c == 1:
        return "correct_to_correct"
    if b == 0 and c == 1:
        return "incorrect_to_correct"
    if b == 1 and c == 0:
        return "correct_to_incorrect"
    return "incorrect_to_incorrect"


def build_pairwise_taxonomy(
    df_examples: pd.DataFrame,
    manifest: pd.DataFrame,
    baseline_strategy: str,
    comparison_strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_pair = manifest[manifest["strategy"].isin([baseline_strategy, comparison_strategy])]
    valid_pairs = (
        manifest_pair.groupby(["model", "task"])["strategy"]
        .nunique()
        .reset_index(name="n_strategies")
    )
    valid_pairs = valid_pairs[valid_pairs["n_strategies"] == 2][["model", "task"]]

    work = df_examples.merge(valid_pairs, on=["model", "task"], how="inner")

    baseline = work[work["strategy"] == baseline_strategy].copy()
    comparison = work[work["strategy"] == comparison_strategy].copy()

    baseline = baseline.rename(
        columns={
            "run_dir": "baseline_run_dir",
            "score_int": "baseline_score_int",
            "output": "baseline_output",
            "pred": "baseline_pred",
            "gold": "baseline_gold",
            "latency_s": "baseline_latency_s",
            "usage_total_total_tokens": "baseline_usage_total_total_tokens",
            "usage_total_tokens": "baseline_usage_total_tokens",
            "usage_input_tokens": "baseline_usage_input_tokens",
            "usage_output_tokens": "baseline_usage_output_tokens",
        }
    )
    comparison = comparison.rename(
        columns={
            "run_dir": "comparison_run_dir",
            "score_int": "comparison_score_int",
            "output": "comparison_output",
            "pred": "comparison_pred",
            "gold": "comparison_gold",
            "latency_s": "comparison_latency_s",
            "usage_total_total_tokens": "comparison_usage_total_total_tokens",
            "usage_total_tokens": "comparison_usage_total_tokens",
            "usage_input_tokens": "comparison_usage_input_tokens",
            "usage_output_tokens": "comparison_usage_output_tokens",
            "all_outputs_json": "comparison_all_outputs_json",
            "intermediate_steps_json": "comparison_intermediate_steps_json",
        }
    )

    merged = baseline.merge(
        comparison,
        on=["model", "task", "example_id"],
        how="inner",
        validate="one_to_one",
    )
    merged["baseline_strategy"] = baseline_strategy
    merged["comparison_strategy"] = comparison_strategy
    merged["transition_label"] = merged.apply(
        lambda row: _label_transition(
            row["baseline_score_int"],
            row["comparison_score_int"],
        ),
        axis=1,
    )

    summary = (
        merged.groupby(
            ["model", "task", "baseline_strategy", "comparison_strategy", "transition_label"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_examples")
    )

    totals = (
        merged.groupby(
            ["model", "task", "baseline_strategy", "comparison_strategy"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_total_examples")
    )

    summary = summary.merge(
        totals,
        on=["model", "task", "baseline_strategy", "comparison_strategy"],
        how="left",
    )
    summary["share_examples"] = summary["n_examples"] / summary["n_total_examples"]

    pivot = (
        summary.pivot_table(
            index=["model", "task", "baseline_strategy", "comparison_strategy", "n_total_examples"],
            columns="transition_label",
            values="n_examples",
            fill_value=0,
        )
        .reset_index()
    )

    for col in [
        "correct_to_correct",
        "incorrect_to_correct",
        "correct_to_incorrect",
        "incorrect_to_incorrect",
    ]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["revision_success_rate"] = (
        pivot["incorrect_to_correct"]
        / (pivot["incorrect_to_correct"] + pivot["incorrect_to_incorrect"])
    ).fillna(0.0)
    pivot["error_amplification_rate"] = (
        pivot["correct_to_incorrect"]
        / (pivot["correct_to_correct"] + pivot["correct_to_incorrect"])
    ).fillna(0.0)
    pivot["net_gain_examples"] = (
        pivot["incorrect_to_correct"] - pivot["correct_to_incorrect"]
    )
    pivot["net_gain_rate"] = pivot["net_gain_examples"] / pivot["n_total_examples"]

    pivot = pivot.sort_values(["model", "task"]).reset_index(drop=True)
    merged = merged.sort_values(["model", "task", "example_id"]).reset_index(drop=True)

    return merged, pivot


def build_self_refine_internal_taxonomy(
    df_examples: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_pairs = manifest[manifest["strategy"] == "self_refine"][["model", "task"]].drop_duplicates()
    work = df_examples.merge(valid_pairs, on=["model", "task"], how="inner")
    work = work[work["strategy"] == "self_refine"].copy()

    def extract_draft(row: pd.Series) -> str | None:
        raw = row.get("all_outputs_json")
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            items = json.loads(raw)
        except Exception:
            return None
        if not items:
            return None
        first = items[0]
        if isinstance(first, dict):
            return str(first.get("text", "")).strip()
        return str(first).strip()

    def score_like_final(row: pd.Series, text: str | None) -> float | int | None:
        if text is None:
            return pd.NA
        pred = row.get("pred")
        gold = row.get("gold")
        if pred is None or gold is None:
            return pd.NA
        pred = str(pred).strip()
        gold = str(gold).strip()
        text_norm = str(text).strip()

        # Match the simple exact/string-normalized nature of the stored outputs.
        if pred == gold:
            return 1 if text_norm == row.get("output", text_norm) else pd.NA
        return 0 if text_norm == row.get("output", text_norm) else pd.NA

    work["draft_output"] = work.apply(extract_draft, axis=1)
    work["draft_score_int"] = work.apply(
        lambda row: 1 if isinstance(row["draft_output"], str) and str(row.get("gold", "")).strip() and str(row["draft_output"]).strip() == str(row.get("gold", "")).strip()
        else (
            0 if isinstance(row["draft_output"], str) and str(row.get("gold", "")).strip()
            else pd.NA
        ),
        axis=1,
    )

    # Use draft exact-vs-gold comparison as a conservative internal estimate.
    work["baseline_strategy"] = "self_refine_draft"
    work["comparison_strategy"] = "self_refine_final"
    work["baseline_run_dir"] = work["run_dir"]
    work["comparison_run_dir"] = work["run_dir"]
    work["baseline_score_int"] = pd.to_numeric(work["draft_score_int"], errors="coerce")
    work["comparison_score_int"] = pd.to_numeric(work["score_int"], errors="coerce")
    work["baseline_output"] = work["draft_output"]
    work["comparison_output"] = work["output"]
    work["transition_label"] = work.apply(
        lambda row: _label_transition(row["baseline_score_int"], row["comparison_score_int"]),
        axis=1,
    )

    details = work[
        [
            "model",
            "task",
            "example_id",
            "baseline_strategy",
            "comparison_strategy",
            "baseline_run_dir",
            "comparison_run_dir",
            "baseline_score_int",
            "comparison_score_int",
            "baseline_output",
            "comparison_output",
            "transition_label",
            "gold",
            "pred",
            "run_dir",
        ]
    ].copy()

    summary = (
        details.groupby(
            ["model", "task", "baseline_strategy", "comparison_strategy", "transition_label"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_examples")
    )
    totals = (
        details.groupby(["model", "task", "baseline_strategy", "comparison_strategy"], dropna=False)
        .size()
        .reset_index(name="n_total_examples")
    )
    summary = summary.merge(
        totals,
        on=["model", "task", "baseline_strategy", "comparison_strategy"],
        how="left",
    )
    summary["share_examples"] = summary["n_examples"] / summary["n_total_examples"]

    pivot = (
        summary.pivot_table(
            index=["model", "task", "baseline_strategy", "comparison_strategy", "n_total_examples"],
            columns="transition_label",
            values="n_examples",
            fill_value=0,
        )
        .reset_index()
    )
    for col in [
        "correct_to_correct",
        "incorrect_to_correct",
        "correct_to_incorrect",
        "incorrect_to_incorrect",
        "missing",
    ]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["revision_success_rate"] = (
        pivot["incorrect_to_correct"]
        / (pivot["incorrect_to_correct"] + pivot["incorrect_to_incorrect"])
    ).fillna(0.0)
    pivot["error_amplification_rate"] = (
        pivot["correct_to_incorrect"]
        / (pivot["correct_to_correct"] + pivot["correct_to_incorrect"])
    ).fillna(0.0)
    pivot["net_gain_examples"] = pivot["incorrect_to_correct"] - pivot["correct_to_incorrect"]
    pivot["net_gain_rate"] = pivot["net_gain_examples"] / pivot["n_total_examples"]

    details = details.sort_values(["model", "task", "example_id"]).reset_index(drop=True)
    pivot = pivot.sort_values(["model", "task"]).reset_index(drop=True)
    return details, pivot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-instance transition taxonomy from canonical runs."
    )
    parser.add_argument(
        "--baseline-strategy",
        default="single_pass",
        help="Baseline strategy to compare from.",
    )
    parser.add_argument(
        "--comparison-strategy",
        default="self_refine",
        help="Comparison strategy to compare against the baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _load_manifest(MANIFEST_PATH)
    df_examples = _load_canonical_examples(manifest)
    details, summary = build_pairwise_taxonomy(
        df_examples=df_examples,
        manifest=manifest,
        baseline_strategy=args.baseline_strategy,
        comparison_strategy=args.comparison_strategy,
    )

    stem = f"{args.baseline_strategy}_vs_{args.comparison_strategy}"
    details_path = ANALYSIS_DIR / f"instance_taxonomy_details_{stem}.csv"
    summary_path = ANALYSIS_DIR / f"instance_taxonomy_summary_{stem}.csv"

    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved details to: {details_path}")
    print(f"Saved summary to: {summary_path}")
    print("\nSummary preview:")
    print(summary.to_string(index=False))

    if args.baseline_strategy == "single_pass" and args.comparison_strategy == "self_refine":
        best_details, best_summary = build_pairwise_taxonomy(
            df_examples=df_examples,
            manifest=manifest,
            baseline_strategy="single_pass",
            comparison_strategy="best_of_n",
        )
        best_details_path = ANALYSIS_DIR / "instance_taxonomy_details_single_pass_vs_best_of_n.csv"
        best_summary_path = ANALYSIS_DIR / "instance_taxonomy_summary_single_pass_vs_best_of_n.csv"
        best_details.to_csv(best_details_path, index=False)
        best_summary.to_csv(best_summary_path, index=False)

        internal_details, internal_summary = build_self_refine_internal_taxonomy(
            df_examples=df_examples,
            manifest=manifest,
        )
        internal_details_path = ANALYSIS_DIR / "instance_taxonomy_details_self_refine_draft_vs_final.csv"
        internal_summary_path = ANALYSIS_DIR / "instance_taxonomy_summary_self_refine_draft_vs_final.csv"
        internal_details.to_csv(internal_details_path, index=False)
        internal_summary.to_csv(internal_summary_path, index=False)

        print(f"\nSaved details to: {best_details_path}")
        print(f"Saved summary to: {best_summary_path}")
        print(f"Saved details to: {internal_details_path}")
        print(f"Saved summary to: {internal_summary_path}")


if __name__ == "__main__":
    main()
