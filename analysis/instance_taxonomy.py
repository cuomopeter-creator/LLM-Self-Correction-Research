from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.load_results import load_results_jsonl_flat
from data.loaders import load_arc, load_gsm8k, load_humaneval, load_truthfulqa
from evaluators.code_evaluator import evaluate_humaneval
from evaluators.math_evaluator import evaluate_math
from evaluators.qa_evaluator import evaluate_qa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_MANIFEST = PROJECT_ROOT / "analysis" / "run_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"

PAIRWISE_COMPARISONS = [
    ("single_pass", "best_of_n"),
    ("single_pass", "self_refine"),
]


def _load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "task", "strategy", "run_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return df


def _select_latest_runs(manifest: pd.DataFrame) -> pd.DataFrame:
    work = manifest.copy()
    work["run_dir"] = work["run_dir"].astype(str)
    work["n_examples"] = pd.to_numeric(work["n_examples"], errors="coerce")
    latest = (
        work.sort_values(["model", "task", "strategy", "n_examples", "run_dir"])
        .groupby(["model", "task", "strategy", "n_examples"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def _load_dataset_lookup(task: str) -> dict[str, Any]:
    if task == "gsm8k":
        return {str(ex.id): ex for ex in load_gsm8k(limit=None)}
    if task == "arc":
        return {str(ex.id): ex for ex in load_arc(limit=None)}
    if task == "truthfulqa":
        return {str(ex.id): ex for ex in load_truthfulqa(limit=None)}
    if task == "humaneval":
        return {str(ex.id): ex for ex in load_humaneval(limit=None)}
    raise ValueError(f"Unsupported task: {task}")


def _score_output(task: str, output: str, example: Any) -> tuple[int, str, str]:
    if task == "gsm8k":
        res = evaluate_math(output, example.answer)
        return int(res.correct), res.pred, res.gold
    if task in {"arc", "truthfulqa"}:
        res = evaluate_qa(output, example.answer)
        return int(res.correct), res.pred, res.gold
    if task == "humaneval":
        res = evaluate_humaneval(
            prompt=example.prompt,
            completion=output,
            test_code=example.test,
            entry_point=example.entry_point,
        )
        return int(res.passed), output, example.entry_point or ""
    raise ValueError(f"Unsupported task: {task}")


def _transition_label(baseline_score: Any, comparison_score: Any) -> str:
    b = int(baseline_score)
    c = int(comparison_score)
    if b == 1 and c == 1:
        return "correct_to_correct"
    if b == 1 and c == 0:
        return "correct_to_incorrect"
    if b == 0 and c == 1:
        return "incorrect_to_correct"
    return "incorrect_to_incorrect"


def _load_run_df(run_dir: str) -> pd.DataFrame:
    df = load_results_jsonl_flat(RUNS_DIR / run_dir / "results.jsonl").copy()
    df["example_id"] = df["example_id"].astype(str)
    return df


def _build_pairwise_details(
    latest_runs: pd.DataFrame,
    baseline_strategy: str,
    comparison_strategy: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for _, base_meta in latest_runs[latest_runs["strategy"] == baseline_strategy].iterrows():
        comp_matches = latest_runs[
            (latest_runs["model"] == base_meta["model"])
            & (latest_runs["task"] == base_meta["task"])
            & (latest_runs["n_examples"] == base_meta["n_examples"])
            & (latest_runs["strategy"] == comparison_strategy)
        ]
        if comp_matches.empty:
            continue

        comp_meta = comp_matches.iloc[0]
        base_df = _load_run_df(str(base_meta["run_dir"]))
        comp_df = _load_run_df(str(comp_meta["run_dir"]))

        merged = base_df.merge(
            comp_df,
            on="example_id",
            suffixes=("_x", "_y"),
            how="inner",
        )
        if merged.empty:
            continue

        merged["baseline_run_dir"] = str(base_meta["run_dir"])
        merged["comparison_run_dir"] = str(comp_meta["run_dir"])
        merged["model"] = str(base_meta["model"])
        merged["task"] = str(base_meta["task"])
        merged["strategy_x"] = baseline_strategy
        merged["strategy_y"] = comparison_strategy
        merged["baseline_strategy"] = baseline_strategy
        merged["comparison_strategy"] = comparison_strategy
        merged["transition_label"] = merged.apply(
            lambda row: _transition_label(row["score_int_x"], row["score_int_y"]),
            axis=1,
        )

        keep_cols = [
            "baseline_run_dir",
            "model",
            "task",
            "strategy_x",
            "example_id",
            "score_int_x",
            "output_x",
            "pred_x",
            "gold_x",
            "latency_s_x",
            "usage_total_total_tokens_x",
            "usage_total_tokens_x",
            "usage_input_tokens_x",
            "usage_output_tokens_x",
            "all_outputs_json_x",
            "intermediate_steps_json_x",
            "comparison_run_dir",
            "strategy_y",
            "score_int_y",
            "output_y",
            "pred_y",
            "gold_y",
            "latency_s_y",
            "usage_total_total_tokens_y",
            "usage_total_tokens_y",
            "usage_input_tokens_y",
            "usage_output_tokens_y",
            "all_outputs_json_y",
            "intermediate_steps_json_y",
            "baseline_strategy",
            "comparison_strategy",
            "transition_label",
        ]
        details = merged[keep_cols].rename(
            columns={
                "score_int_x": "baseline_score_int",
                "output_x": "baseline_output",
                "pred_x": "baseline_pred",
                "gold_x": "baseline_gold",
                "latency_s_x": "baseline_latency_s",
                "usage_total_total_tokens_x": "baseline_usage_total_total_tokens",
                "usage_total_tokens_x": "baseline_usage_total_tokens",
                "usage_input_tokens_x": "baseline_usage_input_tokens",
                "usage_output_tokens_x": "baseline_usage_output_tokens",
                "all_outputs_json_x": "all_outputs_json",
                "intermediate_steps_json_x": "intermediate_steps_json",
                "score_int_y": "comparison_score_int",
                "output_y": "comparison_output",
                "pred_y": "comparison_pred",
                "gold_y": "comparison_gold",
                "latency_s_y": "comparison_latency_s",
                "usage_total_total_tokens_y": "comparison_usage_total_total_tokens",
                "usage_total_tokens_y": "comparison_usage_total_tokens",
                "usage_input_tokens_y": "comparison_usage_input_tokens",
                "usage_output_tokens_y": "comparison_usage_output_tokens",
                "all_outputs_json_y": "comparison_all_outputs_json",
                "intermediate_steps_json_y": "comparison_intermediate_steps_json",
            }
        )
        rows.append(details)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(
        ["model", "task", "baseline_run_dir", "comparison_run_dir", "example_id"]
    ).reset_index(drop=True)
    return out


def _build_pairwise_summary(details: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, task, baseline_strategy, comparison_strategy), group in details.groupby(
        ["model", "task", "baseline_strategy", "comparison_strategy"],
        dropna=False,
    ):
        cc = float((group["transition_label"] == "correct_to_correct").sum())
        ci = float((group["transition_label"] == "correct_to_incorrect").sum())
        ic = float((group["transition_label"] == "incorrect_to_correct").sum())
        ii = float((group["transition_label"] == "incorrect_to_incorrect").sum())
        records.append(
            {
                "model": model,
                "task": task,
                "baseline_strategy": baseline_strategy,
                "comparison_strategy": comparison_strategy,
                "n_total_examples": int(len(group)),
                "correct_to_correct": cc,
                "correct_to_incorrect": ci,
                "incorrect_to_correct": ic,
                "incorrect_to_incorrect": ii,
                "revision_success_rate": (ic / (ic + ii)) if (ic + ii) else 0.0,
                "error_amplification_rate": (ci / (cc + ci)) if (cc + ci) else 0.0,
                "net_gain_examples": ic - ci,
                "net_gain_rate": (ic - ci) / len(group) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(records).sort_values(["model", "task"]).reset_index(drop=True)


def _build_self_refine_draft_details(latest_runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dataset_cache: dict[str, dict[str, Any]] = {}

    self_refine_runs = latest_runs[latest_runs["strategy"] == "self_refine"]
    for _, meta in self_refine_runs.iterrows():
        task = str(meta["task"])
        model = str(meta["model"])
        run_dir = str(meta["run_dir"])
        if task not in dataset_cache:
            dataset_cache[task] = _load_dataset_lookup(task)

        run_df = _load_run_df(run_dir)
        for _, row in run_df.iterrows():
            example_id = str(row["example_id"])
            example = dataset_cache[task].get(example_id)
            if example is None:
                continue

            steps = json.loads(row["intermediate_steps_json"] or "[]")
            draft_step = next(
                (
                    step
                    for step in steps
                    if isinstance(step, dict) and step.get("step") == "initial_draft"
                ),
                None,
            )
            if not draft_step:
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "example_id": example_id,
                        "baseline_strategy": "self_refine_draft",
                        "comparison_strategy": "self_refine_final",
                        "baseline_run_dir": run_dir,
                        "comparison_run_dir": run_dir,
                        "baseline_score_int": pd.NA,
                        "comparison_score_int": row["score_int"],
                        "baseline_output": pd.NA,
                        "comparison_output": row["output"],
                        "transition_label": "missing",
                        "gold": row["gold"],
                        "pred": row["pred"],
                        "run_dir": run_dir,
                    }
                )
                continue

            draft_output = str(draft_step.get("draft", ""))
            draft_score, draft_pred, draft_gold = _score_output(task, draft_output, example)
            final_score = int(row["score_int"])
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "example_id": example_id,
                    "baseline_strategy": "self_refine_draft",
                    "comparison_strategy": "self_refine_final",
                    "baseline_run_dir": run_dir,
                    "comparison_run_dir": run_dir,
                    "baseline_score_int": draft_score,
                    "comparison_score_int": final_score,
                    "baseline_output": draft_output,
                    "comparison_output": row["output"],
                    "transition_label": _transition_label(draft_score, final_score),
                    "gold": draft_gold,
                    "pred": row["pred"] if task != "humaneval" else draft_pred,
                    "run_dir": run_dir,
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["model", "task", "run_dir", "example_id"])
    return out.reset_index(drop=True)


def _build_self_refine_draft_summary(details: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, task, baseline_strategy, comparison_strategy), group in details.groupby(
        ["model", "task", "baseline_strategy", "comparison_strategy"],
        dropna=False,
    ):
        cc = float((group["transition_label"] == "correct_to_correct").sum())
        ci = float((group["transition_label"] == "correct_to_incorrect").sum())
        ic = float((group["transition_label"] == "incorrect_to_correct").sum())
        ii = float((group["transition_label"] == "incorrect_to_incorrect").sum())
        missing = float((group["transition_label"] == "missing").sum())
        valid = group[group["transition_label"] != "missing"]
        records.append(
            {
                "model": model,
                "task": task,
                "baseline_strategy": baseline_strategy,
                "comparison_strategy": comparison_strategy,
                "n_total_examples": int(len(group)),
                "correct_to_correct": cc,
                "correct_to_incorrect": ci,
                "incorrect_to_correct": ic,
                "incorrect_to_incorrect": ii,
                "missing": missing,
                "revision_success_rate": (ic / (ic + ii)) if (ic + ii) else 0.0,
                "error_amplification_rate": (ci / (cc + ci)) if (cc + ci) else 0.0,
                "net_gain_examples": ic - ci,
                "net_gain_rate": (ic - ci) / len(valid) if len(valid) else 0.0,
            }
        )
    return pd.DataFrame(records).sort_values(["model", "task"]).reset_index(drop=True)


def write_outputs(manifest_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_path)
    latest_runs = _select_latest_runs(manifest)

    for baseline_strategy, comparison_strategy in PAIRWISE_COMPARISONS:
        details = _build_pairwise_details(latest_runs, baseline_strategy, comparison_strategy)
        summary = _build_pairwise_summary(details)
        details_path = (
            output_dir
            / f"instance_taxonomy_details_{baseline_strategy}_vs_{comparison_strategy}.csv"
        )
        summary_path = (
            output_dir
            / f"instance_taxonomy_summary_{baseline_strategy}_vs_{comparison_strategy}.csv"
        )
        details.to_csv(details_path, index=False)
        summary.to_csv(summary_path, index=False)

    draft_details = _build_self_refine_draft_details(latest_runs)
    draft_summary = _build_self_refine_draft_summary(draft_details)
    draft_details.to_csv(
        output_dir / "instance_taxonomy_details_self_refine_draft_vs_final.csv",
        index=False,
    )
    draft_summary.to_csv(
        output_dir / "instance_taxonomy_summary_self_refine_draft_vs_final.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate instance-level taxonomy CSVs from the current run manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to run_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write taxonomy CSVs into",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_outputs(args.manifest, args.output_dir)
    print(f"Wrote taxonomy CSVs to: {args.output_dir}")


if __name__ == "__main__":
    main()
