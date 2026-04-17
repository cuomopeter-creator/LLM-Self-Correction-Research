from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "instance_taxonomy_summary_single_pass_vs_self_refine.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "analysis"
    / "figures"
    / "instance_taxonomy_single_pass_vs_self_refine_stacked.html"
)

TRANSITION_ORDER = [
    "correct_to_correct",
    "incorrect_to_correct",
    "correct_to_incorrect",
    "incorrect_to_incorrect",
]

TRANSITION_LABELS = {
    "correct_to_correct": "Success Preserved",
    "incorrect_to_correct": "Self-Correction (Success)",
    "correct_to_incorrect": "Error Amplification",
    "incorrect_to_incorrect": "Failed Correction",
}

TRANSITION_COLORS = {
    "correct_to_correct": "rgba(31, 119, 180, 0.35)",
    "incorrect_to_correct": "rgba(44, 160, 44, 0.35)",
    "correct_to_incorrect": "#d62728",
    "incorrect_to_incorrect": "rgba(148, 103, 189, 0.35)",
}

TASK_ORDER = ["arc", "gsm8k", "humaneval", "truthfulqa"]
MODEL_ORDER = ["claude", "gpt", "kimi", "llama", "qwen"]


def build_figure(input_csv: Path) -> go.Figure:
    df = pd.read_csv(input_csv).copy()
    for col in TRANSITION_ORDER + ["n_total_examples"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["task"] = pd.Categorical(df["task"], categories=TASK_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["model", "task"]).reset_index(drop=True)

    df["condition"] = df["model"].astype(str) + " / " + df["task"].astype(str)
    for col in TRANSITION_ORDER:
        df[f"{col}_pct"] = (df[col] / df["n_total_examples"]) * 100.0

    baseline_name = str(df["baseline_strategy"].iloc[0]).replace("_", "-")
    comparison_name = str(df["comparison_strategy"].iloc[0]).replace("_", "-")

    fig = go.Figure()
    for transition in TRANSITION_ORDER:
        customdata = df[["model", "task", transition, "n_total_examples"]]
        fig.add_trace(
            go.Bar(
                x=df["condition"],
                y=df[f"{transition}_pct"],
                name=TRANSITION_LABELS[transition],
                marker_color=TRANSITION_COLORS[transition],
                customdata=customdata,
                hovertemplate=(
                    "model=%{customdata[0]}<br>"
                    "task=%{customdata[1]}<br>"
                    "transition=" + TRANSITION_LABELS[transition] + "<br>"
                    "count=%{customdata[2]:.0f} / %{customdata[3]:.0f}<br>"
                    "share=%{y:.1f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        title=f"Instance-Level Taxonomy: {comparison_name.title()} vs {baseline_name.title()}",
        title_x=0.5,
        width=900,
        height=500,
        template="plotly_white",
        font=dict(family="Aptos", size=16, color="black"),
        legend_title="Transition Type",
    )
    fig.update_xaxes(title="Model / Task", tickangle=-35)
    fig.update_yaxes(
        title="Share of Examples (%)",
        range=[0, 100],
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.2)",
        zeroline=False,
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked instance-taxonomy bars for single-pass vs self-refine."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to instance taxonomy summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig = build_figure(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Wrote taxonomy plot to: {args.output}")


if __name__ == "__main__":
    main()
