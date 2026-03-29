from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "bootstrap_accuracy_ci_samples.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "figures" / "bootstrap_ci_model_distributions.html"

COMPARISON_ORDER = [
    "best_of_n_vs_single_pass",
    "self_refine_vs_single_pass",
    "oracle_vs_single_pass",
]
COMPARISON_TITLES = {
    "best_of_n_vs_single_pass": "Best-of-N vs Single-Pass",
    "self_refine_vs_single_pass": "Self-Refine vs Single-Pass",
    "oracle_vs_single_pass": "Oracle vs Single-Pass",
}
MODEL_ORDER = ["claude", "gpt", "kimi", "llama", "qwen"]
MODEL_COLORS = {
    "claude": "#1f77b4",
    "gpt": "#ff7f0e",
    "kimi": "#2ca02c",
    "llama": "#d62728",
    "qwen": "#9467bd",
}


def build_figure(samples_csv: Path) -> go.Figure:
    df = pd.read_csv(samples_csv)
    df = df[df["group_level"] == "model"].copy()

    fig = make_subplots(
        rows=len(MODEL_ORDER),
        cols=3,
        subplot_titles=[
            COMPARISON_TITLES[comparison] if row_idx == 0 else ""
            for row_idx, _model in enumerate(MODEL_ORDER)
            for comparison in COMPARISON_ORDER
        ],
        shared_xaxes=False,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    for row_idx, model in enumerate(MODEL_ORDER, start=1):
        for col_idx, comparison in enumerate(COMPARISON_ORDER, start=1):
            model_df = df[
                (df["comparison"] == comparison)
                & (df["model"] == model)
            ]
            if model_df.empty:
                continue
            fig.add_trace(
                go.Histogram(
                    x=model_df["accuracy_diff"],
                    name=model,
                    legendgroup=model,
                    marker=dict(color=MODEL_COLORS[model]),
                    opacity=0.8,
                    showlegend=(col_idx == 1),
                    hovertemplate=(
                        f"comparison={COMPARISON_TITLES[comparison]}<br>"
                        f"model={model}<br>"
                        "accuracy diff=%{x:.4f}<br>"
                        "count=%{y}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=col_idx,
            )

            fig.add_vline(
                x=0,
                line_width=2,
                line_dash="dash",
                line_color="rgba(0,0,0,1.0)",
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(
        title="Bootstrap Accuracy-Difference Distributions by Model",
        title_x=0.5,
        width=1400,
        height=1600,
        template="plotly_white",
        font=dict(family="Aptos", size=15, color="black"),
        showlegend=True,
        legend_title="Model",
        bargap=0.05,
    )
    for row_idx, model in enumerate(MODEL_ORDER, start=1):
        for col_idx in range(1, len(COMPARISON_ORDER) + 1):
            fig.update_xaxes(
                title="Accuracy Difference" if row_idx == len(MODEL_ORDER) else None,
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                title=f"{model} Count" if col_idx == 1 else None,
                showticklabels=(col_idx == 1),
                row=row_idx,
                col=col_idx,
            )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model-level bootstrap CI distributions as Plotly subplots."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to bootstrap_accuracy_ci_samples.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output HTML path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig = build_figure(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Wrote bootstrap CI distribution plot to: {args.output}")


if __name__ == "__main__":
    main()
