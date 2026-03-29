from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "analysis" / "master_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figures"
DEFAULT_EFFICIENCY_INPUT = PROJECT_ROOT / "data" / "compute_efficiency.csv"

TASKS = ["arc", "gsm8k", "humaneval", "truthfulqa"]
STRATEGIES = ["single_pass", "best_of_n", "self_refine", "oracle"]
COLORS = {
    "single_pass": "#d62728",
    "best_of_n": "#1f77b4",
    "self_refine": "#2ca02c",
    "oracle": "#9467bd",
}
TASK_SYMBOLS = {
    "arc": "circle",
    "gsm8k": "square",
    "humaneval": "diamond",
    "truthfulqa": "x",
}


def _plot_model_scatter(
    df: pd.DataFrame,
    model_name: str,
    y_col: str,
    title_prefix: str,
    yaxis_title: str,
) -> go.Figure:
    model_df = df[df["model"] == model_name].copy()

    fig = go.Figure()
    for strategy in STRATEGIES:
        strat_df = (
            model_df[model_df["strategy"] == strategy]
            .copy()
            .set_index("task")
            .reindex(TASKS)
            .reset_index()
        )

        fig.add_trace(
            go.Scatter(
                x=strat_df["task"],
                y=strat_df[y_col],
                mode="markers",
                name=strategy,
                opacity=1.0 if strategy == "single_pass" else 0.55,
                marker=dict(
                    size=14 if strategy == "single_pass" else 10,
                    color=COLORS[strategy],
                ),
            )
        )

    fig.update_layout(
        title=f"{title_prefix} Across Datasets — {model_name}",
        xaxis_title="Dataset",
        yaxis_title=yaxis_title,
        width=900,
        height=500,
        legend_title="Strategy",
        title_x=0.5,
        font=dict(family="Aptos", size=16, color="black"),
        template="plotly_white",
    )

    if y_col == "accuracy_mean":
        fig.update_yaxes(range=[0.0, 1.05])

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.2)",
        zeroline=False,
    )
    return fig


def write_plots(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(df["model"].dropna().unique())
    for model_name in models:
        accuracy_fig = _plot_model_scatter(
            df=df,
            model_name=model_name,
            y_col="accuracy_mean",
            title_prefix="Accuracy",
            yaxis_title="Accuracy",
        )
        token_fig = _plot_model_scatter(
            df=df,
            model_name=model_name,
            y_col="token_usage_mean",
            title_prefix="Token Usage",
            yaxis_title="Mean Tokens",
        )

        accuracy_fig.write_html(
            output_dir / f"{model_name}_accuracy_scatter.html",
            include_plotlyjs="cdn",
        )
        token_fig.write_html(
            output_dir / f"{model_name}_token_usage_scatter.html",
            include_plotlyjs="cdn",
        )


def write_efficiency_heatmap(efficiency_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(efficiency_csv)
    df = df[df["group_level"] == "model"].copy()
    baseline = (
        df[df["strategy"] == "single_pass"][["model", "accuracy_per_1000_tokens"]]
        .rename(columns={"accuracy_per_1000_tokens": "baseline_accuracy_per_1000_tokens"})
    )
    comp = df.copy()
    comp = comp.merge(baseline, on=["model"], how="left")
    comp["delta_accuracy_per_1000_tokens"] = (
        comp["accuracy_per_1000_tokens"] - comp["baseline_accuracy_per_1000_tokens"]
    )

    fig = px.density_heatmap(
        comp,
        x="model",
        y="strategy",
        z="delta_accuracy_per_1000_tokens",
        text_auto=".2f",
        category_orders={
            "strategy": ["single_pass", "oracle", "best_of_n", "self_refine"],
            "model": sorted(comp["model"].dropna().unique()),
        },
        color_continuous_scale="RdBu",
    )
    fig.update_traces(
        zmid=0,
        hovertemplate=(
            "model=%{x}<br>"
            "strategy=%{y}<br>"
            "delta acc/1000 tok=%{z:.2f}<extra></extra>"
        ),
    )
    fig.update_layout(
        title="Model-Level Efficiency Delta vs Single-Pass: Accuracy per 1,000 Tokens",
        title_x=0.5,
        width=950,
        height=450,
        font=dict(family="Aptos", size=15, color="black"),
        template="plotly_white",
        coloraxis_colorbar_title="Delta",
    )
    fig.update_yaxes(title="Strategy", autorange="reversed")
    fig.update_xaxes(title="Model")
    fig.write_html(
        output_dir / "efficiency_delta_heatmap.html",
        include_plotlyjs="cdn",
    )


def write_master_tradeoff_plot(efficiency_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(efficiency_csv)
    df = df[df["group_level"] == "model_task"].copy()

    baseline = (
        df[df["strategy"] == "single_pass"][
            ["model", "task", "accuracy_mean", "token_usage_mean"]
        ]
        .rename(
            columns={
                "accuracy_mean": "baseline_accuracy_mean",
                "token_usage_mean": "baseline_token_usage_mean",
            }
        )
    )

    comp = df.merge(baseline, on=["model", "task"], how="left")
    comp["delta_accuracy"] = comp["accuracy_mean"] - comp["baseline_accuracy_mean"]
    comp["delta_accuracy_points"] = comp["delta_accuracy"] * 100.0
    comp["delta_token_usage_pct"] = (
        (comp["token_usage_mean"] - comp["baseline_token_usage_mean"])
        / comp["baseline_token_usage_mean"]
    ) * 100.0
    comp["model_task_label"] = comp["model"] + " / " + comp["task"]

    plot_df = comp[comp["strategy"] != "single_pass"].copy()

    def _pareto_frontier(panel_df: pd.DataFrame) -> pd.DataFrame:
        work = panel_df[
            ["model", "task", "strategy", "delta_token_usage_pct", "delta_accuracy_points"]
        ].dropna().copy()
        keep_rows = []
        for idx, row in work.iterrows():
            dominated = (
                (work["delta_token_usage_pct"] <= row["delta_token_usage_pct"])
                & (work["delta_accuracy_points"] >= row["delta_accuracy_points"])
                & (
                    (work["delta_token_usage_pct"] < row["delta_token_usage_pct"])
                    | (work["delta_accuracy_points"] > row["delta_accuracy_points"])
                )
            ).any()
            if not dominated:
                keep_rows.append(idx)
        frontier = work.loc[keep_rows].sort_values(
            ["delta_token_usage_pct", "delta_accuracy_points"]
        )
        return frontier

    x_min = min(-20.0, float(plot_df["delta_token_usage_pct"].min()) - 10.0)
    x_max = float(plot_df["delta_token_usage_pct"].max()) + 15.0
    y_min = min(-10.0, float(plot_df["delta_accuracy_points"].min()) - 2.0)
    y_max = float(plot_df["delta_accuracy_points"].max()) + 2.0

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=["<b>Claude</b>", "<b>GPT</b>", "<b>Kimi</b>", "<b>Llama</b>", "<b>Qwen</b>", ""],
        horizontal_spacing=0.08,
        vertical_spacing=0.1,
    )

    model_positions = {
        "claude": (1, 1),
        "gpt": (1, 2),
        "kimi": (2, 1),
        "llama": (2, 2),
        "qwen": (3, 1),
    }

    quadrant_rects = [
        (x_min, 0, 0, y_max, "rgba(46, 204, 113, 0.08)"),
        (0, x_max, 0, y_max, "rgba(241, 196, 15, 0.08)"),
        (0, x_max, y_min, 0, "rgba(231, 76, 60, 0.08)"),
    ]

    for model_name, (row, col) in model_positions.items():
        for x0, x1, y0, y1, color in quadrant_rects:
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                fillcolor=color,
                line=dict(width=0),
                layer="below",
                row=row,
                col=col,
            )
        fig.add_hline(y=0, line_width=1.2, line_dash="dash", line_color="gray", row=row, col=col)
        fig.add_vline(x=0, line_width=1.2, line_dash="dash", line_color="gray", row=row, col=col)

    for strategy in ["best_of_n", "self_refine", "oracle"]:
        strategy_df = plot_df[plot_df["strategy"] == strategy].copy()
        for model_name, (row, col) in model_positions.items():
            panel_df = strategy_df[strategy_df["model"] == model_name].copy()
            if panel_df.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=panel_df["delta_token_usage_pct"],
                    y=panel_df["delta_accuracy_points"],
                    mode="markers",
                    name=strategy,
                    legendgroup=strategy,
                    showlegend=(model_name == "claude"),
                    marker=dict(
                        size=12,
                        color=COLORS[strategy],
                        symbol=[TASK_SYMBOLS.get(task, "circle") for task in panel_df["task"]],
                        line=dict(color="white", width=1),
                    ),
                    customdata=panel_df[["model", "task", "accuracy_mean", "token_usage_mean"]],
                    hovertemplate=(
                        "model=%{customdata[0]}<br>"
                        "task=%{customdata[1]}<br>"
                        f"strategy={strategy}<br>"
                        "delta accuracy=%{y:.2f} points<br>"
                        "delta token usage=%{x:.1f}%<br>"
                        "accuracy=%{customdata[2]:.3f}<br>"
                        "mean tokens=%{customdata[3]:.1f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    for model_name, (row, col) in model_positions.items():
        frontier = _pareto_frontier(plot_df[plot_df["model"] == model_name].copy())
        if frontier.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=frontier["delta_token_usage_pct"],
                y=frontier["delta_accuracy_points"],
                mode="lines+markers",
                line=dict(color="black", width=2, dash="dot"),
                marker=dict(size=7, color="black"),
                name="Pareto frontier",
                legendgroup="pareto",
                showlegend=(model_name == "claude"),
                hovertemplate=(
                    "Pareto frontier<br>"
                    "delta accuracy=%{y:.2f} points<br>"
                    "delta token usage=%{x:.1f}%<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    for idx, task in enumerate(TASKS):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=11,
                    color="rgba(80,80,80,0.85)",
                    symbol=TASK_SYMBOLS[task],
                    line=dict(color="white", width=1),
                ),
                name=task,
                legend="legend2",
                legendgroup=f"task_{task}",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.99,
        text="<b>Better + cheaper</b>",
        showarrow=False,
        font=dict(size=18, color="#0f9d58"),
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.99,
        text="<b>Better + costlier</b>",
        showarrow=False,
        xanchor="right",
        font=dict(size=18, color="#f59e0b"),
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.01,
        text="<b>Worse + costlier</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=18, color="#d62828"),
        bgcolor="rgba(255,255,255,0.8)",
    )

    for annotation in fig.layout.annotations:
        if annotation.text in {"<b>Claude</b>", "<b>GPT</b>", "<b>Kimi</b>", "<b>Llama</b>", "<b>Qwen</b>"}:
            annotation.font = dict(size=19, color="black")

    fig.update_layout(
        title="Master Strategy Tradeoff Plot: Delta Accuracy vs Delta Token Usage Relative to Single-Pass",
        title_x=0.5,
        width=1250,
        height=1050,
        template="plotly_white",
        font=dict(family="Aptos", size=15, color="black"),
        legend_title="Strategy",
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
        legend2=dict(
            title="Task",
            x=1.02,
            y=0.45,
            xanchor="left",
            yanchor="top",
        ),
    )

    for row in [1, 2, 3]:
        for col in [1, 2]:
            if (row, col) == (3, 2):
                continue
            show_x = row == 3 or (row, col) == (2, 2)
            show_y = col == 1
            fig.update_xaxes(
                title="Δ Token Usage vs Single-Pass (%)" if show_x else None,
                showticklabels=show_x,
                range=[x_min, x_max],
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(0,0,0,0.12)",
                zeroline=False,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title="Δ Accuracy vs Single-Pass (points)" if show_y else None,
                showticklabels=show_y,
                range=[y_min, y_max],
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(0,0,0,0.12)",
                zeroline=False,
                row=row,
                col=col,
            )
    fig.write_html(
        output_dir / "master_strategy_tradeoff_scatter.html",
        include_plotlyjs="cdn",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate starter Plotly charts from analysis/master_results.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to master_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write Plotly HTML files into",
    )
    parser.add_argument(
        "--efficiency-input",
        type=Path,
        default=DEFAULT_EFFICIENCY_INPUT,
        help="Path to compute_efficiency.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_plots(args.input, args.output_dir)
    write_efficiency_heatmap(args.efficiency_input, args.output_dir)
    write_master_tradeoff_plot(args.efficiency_input, args.output_dir)
    print(f"Wrote Plotly figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
