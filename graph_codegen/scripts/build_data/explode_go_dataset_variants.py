from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "datasets" / "plotly_go_train_runtime_pass.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "datasets" / "plotly_go_train_exploded.jsonl"

SYSTEM_PROMPT = (
    "Generate a Streamlit app snippet using plotly.graph_objects only. "
    "Assume df already exists. Return code only."
)

REQUEST_PREFIXES = ["Create", "Build", "Show", "Plot", "Visualize", "Generate"]
SCHEMA_HEADERS = ["Dataframe columns:", "Columns in df:", "Available columns:"]
TITLE_FUNCS = ["title", "header", "subheader"]
USE_CONTAINER_WIDTH_OPTIONS = [True, False]
HOVERMODES = ["closest", "x unified"]
BARMODES = ["group", "stack", "overlay"]
BOX_POINTS = [None, "outliers", "all"]
HISTNORMS = [None, "percent", "probability"]
ANNOTATION_POSITIONS = ["outside", "inside"]
REFERENCE_LINE_MODES = ["none", "zero", "margin"]
LEGEND_ORIENTATIONS = ["v", "h"]
PLOT_BACKGROUNDS = ["white", "#fafafa", "#f9f9f9"]
GRID_COLORS = ["lightgrey", "#e5e5e5", "#eeeeee"]
PALETTE_A = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
PALETTE_B = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]
MARKER_SYMBOLS = ["circle", "diamond", "square", "x", "triangle-up"]

MEASURE_DTYPES = {"float", "int", "number", "numeric"}
CATEGORY_DTYPES = {"category", "string", "object"}
TIME_DTYPES = {"datetime", "date", "timestamp"}

KNOWN_REGION_VALUES = ["East", "West", "Central", "South", "North"]
KNOWN_SEGMENT_VALUES = ["Consumer", "Corporate", "Home Office"]
KNOWN_PRODUCT_VALUES = ["A", "B", "C"]
KNOWN_MONTH_VALUES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
KNOWN_YEAR_VALUES = [2021, 2022, 2023, 2024, 2025]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_dtype(dtype: str) -> str:
    return dtype.strip().lower()


def get_measure_cols(schema: dict[str, str]) -> list[str]:
    return [k for k, v in schema.items() if normalize_dtype(v) in MEASURE_DTYPES]


def get_category_cols(schema: dict[str, str]) -> list[str]:
    return [k for k, v in schema.items() if normalize_dtype(v) in CATEGORY_DTYPES]


def get_time_cols(schema: dict[str, str]) -> list[str]:
    return [k for k, v in schema.items() if normalize_dtype(v) in TIME_DTYPES or k in {"year", "month"}]


def parse_user_message(content: str) -> tuple[str, dict[str, str]]:
    content = content.strip()
    if "\n\n" not in content:
        raise ValueError("Could not split request from schema block.")

    first, second = content.split("\n\n", 1)
    request = first.replace("User request:\n", "", 1).strip()

    schema_lines = [line.rstrip() for line in second.splitlines() if line.strip()]
    if len(schema_lines) < 2:
        raise ValueError("Schema block missing.")

    schema: dict[str, str] = {}
    for line in schema_lines[1:]:
        if ":" not in line:
            continue
        col, dtype = line.split(":", 1)
        schema[col.strip()] = dtype.strip()

    if not schema:
        raise ValueError("Could not parse schema.")

    return request, schema


def extract_title(code: str) -> str | None:
    m = re.search(r'st\.(?:title|header|subheader)\("([^"]+)"\)', code)
    return m.group(1) if m else None


def infer_title_func(code: str) -> str:
    m = re.search(r"st\.(title|header|subheader)\(", code)
    return m.group(1) if m else "title"


def infer_use_container_width(code: str) -> bool:
    m = re.search(r"use_container_width=(True|False)", code)
    return m.group(1) == "True" if m else True


def infer_has_make_subplots(code: str) -> bool:
    return "make_subplots" in code


def infer_has_secondary_y(code: str) -> bool:
    return "secondary_y=True" in code


def infer_has_selectbox(code: str) -> bool:
    return "st.selectbox(" in code


def infer_widget_column(code: str, schema: dict[str, str]) -> str | None:
    m = re.search(r'st\.selectbox\("([^"]+)"', code)
    if not m:
        return None
    label = m.group(1).strip().lower().replace(" ", "_")
    for col in schema:
        if col.lower() == label or col.replace("_", " ").lower() == m.group(1).strip().lower():
            return col
    return None


def infer_chart_family(request: str, code: str, schema: dict[str, str]) -> str:
    lowered = code.lower()
    req = request.lower()

    if "go.histogram(" in lowered:
        return "histogram"
    if "go.box(" in lowered and infer_has_make_subplots(code):
        return "box_subplots"
    if "go.box(" in lowered:
        return "box"
    if "go.scatter(" in lowered and "go.bar(" not in lowered:
        return "scatter"
    if infer_has_make_subplots(code):
        return "dashboard"
    if "stacked bar" in req or 'barmode="stack"' in lowered:
        return "stacked_bar"
    if "grouped bar" in req or "go.bar(" in lowered:
        return "grouped_bar"

    measures = get_measure_cols(schema)
    categories = get_category_cols(schema)
    if len(measures) >= 2 and categories:
        return "grouped_bar"
    if len(measures) >= 2:
        return "scatter"
    raise ValueError("Could not infer graph_objects family.")


def pick_title(family: str, schema: dict[str, str], spec: dict[str, Any]) -> str:
    if family == "scatter":
        return f'{spec["measure_b"].replace("_", " ").title()} vs. {spec["measure_a"].replace("_", " ").title()} by {spec["group"].replace("_", " ").title()}'
    if family == "histogram":
        return f'{spec["measure_a"].replace("_", " ").title()} Distribution by {spec["group"].replace("_", " ").title()}'
    if family in {"grouped_bar", "stacked_bar"}:
        return f'{spec["measure_a"].replace("_", " ").title()} and {spec["measure_b"].replace("_", " ").title()} by {spec["x"].replace("_", " ").title()} and {spec["group"].replace("_", " ").title()}'
    if family in {"box", "box_subplots"}:
        return f'{spec["measure_a"].replace("_", " ").title()} and {spec["measure_b"].replace("_", " ").title()} Distribution by {spec["group"].replace("_", " ").title()}'
    if family == "dashboard":
        return "Performance Dashboard"
    return "Chart"


def parse_example(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages", [])
    if len(messages) != 3:
        raise ValueError("messages must contain 3 items")

    system = messages[0]["content"].strip()
    user_content = messages[1]["content"]
    code = messages[2]["content"]

    request, schema = parse_user_message(user_content)
    family = infer_chart_family(request, code, schema)
    measures = get_measure_cols(schema)
    categories = get_category_cols(schema)
    time_cols = get_time_cols(schema)

    if len(measures) < 1:
        raise ValueError("No measure columns detected")
    if len(categories) < 1 and family != "scatter":
        raise ValueError("No category columns detected")

    x_default = categories[0] if categories else measures[0]
    group_default = categories[1] if len(categories) > 1 else (categories[0] if categories else None)
    measure_a = measures[0]
    measure_b = measures[1] if len(measures) > 1 else measures[0]
    time_default = time_cols[0] if time_cols else None

    spec = {
        "family": family,
        "x": x_default,
        "group": group_default,
        "measure_a": measure_a,
        "measure_b": measure_b,
        "time": time_default,
        "title": extract_title(code),
        "title_func": infer_title_func(code),
        "use_container_width": infer_use_container_width(code),
        "use_widget": infer_has_selectbox(code),
        "widget_column": infer_widget_column(code, schema),
        "secondary_y": infer_has_secondary_y(code),
        "uses_subplots": infer_has_make_subplots(code),
        "annotation": "add_annotation(" in code,
        "reference_line": "add_hline(" in code,
    }

    if not spec["title"]:
        spec["title"] = pick_title(family, schema, spec)

    if system != SYSTEM_PROMPT:
        system = SYSTEM_PROMPT

    return {
        "system": system,
        "request": request,
        "schema": schema,
        "spec": spec,
        "raw_code": code,
    }


def pick_filter(schema: dict[str, str]) -> tuple[str, str, Any] | None:
    candidates: list[tuple[str, str, Any]] = []
    for col in schema:
        if col == "region":
            candidates.extend((col, "==", v) for v in KNOWN_REGION_VALUES)
        elif col == "segment":
            candidates.extend((col, "==", v) for v in KNOWN_SEGMENT_VALUES)
        elif col == "product":
            candidates.extend((col, "==", v) for v in KNOWN_PRODUCT_VALUES)
        elif col == "month":
            candidates.extend((col, "==", v) for v in KNOWN_MONTH_VALUES)
        elif col == "year":
            candidates.extend((col, "==", v) for v in KNOWN_YEAR_VALUES)
        elif normalize_dtype(schema[col]) in MEASURE_DTYPES:
            candidates.extend([
                (col, ">", 10),
                (col, ">", 50),
                (col, ">", 100),
            ])
    return random.choice(candidates) if candidates and random.random() < 0.5 else None


def choose_other_category(schema: dict[str, str], excluded: set[str]) -> str | None:
    candidates = [c for c in get_category_cols(schema) if c not in excluded]
    return random.choice(candidates) if candidates else None


def choose_other_measure(schema: dict[str, str], excluded: set[str]) -> str | None:
    candidates = [c for c in get_measure_cols(schema) if c not in excluded]
    return random.choice(candidates) if candidates else None


def render_schema(schema: dict[str, str], header: str, shuffle: bool) -> str:
    items = list(schema.items())
    if shuffle:
        random.shuffle(items)
    return header + "\n" + "\n".join(f"{k}: {v}" for k, v in items)


def render_filter_code(filter_spec: tuple[str, str, Any] | None, base_df: str) -> tuple[str, list[str]]:
    if not filter_spec:
        return base_df, []
    col, op, value = filter_spec
    value_repr = json.dumps(value)
    filtered = f"{base_df}_filtered"
    return filtered, [f'{filtered} = {base_df}[{base_df}["{col}"] {op} {value_repr}]']


def render_widget_code(widget_col: str | None, current_df: str) -> tuple[str, list[str]]:
    if not widget_col:
        return current_df, []
    safe = widget_col.replace(" ", "_")
    widget_df = f"{current_df}_widget"
    label = widget_col.replace("_", " ").title()
    lines = [
        f'options = ["All"] + sorted(df["{widget_col}"].dropna().unique().tolist())',
        f'selected_{safe} = st.selectbox("{label}", options)',
        f'if selected_{safe} == "All":',
        f'    {widget_df} = {current_df}',
        "else:",
        f'    {widget_df} = {current_df}[{current_df}["{widget_col}"] == selected_{safe}]',
    ]
    return widget_df, lines


def render_title(spec: dict[str, Any], title_func: str) -> str:
    return f'st.{title_func}("{spec["title"]}")'


def build_prompt(spec: dict[str, Any], style: dict[str, Any]) -> str:
    verb = REQUEST_PREFIXES[style["variant_idx"] % len(REQUEST_PREFIXES)]
    family = spec["family"]
    x = spec["x"]
    group = spec.get("group")
    a = spec["measure_a"]
    b = spec["measure_b"]
    time_col = spec.get("time")

    if family == "scatter":
        templates = [
            f"{verb} a scatter chart comparing total {a} vs. total {b} aggregated by {time_col or group}, with one trace per {group}",
            f"{verb} a scatter plot of monthly {a} against monthly {b}, split into separate traces by {group}",
            f"{verb} a scatter chart showing the relationship between {a} and {b} for each {group}",
        ]
    elif family == "histogram":
        templates = [
            f"{verb} overlaid histograms of {a} broken out by {group}",
            f"{verb} a histogram dashboard comparing the distribution of {a} across {group}",
            f"{verb} an overlay histogram of {a} for each {group}",
        ]
    elif family == "stacked_bar":
        templates = [
            f"{verb} a stacked bar chart showing total {a} composition by {group} across {x}",
            f"{verb} a stacked bar chart of {a} by {x}, with each stack representing {group}",
            f"{verb} a composition view of {a} across {x}, stacked by {group}",
        ]
    elif family == "grouped_bar":
        templates = [
            f"{verb} a grouped bar chart comparing total {a} and total {b} by {x}, broken out by {group}",
            f"{verb} grouped bars showing {a} and {b} across {x} for each {group}",
            f"{verb} a regional comparison of {a} versus {b} with one grouped trace set per {group}",
        ]
    elif family == "box":
        templates = [
            f"{verb} grouped box plots comparing the distribution of {a} and {b} across {group}, broken down by {x}",
            f"{verb} a box chart showing spread and outliers for {a} and {b} by {x} and {group}",
            f"{verb} distribution box plots for {a} and {b}, grouped by {x} and split by {group}",
        ]
    elif family == "box_subplots":
        templates = [
            f"{verb} a two-panel box plot dashboard with {a} in one subplot and {b} in another, grouped by {x} and colored by {group}",
            f"{verb} side-by-side box plot subplots for {a} and {b} across {x}, broken out by {group}",
            f"{verb} a subplot layout comparing {a} and {b} distributions by {x} and {group}",
        ]
    else:
        templates = [
            f"{verb} a two-panel dashboard combining grouped bars for {a} and {b} with another supporting view by {x} and {group}",
            f"{verb} a dashboard with subplot views so managers can compare {a}, {b}, and segment mix across {x}",
            f"{verb} a retail-style dashboard using subplot panels to compare {a} and {b} by {x} and {group}",
        ]

    req = templates[style["variant_idx"] % len(templates)]

    if style.get("filter"):
        col, op, value = style["filter"]
        req += f" where {col} {op} {value}"
    if style.get("widget_column"):
        req += f" with a Streamlit selectbox for {style['widget_column']}"
    if style.get("secondary_y"):
        req += " and include a secondary y-axis overlay"
    if style.get("annotation"):
        req += " with annotations for added context"
    if style.get("reference_line") == "zero":
        req += " and include a break-even reference line"
    elif style.get("reference_line") == "margin":
        req += " and include a margin reference line"

    return req


def render_grouped_bar(spec: dict[str, Any], style: dict[str, Any]) -> str:
    x = spec["x"]
    group = spec["group"]
    a = spec["measure_a"]
    b = spec["measure_b"]
    hovermode = style["hovermode"]
    bg = style["plot_bgcolor"]
    grid = style["gridcolor"]
    legend_orientation = style["legend_orientation"]

    lines = [
        "import streamlit as st",
        "import plotly.graph_objects as go",
        "",
        render_title(spec, style["title_func"]),
        "",
    ]

    current_df, filter_lines = render_filter_code(style.get("filter"), "df")
    lines.extend(filter_lines)
    if filter_lines:
        lines.append("")

    current_df, widget_lines = render_widget_code(style.get("widget_column"), current_df)
    lines.extend(widget_lines)
    if widget_lines:
        lines.append("")

    lines.extend([
        f'agg = {current_df}.groupby(["{x}", "{group}"], observed=True)[["{a}", "{b}"]].sum().reset_index()',
        f'x_vals = agg["{x}"].unique().tolist()',
        f'groups = agg["{group}"].unique().tolist()',
        f'colors_a = {repr(style["palette_a"])}',
        f'colors_b = {repr(style["palette_b"])}',
        "",
        "fig = go.Figure()",
        "",
        "for i, grp in enumerate(groups):",
        f'    grp_df = agg[agg["{group}"] == grp].set_index("{x}").reindex(x_vals).reset_index()',
        "    fig.add_trace(go.Bar(",
        f'        name=f"{{grp}} — {a.replace("_", " ").title()}",',
        f'        x=grp_df["{x}"],',
        f'        y=grp_df["{a}"],',
        "        offsetgroup=i,",
        "        legendgroup=str(grp),",
        "        marker_color=colors_a[i % len(colors_a)],",
        "    ))",
        "    fig.add_trace(go.Bar(",
        f'        name=f"{{grp}} — {b.replace("_", " ").title()}",',
        f'        x=grp_df["{x}"],',
        f'        y=grp_df["{b}"],',
        "        offsetgroup=i,",
        f'        base=grp_df["{a}"].fillna(0),',
        "        legendgroup=str(grp),",
        "        marker_color=colors_b[i % len(colors_b)],",
        "    ))",
        "",
        "fig.update_layout(",
        f'    barmode="{style["barmode"]}",',
        f'    title="{spec["title"]}",',
        f'    xaxis_title="{x.replace("_", " ").title()}",',
        '    yaxis_title="Amount ($)",',
        '    yaxis_tickformat="$,.0f",',
        f'    hovermode="{hovermode}",',
        f'    plot_bgcolor="{bg}",',
        f'    paper_bgcolor="{bg}",',
        f'    legend=dict(title="{group.replace("_", " ").title()} / Metric", orientation="{legend_orientation}"),',
        ")",
        f'fig.update_yaxes(showgrid=True, gridcolor="{grid}")',
    ])

    if style.get("annotation"):
        lines.extend([
            f'totals = agg.groupby("{x}")["{a}"].sum()',
            "for x_val, total_val in totals.items():",
            '    fig.add_annotation(x=x_val, y=total_val, text=f"${total_val:,.0f}", showarrow=False, yshift=8)',
        ])

    lines.extend([
        "",
        f'st.plotly_chart(fig, use_container_width={style["use_container_width"]})',
    ])
    return "\n".join(lines)


def render_stacked_bar(spec: dict[str, Any], style: dict[str, Any]) -> str:
    x = spec["x"]
    group = spec["group"]
    a = spec["measure_a"]
    time_col = spec.get("time")
    if not time_col:
        time_col = choose_other_category(spec["schema"], {x, group}) or x

    lines = [
        "import streamlit as st",
        "import plotly.graph_objects as go",
        "",
        render_title(spec, style["title_func"]),
        "",
    ]

    current_df, filter_lines = render_filter_code(style.get("filter"), "df")
    lines.extend(filter_lines)
    if filter_lines:
        lines.append("")

    current_df, widget_lines = render_widget_code(style.get("widget_column"), current_df)
    lines.extend(widget_lines)
    if widget_lines:
        lines.append("")

    if style.get("year_select") and time_col in {"year", "month"}:
        lines.extend([
            f'options_{time_col} = sorted({current_df}["{time_col}"].dropna().unique().tolist())',
            f'selected_{time_col} = st.selectbox("Select {time_col.replace("_", " ").title()}", options_{time_col})',
            f'{current_df} = {current_df}[{current_df}["{time_col}"] == selected_{time_col}]',
            "",
        ])

    if style.get("secondary_y") and spec["measure_b"] != a:
        lines = [
            "import streamlit as st",
            "import plotly.graph_objects as go",
            "from plotly.subplots import make_subplots",
            "",
            render_title(spec, style["title_func"]),
            "",
        ]
        lines.extend(filter_lines)
        if filter_lines:
            lines.append("")
        lines.extend(widget_lines)
        if widget_lines:
            lines.append("")
        lines.extend([
            f'agg = {current_df}.groupby(["{x}", "{group}"], observed=True)["{a}"].sum().reset_index()',
            f'overlay = {current_df}.groupby("{x}", observed=True)["{spec["measure_b"]}"].sum().reset_index()',
            f'x_vals = sorted(agg["{x}"].unique().tolist())',
            f'groups = sorted(agg["{group}"].unique().tolist())',
            f'colors = {repr(style["palette_a"])}',
            'fig = make_subplots(specs=[[{"secondary_y": True}]])',
            "",
            "for i, grp in enumerate(groups):",
            f'    grp_df = agg[agg["{group}"] == grp].set_index("{x}").reindex(x_vals).fillna(0).reset_index()',
            "    fig.add_trace(go.Bar(",
            "        name=str(grp),",
            f'        x=grp_df["{x}"],',
            f'        y=grp_df["{a}"],',
            "        marker_color=colors[i % len(colors)],",
            "    ), secondary_y=False)",
            "",
            f'overlay = overlay.set_index("{x}").reindex(x_vals).fillna(0).reset_index()',
            "fig.add_trace(go.Scatter(",
            f'    x=overlay["{x}"],',
            f'    y=overlay["{spec["measure_b"]}"],',
            f'    name="Total {spec["measure_b"].replace("_", " ").title()}",',
            '    mode="lines+markers",',
            '    line=dict(color="crimson", width=2, dash="dot"),',
            "), secondary_y=True)",
            "fig.update_layout(",
            '    barmode="stack",',
            f'    title="{spec["title"]}",',
            f'    hovermode="{style["hovermode"]}",',
            f'    plot_bgcolor="{style["plot_bgcolor"]}",',
            ")",
            f'fig.update_yaxes(title_text="Total {a.replace("_", " ").title()} ($)", secondary_y=False, tickformat="$,.0f")',
            f'fig.update_yaxes(title_text="Total {spec["measure_b"].replace("_", " ").title()} ($)", secondary_y=True, tickformat="$,.0f", showgrid=False)',
            "",
            f'st.plotly_chart(fig, use_container_width={style["use_container_width"]})',
        ])
        return "\n".join(lines)

    lines.extend([
        f'agg = {current_df}.groupby(["{x}", "{group}"], observed=True)["{a}"].sum().reset_index()',
        f'x_vals = sorted(agg["{x}"].unique().tolist())',
        f'groups = sorted(agg["{group}"].unique().tolist())',
        f'colors = {repr(style["palette_a"])}',
        "fig = go.Figure()",
        "",
        "for i, grp in enumerate(groups):",
        f'    grp_df = agg[agg["{group}"] == grp].set_index("{x}").reindex(x_vals).fillna(0).reset_index()',
        "    fig.add_trace(go.Bar(",
        "        name=str(grp),",
        f'        x=grp_df["{x}"],',
        f'        y=grp_df["{a}"],',
        "        marker_color=colors[i % len(colors)],",
        "    ))",
        "",
        "fig.update_layout(",
        '    barmode="stack",',
        f'    title="{spec["title"]}",',
        f'    xaxis_title="{x.replace("_", " ").title()}",',
        f'    yaxis_title="Total {a.replace("_", " ").title()} ($)",',
        '    yaxis_tickformat="$,.0f",',
        f'    hovermode="{style["hovermode"]}",',
        f'    plot_bgcolor="{style["plot_bgcolor"]}",',
        f'    paper_bgcolor="{style["plot_bgcolor"]}",',
        ")",
        f'fig.update_yaxes(showgrid=True, gridcolor="{style["gridcolor"]}")',
    ])

    if style.get("annotation"):
        lines.extend([
            f'totals = agg.groupby("{x}")["{a}"].sum()',
            "for x_val, total_val in totals.items():",
            '    fig.add_annotation(x=x_val, y=total_val, text=f"${total_val:,.0f}", showarrow=False, yshift=8)',
        ])

    lines.extend([
        "",
        f'st.plotly_chart(fig, use_container_width={style["use_container_width"]})',
    ])
    return "\n".join(lines)


def render_scatter(spec: dict[str, Any], style: dict[str, Any]) -> str:
    x = spec["measure_a"]
    y = spec["measure_b"]
    group = spec["group"] or spec["x"]
    time_col = spec.get("time") or choose_other_category(spec["schema"], {group}) or group

    size_col = style.get("size_col") or choose_other_measure(spec["schema"], {x, y})
    size_expr = "12"
    if size_col:
        size_expr = f'grp["{size_col}"]'

    lines = [
        "import streamlit as st",
        "import plotly.graph_objects as go",
        "",
        render_title(spec, style["title_func"]),
        "",
    ]

    current_df, filter_lines = render_filter_code(style.get("filter"), "df")
    lines.extend(filter_lines)
    if filter_lines:
        lines.append("")

    current_df, widget_lines = render_widget_code(style.get("widget_column"), current_df)
    lines.extend(widget_lines)
    if widget_lines:
        lines.append("")

    agg_expr = f'{current_df}.groupby(["{group}", "{time_col}"], observed=True).agg({x}=("{x}", "sum"), {y}=("{y}", "sum")'
    if size_col:
        agg_expr += f', {size_col}=("{size_col}", "sum")'
    agg_expr += ").reset_index()"

    lines.extend([
        f"agg = {agg_expr}",
        f'colors = {repr(style["palette_a"])}',
        f'symbols = {repr(MARKER_SYMBOLS)}',
        "fig = go.Figure()",
        "",
        f'for i, (grp_name, grp) in enumerate(agg.groupby("{group}", observed=True)):',
        "    marker_dict = dict(",
        f"        size={size_expr},",
        "        color=colors[i % len(colors)],",
        "        opacity=0.82,",
        '        line=dict(width=1, color="white"),',
        "    )",
    ])
    if style.get("symbol_mode"):
        lines.append('    marker_dict["symbol"] = symbols[i % len(symbols)]')
    lines.extend([
        "    fig.add_trace(go.Scatter(",
        f'        x=grp["{x}"],',
        f'        y=grp["{y}"],',
        '        mode="markers",',
        "        name=str(grp_name),",
        f'        text=grp["{time_col}"],',
        "        marker=marker_dict,",
        "    ))",
        "fig.update_layout(",
        f'    title="{spec["title"]}",',
        f'    xaxis_title="Total {x.replace("_", " ").title()} ($)",',
        f'    yaxis_title="Total {y.replace("_", " ").title()} ($)",',
        f'    hovermode="{style["hovermode"]}",',
        f'    plot_bgcolor="{style["plot_bgcolor"]}",',
        ")",
        f'fig.update_xaxes(showgrid=True, gridcolor="{style["gridcolor"]}", tickformat="$,.0f")',
        f'fig.update_yaxes(showgrid=True, gridcolor="{style["gridcolor"]}", tickformat="$,.0f")',
    ])

    if style.get("reference_line") == "zero":
        lines.append('fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")')
    elif style.get("reference_line") == "margin":
        lines.extend([
            f'x_vals = agg["{x}"]',
            "ref_x = [x_vals.min(), x_vals.max()]",
            "ref_y = [v * 0.2 for v in ref_x]",
            'fig.add_trace(go.Scatter(x=ref_x, y=ref_y, mode="lines", name="20% Margin Reference", line=dict(color="gray", dash="dash")))',
        ])

    lines.extend([
        "",
        f'st.plotly_chart(fig, use_container_width={style["use_container_width"]})',
    ])
    return "\n".join(lines)


def render_histogram(spec: dict[str, Any], style: dict[str, Any]) -> str:
    x = spec["measure_a"]
    group = spec["group"] or spec["x"]
    lines = [
        "import streamlit as st",
        "import plotly.graph_objects as go",
        "",
        render_title(spec, style["title_func"]),
        "",
    ]

    current_df, filter_lines = render_filter_code(style.get("filter"), "df")
    lines.extend(filter_lines)
    if filter_lines:
        lines.append("")

    current_df, widget_lines = render_widget_code(style.get("widget_column"), current_df)
    lines.extend(widget_lines)
    if widget_lines:
        lines.append("")

    lines.extend([
        f'groups = sorted({current_df}["{group}"].dropna().unique().tolist())',
        f'colors = {repr(style["palette_a"])}',
        "fig = go.Figure()",
        "",
        "for i, grp in enumerate(groups):",
        f'    grp_df = {current_df}[{current_df}["{group}"] == grp]',
        "    trace = go.Histogram(",
        f'        x=grp_df["{x}"],',
        "        name=str(grp),",
        "        marker_color=colors[i % len(colors)],",
        "        opacity=0.65,",
        "        nbinsx=20,",
        "    )",
    ])
    if style.get("histnorm") is not None:
        lines.append(f'    trace.histnorm = "{style["histnorm"]}"')
    lines.extend([
        "    fig.add_trace(trace)",
        "fig.update_layout(",
        f'    barmode="{style["barmode"]}",',
        f'    title="{spec["title"]}",',
        f'    xaxis_title="{x.replace("_", " ").title()}",',
        '    yaxis_title="Count",',
        f'    plot_bgcolor="{style["plot_bgcolor"]}",',
        ")",
        "",
        f'st.plotly_chart(fig, use_container_width={style["use_container_width"]})',
    ])
    return "\n".join(lines)


def render_box(spec: dict[str, Any], style: dict[str, Any], as_subplots: bool) -> str:
    x = spec["x"]
    group = spec["group"] or x
    a = spec["measure_a"]
    b = spec["measure_b"]

    lines = ["import streamlit as st", "import plotly.graph_objects as go"]
    if as_subplots:
        lines.append("from plotly.subplots import make_subplots")
    lines.extend(["", render_title(spec, style["title_func"]), ""])

    current_df, filter_lines = render_filter_code(style.get("filter"), "df")
    lines.extend(filter_lines)
    if filter_lines:
        lines.append("")

    current_df, widget_lines = render_widget_code(style.get("widget_column"), current_df)
    lines.extend(widget_lines)
    if widget_lines:
        lines.append("")

    boxpoints_value = json.dumps(style["box_points"]) if style.get("box_points") is not None else "False"

    if as_subplots:
        lines.extend([
            f'colors = {repr(style["palette_a"])}',
            f'fig = make_subplots(rows=1, cols=2, subplot_titles=("{a.replace("_", " ").title()}", "{b.replace("_", " ").title()}"))',
            f'groups = sorted({current_df}["{group}"].dropna().unique().tolist())',
            "for i, grp in enumerate(groups):",
            f'    grp_df = {current_df}[{current_df}["{group}"] == grp]',
            "    fig.add_trace(go.Box(",
            f'        x=grp_df["{x}"],',
            f'        y=grp_df["{a}"],',
            "        name=str(grp),",
            "        marker_color=colors[i % len(colors)],",
            f"        boxpoints={boxpoints_value},",
            "        legendgroup=str(grp),",
            "        showlegend=True,",
            "    ), row=1, col=1)",
            "    fig.add_trace(go.Box(",
            f'        x=grp_df["{x}"],',
            f'        y=grp_df["{b}"],',
            "        name=str(grp),",
            "        marker_color=colors[i % len(colors)],",
            f"        boxpoints={boxpoints_value},",
            "        legendgroup=str(grp),",
            "        showlegend=False,",
            "    ), row=1, col=2)",
            f'fig.update_layout(boxmode="group", title="{spec["title"]}", plot_bgcolor="{style["plot_bgcolor"]}")',
        ])
    else:
        lines.extend([
            f'colors = {repr(style["palette_a"])}',
            "fig = go.Figure()",
            f'groups = sorted({current_df}["{group}"].dropna().unique().tolist())',
            "for i, grp in enumerate(groups):",
            f'    grp_df = {current_df}[{current_df}["{group}"] == grp]',
            "    fig.add_trace(go.Box(",
            f'        x=grp_df["{x}"],',
            f'        y=grp_df["{a}"],',
            "        name=str(grp),",
            "        marker_color=colors[i % len(colors)],",
            f"        boxpoints={boxpoints_value},",
            "    ))",
            f'fig.update_layout(boxmode="group", title="{spec["title"]}", plot_bgcolor="{style["plot_bgcolor"]}")',
        ])

    lines.extend([
        "",
        f'st.plotly_chart(fig, use_container_width={style["use_container_width"]})',
    ])
    return "\n".join(lines)


def render_dashboard(spec: dict[str, Any], style: dict[str, Any]) -> str:
    x = spec["x"]
    group = spec["group"] or x
    a = spec["measure_a"]
    b = spec["measure_b"]
    extra = choose_other_measure(spec["schema"], {a, b}) or a

    lines = [
        "import streamlit as st",
        "import plotly.graph_objects as go",
        "from plotly.subplots import make_subplots",
        "",
        render_title(spec, style["title_func"]),
        "",
    ]

    current_df, filter_lines = render_filter_code(style.get("filter"), "df")
    lines.extend(filter_lines)
    if filter_lines:
        lines.append("")

    current_df, widget_lines = render_widget_code(style.get("widget_column"), current_df)
    lines.extend(widget_lines)
    if widget_lines:
        lines.append("")

    lines.extend([
        f'agg_ab = {current_df}.groupby(["{x}", "{group}"], observed=True)[["{a}", "{b}"]].sum().reset_index()',
        f'agg_extra = {current_df}.groupby(["{x}", "{group}"], observed=True)["{extra}"].sum().reset_index()',
        f'x_vals = sorted(agg_ab["{x}"].unique().tolist())',
        f'groups = sorted(agg_ab["{group}"].unique().tolist())',
        f'colors_a = {repr(style["palette_a"])}',
        f'colors_b = {repr(style["palette_b"])}',
        'fig = make_subplots(rows=1, cols=2, subplot_titles=("Metric Comparison", "Supporting View"), shared_yaxes=False)',
        "for i, grp in enumerate(groups):",
        f'    part_ab = agg_ab[agg_ab["{group}"] == grp].set_index("{x}").reindex(x_vals).fillna(0).reset_index()',
        f'    part_ex = agg_extra[agg_extra["{group}"] == grp].set_index("{x}").reindex(x_vals).fillna(0).reset_index()',
        "    fig.add_trace(go.Bar(",
        f'        name=f"{{grp}} — {a.replace("_", " ").title()}",',
        f'        x=part_ab["{x}"],',
        f'        y=part_ab["{a}"],',
        "        marker_color=colors_a[i % len(colors_a)],",
        "        offsetgroup=i,",
        "        legendgroup=str(grp),",
        "    ), row=1, col=1)",
        "    fig.add_trace(go.Bar(",
        f'        name=f"{{grp}} — {b.replace("_", " ").title()}",',
        f'        x=part_ab["{x}"],',
        f'        y=part_ab["{b}"],',
        "        marker_color=colors_b[i % len(colors_b)],",
        "        offsetgroup=i,",
        "        legendgroup=str(grp),",
        "        showlegend=False,",
        "    ), row=1, col=1)",
        "    fig.add_trace(go.Bar(",
        f'        name=f"{{grp}} — {extra.replace("_", " ").title()}",',
        f'        x=part_ex["{x}"],',
        f'        y=part_ex["{extra}"],',
        "        marker_color=colors_a[i % len(colors_a)],",
        "        legendgroup=str(grp),",
        "        showlegend=False,",
        "    ), row=1, col=2)",
        f'fig.update_layout(barmode="group", title="{spec["title"]}", plot_bgcolor="{style["plot_bgcolor"]}", hovermode="{style["hovermode"]}")',
        "",
        f'st.plotly_chart(fig, use_container_width={style["use_container_width"]})',
    ])
    return "\n".join(lines)


def render_code(spec: dict[str, Any], style: dict[str, Any]) -> str:
    family = spec["family"]
    if family == "grouped_bar":
        return render_grouped_bar(spec, style)
    if family == "stacked_bar":
        return render_stacked_bar(spec, style)
    if family == "scatter":
        return render_scatter(spec, style)
    if family == "histogram":
        return render_histogram(spec, style)
    if family == "box":
        return render_box(spec, style, as_subplots=False)
    if family == "box_subplots":
        return render_box(spec, style, as_subplots=True)
    return render_dashboard(spec, style)


def choose_variant_style(base: dict[str, Any], variant_idx: int) -> dict[str, Any]:
    schema = base["schema"]
    spec = copy.deepcopy(base["spec"])

    style = {
        "variant_idx": variant_idx,
        "title_func": random.choice(TITLE_FUNCS),
        "use_container_width": random.choice(USE_CONTAINER_WIDTH_OPTIONS),
        "hovermode": random.choice(HOVERMODES),
        "barmode": random.choice(BARMODES),
        "box_points": random.choice(BOX_POINTS),
        "histnorm": random.choice(HISTNORMS),
        "annotation": random.choice([True, False]),
        "reference_line": random.choice(REFERENCE_LINE_MODES),
        "legend_orientation": random.choice(LEGEND_ORIENTATIONS),
        "plot_bgcolor": random.choice(PLOT_BACKGROUNDS),
        "gridcolor": random.choice(GRID_COLORS),
        "palette_a": random.sample(PALETTE_A, k=len(PALETTE_A)),
        "palette_b": random.sample(PALETTE_B, k=len(PALETTE_B)),
        "widget_column": spec.get("widget_column") if spec.get("use_widget") else None,
        "filter": None,
        "secondary_y": random.choice([True, False]),
        "year_select": random.choice([True, False]),
        "size_col": choose_other_measure(schema, {spec["measure_a"], spec["measure_b"]}),
        "symbol_mode": random.choice([True, False]),
    }

    mode = variant_idx % 10
    if mode == 0:
        style["filter"] = pick_filter(schema)
    elif mode == 1:
        style["widget_column"] = choose_other_category(schema, {spec["x"], spec.get("group") or ""})
    elif mode == 2:
        style["annotation"] = True
        style["reference_line"] = "none"
    elif mode == 3:
        style["secondary_y"] = True
    elif mode == 4:
        style["year_select"] = True
    elif mode == 5:
        style["reference_line"] = "zero"
    elif mode == 6:
        style["reference_line"] = "margin"
    elif mode == 7:
        style["widget_column"] = choose_other_category(schema, {spec["x"], spec.get("group") or ""})
        style["filter"] = pick_filter(schema)
    elif mode == 8:
        style["annotation"] = True
        style["secondary_y"] = True
    elif mode == 9:
        style["symbol_mode"] = True
        style["size_col"] = choose_other_measure(schema, {spec["measure_a"], spec["measure_b"]})

    return style


def maybe_swap_semantics(spec: dict[str, Any], schema: dict[str, str], variant_idx: int) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    measures = get_measure_cols(schema)
    categories = get_category_cols(schema)
    times = get_time_cols(schema)

    if variant_idx % 3 == 0 and len(measures) >= 2:
        spec["measure_a"], spec["measure_b"] = spec["measure_b"], spec["measure_a"]

    if variant_idx % 4 == 0 and categories:
        spec["x"] = random.choice(categories)

    if variant_idx % 5 == 0 and len(categories) >= 2:
        group_candidates = [c for c in categories if c != spec["x"]]
        if group_candidates:
            spec["group"] = random.choice(group_candidates)

    if variant_idx % 6 == 0 and times:
        spec["time"] = random.choice(times)

    spec["schema"] = schema
    spec["title"] = pick_title(spec["family"], schema, spec)
    return spec


def build_variant(base: dict[str, Any], variant_idx: int) -> dict[str, Any] | None:
    schema = copy.deepcopy(base["schema"])
    spec = maybe_swap_semantics(base["spec"], schema, variant_idx)
    style = choose_variant_style(base, variant_idx)

    if spec["family"] in {"grouped_bar", "stacked_bar", "box", "box_subplots", "dashboard"}:
        if not spec.get("group"):
            spec["group"] = choose_other_category(schema, {spec["x"]}) or spec["x"]
    if spec["family"] == "scatter" and spec["measure_a"] == spec["measure_b"]:
        alt = choose_other_measure(schema, {spec["measure_a"]})
        if not alt:
            return None
        spec["measure_b"] = alt

    request = build_prompt(spec, style)
    header = SCHEMA_HEADERS[variant_idx % len(SCHEMA_HEADERS)]
    shuffle_schema = variant_idx % 2 == 1
    user_content = f"User request:\n{request}\n\n{render_schema(schema, header, shuffle_schema)}"
    code = render_code(spec, style)

    return {
        "messages": [
            {"role": "system", "content": base["system"]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": code},
        ]
    }


def dedupe_key(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def explode_rows(rows: list[dict[str, Any]], variants_per_example: int, output_path: Path, limit: int | None) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    seen: set[str] = set()
    written = 0
    skipped = 0

    subset = rows[:limit] if limit is not None else rows
    for base_idx, row in enumerate(subset, start=1):
        try:
            parsed = parse_example(row)
        except Exception:
            skipped += 1
            continue

        made = 0
        attempts = 0
        max_attempts = max(variants_per_example * 6, 30)

        while made < variants_per_example and attempts < max_attempts:
            attempts += 1
            variant_idx = base_idx + made + attempts
            variant = build_variant(parsed, variant_idx)
            if variant is None:
                continue
            key = dedupe_key(variant)
            if key in seen:
                continue
            seen.add(key)
            append_jsonl(output_path, variant)
            written += 1
            made += 1

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Explode passing graph_objects training rows into structural variants.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants-per-example", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = read_jsonl(args.input)
    written, skipped = explode_rows(rows, args.variants_per_example, args.output, args.limit)

    base_count = len(rows) if args.limit is None else min(len(rows), args.limit)
    print(f"Base examples read: {base_count}")
    print(f"Examples skipped during parse: {skipped}")
    print(f"Exploded examples written: {written}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
