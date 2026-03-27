from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "datasets" / "plotly_streamlit_train_runtime_pass.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "datasets" / "plotly_streamlit_train_exploded.jsonl"

SYSTEM_PROMPT = (
    "Generate a Streamlit app snippet using plotly.express only. "
    "Assume df already exists. Return code only."
)

REQUEST_PREFIXES = ["Create", "Build", "Show", "Plot", "Visualize", "Generate"]
SCHEMA_HEADERS = ["Dataframe columns:", "Columns in df:", "Available columns:"]
TITLE_FUNCS = ["title", "header", "subheader"]
FIG_NAMES = ["fig", "chart", "plot_fig"]
USE_CONTAINER_WIDTH_OPTIONS = [True, False]
TEMPLATES = ["plotly_white", "plotly", "simple_white", None]
MARGINALS = ["box", "rug", None]
BOX_POINTS = [None, "all", "outliers"]
BARMODES = [None, "group", "stack"]

MEASURE_DTYPES = {"float", "int", "number", "numeric"}
CATEGORY_DTYPES = {"category", "string", "object"}
TIME_DTYPES = {"datetime", "date", "timestamp"}

MEASURE_COLS = {"sales", "profit", "revenue", "units", "customer_age"}
CATEGORY_COLS = {"region", "product", "segment"}
TIME_COLS = {"date"}
TIME_PART_COLS = {"year", "month"}


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
    cols = [k for k, v in schema.items() if normalize_dtype(v) in MEASURE_DTYPES]
    return cols or [k for k in schema if k in MEASURE_COLS]


def get_category_cols(schema: dict[str, str]) -> list[str]:
    cols = [k for k, v in schema.items() if normalize_dtype(v) in CATEGORY_DTYPES]
    return cols or [k for k in schema if k in CATEGORY_COLS]


def get_time_cols(schema: dict[str, str]) -> list[str]:
    cols = [k for k, v in schema.items() if normalize_dtype(v) in TIME_DTYPES]
    return cols or [k for k in schema if k in TIME_COLS]


def get_time_part_cols(schema: dict[str, str]) -> list[str]:
    return [k for k in schema if k in TIME_PART_COLS]


def parse_user_message(content: str) -> tuple[str, dict[str, str]]:
    content = content.strip()
    if "\n\n" not in content:
        raise ValueError("Could not split request and schema block.")

    first, second = content.split("\n\n", 1)

    request = first
    if request.startswith("User request:\n"):
        request = request.replace("User request:\n", "", 1).strip()

    schema_lines = [line.rstrip() for line in second.splitlines() if line.strip()]
    if not schema_lines:
        raise ValueError("Schema block empty.")

    schema: dict[str, str] = {}
    for line in schema_lines[1:]:
        if ":" not in line:
            continue
        col, dtype = line.split(":", 1)
        schema[col.strip()] = dtype.strip()

    if not schema:
        raise ValueError("Could not parse schema.")

    return request, schema


def infer_chart_type(code: str, request: str) -> str:
    lowered = code.lower()
    if "px.scatter(" in lowered:
        return "scatter"
    if "px.bar(" in lowered:
        return "bar"
    if "px.line(" in lowered:
        return "line"
    if "px.box(" in lowered:
        return "box"
    if "px.histogram(" in lowered:
        return "histogram"
    if "px.density_heatmap(" in lowered:
        return "heatmap"

    req = request.lower()
    if "scatter" in req:
        return "scatter"
    if "bar" in req:
        return "bar"
    if "line" in req:
        return "line"
    if "box" in req:
        return "box"
    if "histogram" in req:
        return "histogram"
    if "heatmap" in req:
        return "heatmap"

    raise ValueError("Could not infer chart type.")


def extract_arg(code: str, arg: str) -> str | None:
    m = re.search(rf'{arg}\s*=\s*"([^"]+)"', code)
    return m.group(1) if m else None


def infer_aggregation(code: str, request: str) -> str | None:
    group_m = re.search(r'\]\.(sum|mean|median|count|min|max)\(\)', code)
    if group_m:
        return group_m.group(1)

    req = request.lower()
    for agg in ("sum", "mean", "median", "count", "min", "max"):
        if f" {agg} " in f" {req} ":
            return agg
    return None


def infer_filter(code: str) -> tuple[str, str, Any] | None:
    m = re.search(
        r'df(?:_\w+)?\s*=\s*df\[\s*df\["([^"]+)"\]\s*(==|>|<|>=|<=)\s*(".*?"|\d+)\s*\]',
        code,
    )
    if not m:
        return None

    col, op, raw_value = m.groups()
    if raw_value.startswith('"') and raw_value.endswith('"'):
        value: Any = raw_value[1:-1]
    else:
        value = int(raw_value)
    return (col, op, value)


def infer_widget(code: str) -> str | None:
    m = re.search(r'st\.selectbox\("([^"]+)"', code)
    if not m:
        return None
    label = m.group(1).strip().lower().replace(" ", "_")
    return label


def infer_style(code: str) -> dict[str, Any]:
    style: dict[str, Any] = {
        "title_func": "title",
        "fig_name": "fig",
        "use_container_width": True,
        "template": None,
        "add_layout": False,
        "markers": False,
        "marginal": None,
        "box_points": None,
        "barmode": None,
        "facet_col": None,
        "symbol": None,
        "size": None,
        "sort_by_x": False,
        "top_n": None,
        "horizontal": False,
        "text_auto": False,
    }

    m_title = re.search(r"st\.(title|header|subheader)\(", code)
    if m_title:
        style["title_func"] = m_title.group(1)

    m_fig = re.search(r"^([A-Za-z_]\w*)\s*=\s*px\.", code, flags=re.MULTILINE)
    if m_fig:
        style["fig_name"] = m_fig.group(1)

    m_ucw = re.search(r"st\.plotly_chart\([^)]*use_container_width=(True|False)", code)
    if m_ucw:
        style["use_container_width"] = m_ucw.group(1) == "True"

    m_template = re.search(r'template="([^"]+)"', code)
    if m_template:
        style["template"] = m_template.group(1)

    style["add_layout"] = ".update_layout(" in code
    style["markers"] = "markers=True" in code

    m_marginal = re.search(r'marginal="([^"]+)"', code)
    if m_marginal:
        style["marginal"] = m_marginal.group(1)

    m_points = re.search(r'points="([^"]+)"', code)
    if m_points:
        style["box_points"] = m_points.group(1)

    m_barmode = re.search(r'barmode="([^"]+)"', code)
    if m_barmode:
        style["barmode"] = m_barmode.group(1)

    m_facet = re.search(r'facet_col="([^"]+)"', code)
    if m_facet:
        style["facet_col"] = m_facet.group(1)

    m_symbol = re.search(r'symbol="([^"]+)"', code)
    if m_symbol:
        style["symbol"] = m_symbol.group(1)

    m_size = re.search(r'size="([^"]+)"', code)
    if m_size:
        style["size"] = m_size.group(1)

    if ".sort_values(" in code:
        style["sort_by_x"] = True

    m_topn = re.search(r"\.nlargest\((\d+),\s*\"([^\"]+)\"\)", code)
    if m_topn:
        style["top_n"] = int(m_topn.group(1))

    style["horizontal"] = re.search(r'\borientation="h"', code) is not None
    style["text_auto"] = "text_auto=True" in code

    return style


def parse_example(row: dict[str, Any]) -> dict[str, Any]:
    msgs = row["messages"]
    system = msgs[0]["content"].strip()
    user_content = msgs[1]["content"]
    code = msgs[2]["content"]

    request, schema = parse_user_message(user_content)
    chart_type = infer_chart_type(code, request)

    spec = {
        "chart_type": chart_type,
        "x": extract_arg(code, "x"),
        "y": extract_arg(code, "y"),
        "color": extract_arg(code, "color"),
        "aggregation": infer_aggregation(code, request),
        "filter": infer_filter(code),
        "use_widget": infer_widget(code) is not None,
        "widget_column": infer_widget(code),
        "title": extract_title(code),
    }

    style = infer_style(code)

    if system != SYSTEM_PROMPT:
        system = SYSTEM_PROMPT

    return {
        "system": system,
        "request": request,
        "schema": schema,
        "spec": spec,
        "style": style,
        "raw_code": code,
    }


def extract_title(code: str) -> str | None:
    m = re.search(r'st\.(?:title|header|subheader)\("([^"]+)"\)', code)
    return m.group(1) if m else None


def build_title(spec: dict[str, Any]) -> str:
    x = spec.get("x")
    y = spec.get("y")
    agg = spec.get("aggregation")

    if spec["chart_type"] == "histogram":
        return f"Distribution of {x.replace('_', ' ').title()}" if x else "Histogram"

    if spec["chart_type"] == "bar":
        if agg and y and x:
            return f"{agg.title()} {y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}"
        if y and x:
            return f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}"
        return "Bar Chart"

    if spec["chart_type"] == "line":
        if agg and y and x:
            return f"{agg.title()} {y.replace('_', ' ').title()} over {x.replace('_', ' ').title()}"
        if y and x:
            return f"{y.replace('_', ' ').title()} over {x.replace('_', ' ').title()}"
        return "Line Chart"

    if spec["chart_type"] == "box":
        if y and x:
            return f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}"
        return "Box Plot"

    if spec["chart_type"] == "heatmap":
        if x and y:
            return f"Count of {x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}"
        return "Heatmap"

    if x and y:
        return f"{x.replace('_', ' ').title()} vs {y.replace('_', ' ').title()}"
    return "Chart"


def maybe_choose_filter(schema: dict[str, str]) -> tuple[str, str, Any] | None:
    candidates: list[tuple[str, str, Any]] = []

    for col in schema:
        if col == "region":
            candidates.extend([
                (col, "==", "North"),
                (col, "==", "South"),
                (col, "==", "East"),
                (col, "==", "West"),
            ])
        elif col == "segment":
            candidates.extend([
                (col, "==", "Consumer"),
                (col, "==", "Corporate"),
                (col, "==", "Home Office"),
            ])
        elif col == "product":
            candidates.extend([
                (col, "==", "A"),
                (col, "==", "B"),
                (col, "==", "C"),
            ])
        elif col == "year":
            candidates.extend([
                (col, "==", 2022),
                (col, "==", 2023),
                (col, "==", 2024),
            ])
        elif col in get_measure_cols(schema):
            candidates.extend([
                (col, ">", 10),
                (col, ">", 50),
                (col, ">", 100),
            ])

    return random.choice(candidates) if candidates else None


def paraphrase_request(spec: dict[str, Any], variant_idx: int) -> str:
    verb = REQUEST_PREFIXES[variant_idx % len(REQUEST_PREFIXES)]

    x = spec["x"]
    y = spec["y"]
    color = spec.get("color")

    templates: list[str] = []

    if spec["chart_type"] == "scatter":
        templates = [
            f"{verb} a scatter plot of {x} vs {y}",
            f"{verb} a scatter chart comparing {x} and {y}",
            f"{verb} {x} against {y} in a scatter plot",
            f"{verb} a scatter plot for {x} versus {y}",
        ]
        if color:
            templates = [t + f" colored by {color}" for t in templates]

    elif spec["chart_type"] == "bar":
        agg = spec["aggregation"]
        templates = [
            f"{verb} a bar chart of {agg} {y} by {x}",
            f"{verb} total {y} by {x} as a bar chart" if agg == "sum" else f"{verb} {agg} {y} by {x} in a bar chart",
            f"{verb} a bar plot showing {agg} {y} across {x}",
            f"{verb} {agg} {y} grouped by {x} with a bar chart",
        ]
        if color:
            templates = [t + f" colored by {color}" for t in templates]
        if spec.get("horizontal"):
            templates = [t + " in horizontal form" for t in templates]

    elif spec["chart_type"] == "box":
        templates = [
            f"{verb} a box plot of {y} by {x}",
            f"{verb} the distribution of {y} across {x} using a box plot",
            f"{verb} a box chart for {y} grouped by {x}",
            f"{verb} {y} by {x} in a box plot",
        ]
        if color:
            templates = [t + f" colored by {color}" for t in templates]

    elif spec["chart_type"] == "histogram":
        templates = [
            f"{verb} a histogram of {x}",
            f"{verb} the distribution of {x} as a histogram",
            f"{verb} a histogram showing {x}",
            f"{verb} a frequency distribution for {x}",
        ]
        if color:
            templates = [t + f" colored by {color}" for t in templates]

    elif spec["chart_type"] == "line":
        agg = spec["aggregation"]
        templates = [
            f"{verb} a line chart of {agg} {y} over {x}",
            f"{verb} {agg} {y} across {x} with a line chart",
            f"{verb} a time-style line chart for {agg} {y} by {x}",
            f"{verb} a line plot showing {agg} {y} over {x}",
        ]
        if color:
            templates = [t + f" colored by {color}" for t in templates]
        if spec.get("markers"):
            templates = [t + " with markers" for t in templates]

    else:
        templates = [
            f"{verb} a heatmap of count by {x} and {y}",
            f"{verb} a density heatmap for {x} and {y}",
            f"{verb} a heatmap showing counts across {x} and {y}",
            f"{verb} a count heatmap by {x} versus {y}",
        ]

    req = templates[variant_idx % len(templates)]

    if spec.get("filter"):
        col, op, value = spec["filter"]
        req += f" where {col} {op} {value}"

    if spec.get("use_widget") and spec.get("widget_column"):
        req += f" with a Streamlit selectbox for {spec['widget_column']}"

    if spec.get("facet_col"):
        req += f" with facets by {spec['facet_col']}"

    if spec.get("symbol"):
        req += f" using symbols for {spec['symbol']}"

    if spec.get("size"):
        req += f" with point size based on {spec['size']}"

    return req


def render_schema(schema: dict[str, str], header: str, shuffle: bool) -> str:
    items = list(schema.items())
    if shuffle:
        random.shuffle(items)
    body = "\n".join(f"{k}: {v}" for k, v in items)
    return f"{header}\n{body}"


def render_filter_code(filter_tuple: tuple[str, str, Any] | None, df_name: str) -> tuple[str, str]:
    if not filter_tuple:
        return df_name, ""

    col, op, value = filter_tuple
    value_repr = f'"{value}"' if isinstance(value, str) else str(value)
    new_df = f"{df_name}_filtered"
    code = f'{new_df} = {df_name}[{df_name}["{col}"] {op} {value_repr}]\n'
    return new_df, code


def render_widget_code(spec: dict[str, Any], current_df: str) -> tuple[str, str]:
    if not spec.get("use_widget") or not spec.get("widget_column"):
        return current_df, ""

    col = spec["widget_column"]
    safe_col = col.replace(" ", "_")
    new_df = f"{current_df}_widget"
    code = (
        f'options = ["All"] + sorted(df["{col}"].dropna().unique().tolist())\n'
        f'selected_{safe_col} = st.selectbox("{col.replace("_", " ").title()}", options)\n'
        f'if selected_{safe_col} == "All":\n'
        f'    {new_df} = {current_df}\n'
        f'else:\n'
        f'    {new_df} = {current_df}[{current_df}["{col}"] == selected_{safe_col}]\n'
    )
    return new_df, code


def render_agg_df(spec: dict[str, Any], base_df: str, style: dict[str, Any]) -> tuple[str, str]:
    chart_type = spec["chart_type"]
    if chart_type not in {"bar", "line", "heatmap"}:
        return base_df, ""

    if chart_type == "heatmap":
        agg_df = "df_plot"
        code = (
            f'{agg_df} = {base_df}.groupby(["{spec["x"]}", "{spec["y"]}"], as_index=False)'
            f'.size().rename(columns={{"size": "count"}})\n'
        )
        return agg_df, code

    if not spec.get("aggregation"):
        return base_df, ""

    group_cols = [spec["x"]]
    if spec.get("color") and spec["color"] != spec["x"]:
        group_cols.append(spec["color"])

    cols_expr = ", ".join([f'"{c}"' for c in group_cols])
    agg_df = "df_plot"
    code = (
        f'{agg_df} = {base_df}.groupby([{cols_expr}], as_index=False)["{spec["y"]}"].{spec["aggregation"]}()\n'
    )

    if style.get("sort_by_x"):
        code += f'{agg_df} = {agg_df}.sort_values(by="{spec["x"]}")\n'

    if style.get("top_n") and chart_type == "bar":
        code += f'{agg_df} = {agg_df}.nlargest({style["top_n"]}, "{spec["y"]}")\n'

    return agg_df, code


def build_chart_kwargs(spec: dict[str, Any], style: dict[str, Any]) -> list[str]:
    kwargs: list[str] = []

    if spec["chart_type"] == "histogram":
        kwargs.append(f'x="{spec["x"]}"')
        if spec.get("color"):
            kwargs.append(f'color="{spec["color"]}"')
        if style.get("marginal"):
            kwargs.append(f'marginal="{style["marginal"]}"')

    elif spec["chart_type"] == "heatmap":
        kwargs.append(f'x="{spec["x"]}"')
        kwargs.append(f'y="{spec["y"]}"')
        kwargs.append('z="count"')

    elif spec["chart_type"] == "bar" and style.get("horizontal"):
        kwargs.append(f'x="{spec["y"]}"')
        kwargs.append(f'y="{spec["x"]}"')
        kwargs.append('orientation="h"')
        if spec.get("color"):
            kwargs.append(f'color="{spec["color"]}"')
    else:
        kwargs.append(f'x="{spec["x"]}"')
        if spec.get("y"):
            kwargs.append(f'y="{spec["y"]}"')
        if spec.get("color"):
            kwargs.append(f'color="{spec["color"]}"')

    if style.get("facet_col"):
        kwargs.append(f'facet_col="{style["facet_col"]}"')

    if style.get("symbol") and spec["chart_type"] == "scatter":
        kwargs.append(f'symbol="{style["symbol"]}"')

    if style.get("size") and spec["chart_type"] == "scatter":
        kwargs.append(f'size="{style["size"]}"')

    if style.get("markers") and spec["chart_type"] == "line":
        kwargs.append("markers=True")

    if style.get("box_points") and spec["chart_type"] == "box":
        kwargs.append(f'points="{style["box_points"]}"')

    if style.get("text_auto") and spec["chart_type"] == "bar":
        kwargs.append("text_auto=True")

    if style.get("template"):
        kwargs.append(f'template="{style["template"]}"')

    return kwargs


def render_chart_code(spec: dict[str, Any], df_name: str, style: dict[str, Any]) -> str:
    fig_name = style["fig_name"]
    fn = "density_heatmap" if spec["chart_type"] == "heatmap" else spec["chart_type"]
    kwargs = build_chart_kwargs(spec, style)

    lines = [f"{fig_name} = px.{fn}(", f"    {df_name},"]
    for idx, kw in enumerate(kwargs):
        suffix = "," if idx < len(kwargs) - 1 else ""
        lines.append(f"    {kw}{suffix}")
    lines.append(")")
    return "\n".join(lines) + "\n"


def render_layout_code(spec: dict[str, Any], style: dict[str, Any]) -> str:
    fig_name = style["fig_name"]
    parts: list[str] = []

    if style.get("add_layout"):
        parts.append("height=500")

    if spec["chart_type"] == "bar" and style.get("barmode"):
        parts.append(f'barmode="{style["barmode"]}"')

    if not parts:
        return ""

    return f"{fig_name}.update_layout({', '.join(parts)})\n"


def render_code(spec: dict[str, Any], style: dict[str, Any]) -> str:
    if not spec.get("title"):
        spec["title"] = build_title(spec)

    lines = [
        "import streamlit as st",
        "import plotly.express as px",
        "",
        f'st.{style["title_func"]}("{spec["title"]}")',
        "",
    ]

    current_df = "df"

    current_df, filter_code = render_filter_code(spec.get("filter"), current_df)
    if filter_code:
        lines.append(filter_code.rstrip())
        lines.append("")

    current_df, widget_code = render_widget_code(spec, current_df)
    if widget_code:
        lines.append(widget_code.rstrip())
        lines.append("")

    current_df, agg_code = render_agg_df(spec, current_df, style)
    if agg_code:
        lines.append(agg_code.rstrip())
        lines.append("")
    elif style.get("sort_by_x") and spec["chart_type"] in {"scatter", "line", "histogram", "box"}:
        sorted_df = f"{current_df}_sorted"
        lines.append(f'{sorted_df} = {current_df}.sort_values(by="{spec["x"]}")')
        lines.append("")
        current_df = sorted_df

    lines.append(render_chart_code(spec, current_df, style).rstrip())

    layout_code = render_layout_code(spec, style)
    if layout_code:
        lines.append("")
        lines.append(layout_code.rstrip())

    lines.append("")
    lines.append(
        f'st.plotly_chart({style["fig_name"]}, use_container_width={style["use_container_width"]})'
    )

    return "\n".join(lines)


def choose_alt_color(schema: dict[str, str], spec: dict[str, Any]) -> str | None:
    candidates = [
        c for c in get_category_cols(schema)
        if c not in {spec.get("x"), spec.get("y"), spec.get("widget_column")}
    ]
    return random.choice(candidates) if candidates else None


def choose_facet_col(schema: dict[str, str], spec: dict[str, Any]) -> str | None:
    candidates = [
        c for c in get_category_cols(schema)
        if c not in {spec.get("x"), spec.get("y"), spec.get("color"), spec.get("widget_column")}
    ]
    return random.choice(candidates) if candidates else None


def choose_symbol(schema: dict[str, str], spec: dict[str, Any]) -> str | None:
    candidates = [
        c for c in get_category_cols(schema)
        if c not in {spec.get("x"), spec.get("y"), spec.get("color")}
    ]
    return random.choice(candidates) if candidates else None


def choose_size(schema: dict[str, str], spec: dict[str, Any]) -> str | None:
    candidates = [
        c for c in get_measure_cols(schema)
        if c not in {spec.get("x"), spec.get("y")}
    ]
    return random.choice(candidates) if candidates else None


def choose_widget_col(schema: dict[str, str], spec: dict[str, Any]) -> str | None:
    candidates = [
        c for c in get_category_cols(schema)
        if c not in {spec.get("x"), spec.get("y"), spec.get("color")}
    ]
    if not candidates:
        candidates = get_category_cols(schema)
    return random.choice(candidates) if candidates else None


def build_variant(base: dict[str, Any], variant_idx: int) -> dict[str, Any] | None:
    schema = copy.deepcopy(base["schema"])
    spec = copy.deepcopy(base["spec"])
    style = copy.deepcopy(base["style"])

    if not spec.get("x"):
        return None

    if spec["chart_type"] != "histogram" and spec["chart_type"] != "heatmap" and not spec.get("y"):
        return None

    if not spec.get("title"):
        spec["title"] = build_title(spec)

    mode = variant_idx % 8

    if mode == 0:
        style["title_func"] = random.choice(TITLE_FUNCS)
        style["fig_name"] = random.choice(FIG_NAMES)
        style["template"] = random.choice(TEMPLATES)
        style["add_layout"] = random.choice([True, False])
        style["use_container_width"] = random.choice(USE_CONTAINER_WIDTH_OPTIONS)

    elif mode == 1:
        # prompt/schema explosion only
        pass

    elif mode == 2:
        # add/remove color
        if spec["chart_type"] in {"scatter", "bar", "line", "histogram", "box"}:
            if spec.get("color"):
                spec["color"] = None
            else:
                spec["color"] = choose_alt_color(schema, spec)

            if spec["chart_type"] == "bar" and spec.get("color"):
                style["barmode"] = random.choice(["group", "stack"])
            else:
                style["barmode"] = None

    elif mode == 3:
        # add/remove filter
        if spec.get("filter"):
            spec["filter"] = None
        else:
            spec["filter"] = maybe_choose_filter(schema)

    elif mode == 4:
        # add/remove widget
        if spec.get("use_widget"):
            spec["use_widget"] = False
            spec["widget_column"] = None
        else:
            widget_col = choose_widget_col(schema, spec)
            if widget_col:
                spec["use_widget"] = True
                spec["widget_column"] = widget_col

    elif mode == 5:
        # chart-specific structural knobs
        if spec["chart_type"] == "line":
            style["markers"] = not style.get("markers", False)
            if not spec.get("color"):
                spec["color"] = choose_alt_color(schema, spec)
        elif spec["chart_type"] == "histogram":
            style["marginal"] = random.choice(MARGINALS)
        elif spec["chart_type"] == "box":
            style["box_points"] = random.choice(BOX_POINTS)
            if not spec.get("color"):
                spec["color"] = choose_alt_color(schema, spec)
        elif spec["chart_type"] == "bar":
            if not spec.get("color"):
                spec["color"] = choose_alt_color(schema, spec)
            style["barmode"] = random.choice(BARMODES)
            style["text_auto"] = random.choice([True, False])
        elif spec["chart_type"] == "scatter":
            style["symbol"] = choose_symbol(schema, spec)
            style["size"] = choose_size(schema, spec)
        elif spec["chart_type"] == "heatmap":
            spec["filter"] = spec.get("filter") or maybe_choose_filter(schema)

    elif mode == 6:
        # preprocessing structure
        if spec["chart_type"] in {"bar", "line"}:
            style["sort_by_x"] = True
        elif spec["chart_type"] == "bar":
            style["top_n"] = 10
        else:
            style["sort_by_x"] = True

    elif mode == 7:
        # combined higher-value structural variant
        style["title_func"] = random.choice(TITLE_FUNCS)
        style["fig_name"] = random.choice(FIG_NAMES)
        style["template"] = random.choice(TEMPLATES)
        style["add_layout"] = True
        spec["filter"] = spec.get("filter") or maybe_choose_filter(schema)

        if spec["chart_type"] in {"scatter", "bar", "line", "box", "histogram"} and not spec.get("color"):
            spec["color"] = choose_alt_color(schema, spec)

        if spec["chart_type"] == "scatter":
            style["facet_col"] = choose_facet_col(schema, spec)
            style["symbol"] = choose_symbol(schema, spec)
        elif spec["chart_type"] == "line":
            style["markers"] = True
            style["facet_col"] = choose_facet_col(schema, spec)
            style["sort_by_x"] = True
        elif spec["chart_type"] == "histogram":
            style["marginal"] = random.choice(["box", "rug"])
        elif spec["chart_type"] == "bar":
            style["barmode"] = random.choice(["group", "stack"])
            style["text_auto"] = True
        elif spec["chart_type"] == "box":
            style["box_points"] = random.choice(["all", "outliers"])

    spec["title"] = build_title(spec)

    request = paraphrase_request(spec, variant_idx)
    header = SCHEMA_HEADERS[variant_idx % len(SCHEMA_HEADERS)]
    shuffle_schema = variant_idx % 2 == 1
    user_prompt = f"User request:\n{request}\n\n{render_schema(schema, header, shuffle_schema)}"
    code = render_code(spec, style)

    return {
        "messages": [
            {"role": "system", "content": base["system"]},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": code},
        ]
    }


def dedupe_key(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def explode_rows(
    rows: list[dict[str, Any]],
    variants_per_example: int,
    output_path: Path,
    limit: int | None,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    written = 0
    skipped = 0
    seen: set[str] = set()

    subset = rows[:limit] if limit is not None else rows

    for base_idx, row in enumerate(subset, start=1):
        try:
            parsed = parse_example(row)
        except Exception:
            skipped += 1
            continue

        made = 0
        attempts = 0
        max_attempts = max(variants_per_example * 4, 20)

        while made < variants_per_example and attempts < max_attempts:
            attempts += 1
            variant_idx = made + attempts + base_idx
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
    parser = argparse.ArgumentParser(
        description="Explode passing plotly/streamlit training examples into structural and prompt variants."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants-per-example", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = read_jsonl(args.input)
    written, skipped = explode_rows(
        rows=rows,
        variants_per_example=args.variants_per_example,
        output_path=args.output,
        limit=args.limit,
    )

    print(f"Base examples read: {len(rows) if args.limit is None else min(len(rows), args.limit)}")
    print(f"Examples skipped during parse: {skipped}")
    print(f"Exploded examples written: {written}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
