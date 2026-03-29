import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "datasets" / "plotly_streamlit_train_raw_3000.jsonl"

AGGREGATIONS = ["sum", "mean", "median", "count", "min", "max"]

COLUMNS = [
    ("sales", "float"),
    ("profit", "float"),
    ("revenue", "float"),
    ("units", "int"),
    ("customer_age", "int"),
    ("region", "category"),
    ("product", "category"),
    ("segment", "category"),
    ("year", "int"),
    ("month", "category"),
    ("date", "datetime"),
]

REQUEST_PREFIXES = ["Create", "Build", "Show", "Plot", "Visualize", "Generate"]
SCHEMA_HEADERS = ["Dataframe columns:", "Columns in df:", "Available columns:"]
TITLE_FUNCS = ["title", "header", "subheader"]
FIG_NAMES = ["fig", "chart", "plot_fig"]
USE_CONTAINER_WIDTH_OPTIONS = [True, False]
TEMPLATES = ["plotly_white", "plotly", "simple_white", None]
ADD_LAYOUT_OPTIONS = [True, False]

MEASURE_COLS = {"sales", "profit", "revenue", "units", "customer_age"}
CATEGORY_COLS = {"region", "product", "segment"}
TIME_COLS = {"date"}
TIME_PART_COLS = {"year", "month"}


def get_measure_cols(schema):
    return [k for k in schema if k in MEASURE_COLS]


def get_category_role_cols(schema):
    return [k for k in schema if k in CATEGORY_COLS]


def get_time_role_cols(schema):
    return [k for k in schema if k in TIME_COLS]


def get_time_part_role_cols(schema):
    return [k for k in schema if k in TIME_PART_COLS]


def sample_schema():
    cols = random.sample(COLUMNS, k=5)
    names = [name for name, _ in cols]

    if not any(n in names for n in MEASURE_COLS):
        cols[0] = random.choice(
            [("sales", "float"), ("profit", "float"), ("revenue", "float"), ("units", "int")]
        )

    if not any(n in names for n in CATEGORY_COLS | TIME_PART_COLS | TIME_COLS):
        cols[1] = random.choice(
            [("region", "category"), ("product", "category"), ("segment", "category"), ("date", "datetime"), ("year", "int"), ("month", "category")]
        )

    return {name: dtype for name, dtype in cols}


def maybe_choose_color(schema):
    candidates = get_category_role_cols(schema)
    return random.choice(candidates) if candidates and random.random() < 0.55 else None


def maybe_choose_filter(schema):
    candidates = []

    for col in schema:
        if col in CATEGORY_COLS:
            if col == "region":
                candidates.append((col, "==", random.choice(["North", "South", "East", "West"])))
            elif col == "segment":
                candidates.append((col, "==", random.choice(["Consumer", "Corporate", "Home Office"])))
            elif col == "product":
                candidates.append((col, "==", random.choice(["A", "B", "C"])))
        elif col == "year":
            candidates.append((col, "==", random.choice([2022, 2023, 2024])))
        elif col in MEASURE_COLS and random.random() < 0.25:
            candidates.append((col, ">", random.choice([10, 50, 100])))

    return random.choice(candidates) if candidates and random.random() < 0.45 else None


def choose_intent(schema):
    measures = get_measure_cols(schema)
    categories = get_category_role_cols(schema)
    time_cols = get_time_role_cols(schema)
    time_parts = get_time_part_role_cols(schema)

    intents = []

    if len(measures) >= 2:
        intents.append("scatter")

    if measures:
        intents.append("histogram")

    if measures and categories:
        intents.append("box")
        intents.append("bar_agg")

    if measures and (time_cols or time_parts):
        intents.append("line_time")

    if measures and categories and len(categories) >= 2:
        intents.append("heatmap_cat")

    return random.choice(intents) if intents else None


def build_spec(schema):
    intent = choose_intent(schema)
    if intent is None:
        return None

    measures = get_measure_cols(schema)
    categories = get_category_role_cols(schema)
    time_cols = get_time_role_cols(schema)
    time_parts = get_time_part_role_cols(schema)

    spec = {
        "chart_type": None,
        "x": None,
        "y": None,
        "color": None,
        "aggregation": None,
        "filter": maybe_choose_filter(schema),
        "use_widget": False,
        "widget_column": None,
        "title": None,
    }

    if intent == "scatter":
        x, y = random.sample(measures, 2)
        spec["chart_type"] = "scatter"
        spec["x"] = x
        spec["y"] = y
        spec["color"] = maybe_choose_color(schema)

    elif intent == "bar_agg":
        spec["chart_type"] = "bar"
        spec["x"] = random.choice(categories + time_parts)
        spec["y"] = random.choice(measures)
        spec["aggregation"] = random.choice(AGGREGATIONS)
        spec["color"] = None

    elif intent == "box":
        spec["chart_type"] = "box"
        spec["x"] = random.choice(categories)
        spec["y"] = random.choice(measures)

    elif intent == "histogram":
        spec["chart_type"] = "histogram"
        spec["x"] = random.choice(measures)
        spec["y"] = None
        spec["color"] = maybe_choose_color(schema)

    elif intent == "line_time":
        time_axis = time_cols + time_parts
        spec["chart_type"] = "line"
        spec["x"] = random.choice(time_axis)
        spec["y"] = random.choice(measures)
        spec["aggregation"] = random.choice(["sum", "mean"])
        spec["color"] = maybe_choose_color(schema)

    elif intent == "heatmap_cat":
        cat_x, cat_y = random.sample(categories, 2)
        spec["chart_type"] = "heatmap"
        spec["x"] = cat_x
        spec["y"] = cat_y
        spec["aggregation"] = "count"
        spec["color"] = None

    widget_candidates = categories
    if widget_candidates and random.random() < 0.3:
        spec["use_widget"] = True
        spec["widget_column"] = random.choice(widget_candidates)

    spec["title"] = build_title(spec)
    return spec


def build_title(spec):
    if spec["chart_type"] == "histogram":
        return f"Distribution of {spec['x'].replace('_', ' ').title()}"

    if spec["chart_type"] == "bar":
        return f"{spec['aggregation'].title()} {spec['y'].replace('_', ' ').title()} by {spec['x'].replace('_', ' ').title()}"

    if spec["chart_type"] == "line":
        return f"{spec['aggregation'].title()} {spec['y'].replace('_', ' ').title()} over {spec['x'].replace('_', ' ').title()}"

    if spec["chart_type"] == "box":
        return f"{spec['y'].replace('_', ' ').title()} by {spec['x'].replace('_', ' ').title()}"

    if spec["chart_type"] == "heatmap":
        return f"Count of {spec['x'].replace('_', ' ').title()} vs {spec['y'].replace('_', ' ').title()}"

    return f"{spec['x'].replace('_', ' ').title()} vs {spec['y'].replace('_', ' ').title()}"


def render_request(spec):
    verb = random.choice(REQUEST_PREFIXES)

    if spec["chart_type"] == "scatter":
        req = f"{verb} a scatter plot of {spec['x']} vs {spec['y']}"
        if spec["color"]:
            req += f" colored by {spec['color']}"

    elif spec["chart_type"] == "bar":
        req = f"{verb} a bar chart of {spec['aggregation']} {spec['y']} by {spec['x']}"

    elif spec["chart_type"] == "box":
        req = f"{verb} a box plot of {spec['y']} by {spec['x']}"

    elif spec["chart_type"] == "histogram":
        req = f"{verb} a histogram of {spec['x']}"
        if spec["color"]:
            req += f" colored by {spec['color']}"

    elif spec["chart_type"] == "line":
        req = f"{verb} a line chart of {spec['aggregation']} {spec['y']} over {spec['x']}"
        if spec["color"]:
            req += f" colored by {spec['color']}"

    else:
        req = f"{verb} a heatmap of count by {spec['x']} and {spec['y']}"

    if spec["filter"]:
        col, op, value = spec["filter"]
        req += f" where {col} {op} {value}"

    if spec["use_widget"] and spec["widget_column"]:
        req += f" with a Streamlit selectbox for {spec['widget_column']}"

    return req


def render_schema(schema):
    header = random.choice(SCHEMA_HEADERS)
    body = "\n".join(f"{k}: {v}" for k, v in schema.items())
    return f"{header}\n{body}"


def render_filter_code(filter_tuple, df_name):
    if not filter_tuple:
        return df_name, ""

    col, op, value = filter_tuple
    value_repr = f'"{value}"' if isinstance(value, str) else str(value)
    new_df = f"{df_name}_filtered"
    code = f'{new_df} = {df_name}[{df_name}["{col}"] {op} {value_repr}]\n'
    return new_df, code


def render_widget_code(spec, current_df):
    if not spec["use_widget"] or not spec["widget_column"]:
        return current_df, ""

    col = spec["widget_column"]
    new_df = f"{current_df}_widget"
    code = (
        f'options = ["All"] + sorted(df["{col}"].dropna().unique().tolist())\n'
        f'selected_{col} = st.selectbox("{col.replace("_", " ").title()}", options)\n'
        f'if selected_{col} == "All":\n'
        f'    {new_df} = {current_df}\n'
        f'else:\n'
        f'    {new_df} = {current_df}[{current_df}["{col}"] == selected_{col}]\n'
    )
    return new_df, code


def render_agg_df(spec, base_df):
    if not spec["aggregation"]:
        return base_df, ""

    if spec["chart_type"] in {"bar", "line"}:
        agg_df = "df_plot"
        agg = spec["aggregation"]
        code = f'{agg_df} = {base_df}.groupby("{spec["x"]}", as_index=False)["{spec["y"]}"].{agg}()\n'
        return agg_df, code

    if spec["chart_type"] == "heatmap":
        agg_df = "df_plot"
        code = (
            f'{agg_df} = {base_df}.groupby(["{spec["x"]}", "{spec["y"]}"], as_index=False)'
            f'.size().rename(columns={{"size": "count"}})\n'
        )
        return agg_df, code

    return base_df, ""


def render_chart_code(spec, df_name, fig_name, template):
    template_part = f',\n    template="{template}"' if template else ""

    if spec["chart_type"] == "histogram":
        color_part = f',\n    color="{spec["color"]}"' if spec["color"] else ""
        return (
            f'{fig_name} = px.histogram(\n'
            f'    {df_name},\n'
            f'    x="{spec["x"]}"{color_part}{template_part}\n'
            f')\n'
        )

    if spec["chart_type"] == "heatmap":
        return (
            f'{fig_name} = px.density_heatmap(\n'
            f'    {df_name},\n'
            f'    x="{spec["x"]}",\n'
            f'    y="{spec["y"]}"{template_part}\n'
            f')\n'
        )

    color_part = f',\n    color="{spec["color"]}"' if spec["color"] else ""

    return (
        f'{fig_name} = px.{spec["chart_type"]}(\n'
        f'    {df_name},\n'
        f'    x="{spec["x"]}",\n'
        f'    y="{spec["y"]}"{color_part}{template_part}\n'
        f')\n'
    )


def render_code(spec):
    title_func = random.choice(TITLE_FUNCS)
    fig_name = random.choice(FIG_NAMES)
    use_container_width = random.choice(USE_CONTAINER_WIDTH_OPTIONS)
    template = random.choice(TEMPLATES)
    add_layout = random.choice(ADD_LAYOUT_OPTIONS)

    lines = [
        "import streamlit as st",
        "import plotly.express as px",
        "",
        f'st.{title_func}("{spec["title"]}")',
        "",
    ]

    current_df, filter_code = render_filter_code(spec["filter"], "df")
    if filter_code:
        lines.append(filter_code.rstrip())
        lines.append("")

    current_df, widget_code = render_widget_code(spec, current_df)
    if widget_code:
        lines.append(widget_code.rstrip())
        lines.append("")

    current_df, agg_code = render_agg_df(spec, current_df)
    if agg_code:
        lines.append(agg_code.rstrip())
        lines.append("")

    lines.append(render_chart_code(spec, current_df, fig_name, template).rstrip())

    if add_layout and spec["chart_type"] != "heatmap":
        lines.append("")
        lines.append(f'{fig_name}.update_layout(height=500)')

    lines.append("")
    lines.append(f"st.plotly_chart({fig_name}, use_container_width={use_container_width})")

    return "\n".join(lines)


def generate_example():
    schema = sample_schema()
    spec = build_spec(schema)
    if spec is None:
        return None

    prompt = f"User request:\n{render_request(spec)}\n\n{render_schema(schema)}"
    code = render_code(spec)

    return {
        "messages": [
            {
                "role": "system",
                "content": "Generate a Streamlit app snippet using plotly.express only. Assume df already exists. Return code only."
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": code
            }
        ]
    }


def main(n=3000):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        written = 0
        while written < n:
            ex = generate_example()
            if ex is None:
                continue
            f.write(json.dumps(ex) + "\n")
            written += 1

    print(f"Generated {written} examples at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
