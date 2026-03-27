from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]

RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"
FALLBACK_JSONL_PATH = PROJECT_ROOT / "experiments" / "prompt_visual_inspection.jsonl"


def get_latest_run_path() -> Path:
    run_files = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_files:
        return run_files[0]
    return FALLBACK_JSONL_PATH


JSONL_PATH = get_latest_run_path()
METRICS_PATH = JSONL_PATH.with_name(f"{JSONL_PATH.stem}_ratings.json")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_saved_metrics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_saved_metrics(path: Path, metrics: dict[str, dict[str, Any]]) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def extract_python(code: str) -> str:
    code = (code or "").strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return code


def strip_streamlit_imports(code: str) -> str:
    cleaned_lines = []
    for line in code.splitlines():
        s = line.strip()
        if s == "import streamlit as st":
            continue
        if s.startswith("from streamlit import"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def ensure_dataframe(df_like: Any) -> pd.DataFrame:
    if isinstance(df_like, pd.DataFrame):
        return df_like.copy()

    if isinstance(df_like, str):
        text = df_like.strip()
        if not text:
            return pd.DataFrame()
        try:
            parsed = json.loads(text)
            return ensure_dataframe(parsed)
        except json.JSONDecodeError:
            raise ValueError("record['df'] is a string but not valid JSON")

    if isinstance(df_like, list):
        if not df_like:
            return pd.DataFrame()
        if all(isinstance(x, dict) for x in df_like):
            return pd.DataFrame(df_like)
        raise ValueError("record['df'] list must contain dict rows")

    if isinstance(df_like, dict):
        if "data" in df_like:
            return ensure_dataframe(df_like["data"])
        if "rows" in df_like:
            return ensure_dataframe(df_like["rows"])
        if "records" in df_like:
            return ensure_dataframe(df_like["records"])
        return pd.DataFrame(df_like)

    raise ValueError(f"Unsupported df payload type: {type(df_like).__name__}")


def extract_record_df(record: dict[str, Any]) -> pd.DataFrame:
    if "df" not in record:
        raise ValueError("This record does not contain a 'df' field")
    df = ensure_dataframe(record["df"])

    for col in df.columns:
        if isinstance(col, str):
            lower = col.lower()
            if "date" in lower or lower.endswith("_at"):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

    return df


@dataclass
class StreamlitProxy:
    container: Any
    prefix: str
    last_fig: object | None = None

    def _with_key(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        keyed_widgets = {
            "selectbox",
            "multiselect",
            "radio",
            "checkbox",
            "toggle",
            "slider",
            "select_slider",
            "number_input",
            "text_input",
            "text_area",
            "date_input",
            "time_input",
            "file_uploader",
            "download_button",
            "button",
            "plotly_chart",
        }
        if name in keyed_widgets and "key" not in kwargs:
            kwargs["key"] = f"{self.prefix}_{name}_{abs(hash(str(sorted(kwargs.items()))))}"
        return kwargs

    def plotly_chart(self, fig, **kwargs):
        self.last_fig = fig
        kwargs = self._with_key("plotly_chart", kwargs)
        return self.container.plotly_chart(fig, **kwargs)

    def __getattr__(self, name: str):
        target = getattr(self.container, name)

        def wrapper(*args, **kwargs):
            kwargs = self._with_key(name, kwargs)
            return target(*args, **kwargs)

        return wrapper


def execute_chart_code(code: str, df: pd.DataFrame, record_key: str, output_container):
    cleaned = strip_streamlit_imports(extract_python(code))
    proxy = StreamlitProxy(container=output_container, prefix=record_key)

    local_vars = {
        "df": df.copy(),
        "st": proxy,
        "pd": pd,
        "px": px,
        "go": go,
    }

    exec(cleaned, local_vars, local_vars)

    fig = local_vars.get("fig")
    if fig is None:
        fig = local_vars.get("plot_fig")
    if fig is None:
        fig = local_vars.get("chart")
    if fig is None:
        fig = proxy.last_fig

    return fig


def get_record_id(record: dict[str, Any], idx: int) -> str:
    raw_id = record.get("id")
    if raw_id is None:
        return str(idx)
    return str(raw_id)


st.set_page_config(page_title="Prompt Visual Inspector", layout="wide")
st.title("Prompt Visual Inspector")
st.caption(f"Source: {JSONL_PATH.name}")
st.caption(f"Ratings file: {METRICS_PATH.name}")

records = json.loads(JSONL_PATH.read_text())["items"]
saved_metrics = load_saved_metrics(METRICS_PATH)

show_code = st.sidebar.checkbox("Show code", value=True)
show_dataset = st.sidebar.checkbox("Show dataset preview", value=False)
page_size = st.sidebar.number_input("Graphs per page", min_value=1, max_value=25, value=10, step=1)
page = st.sidebar.number_input("Page", min_value=1, value=1, step=1)

total_pages = max(1, math.ceil(len(records) / page_size))
page = min(page, total_pages)
start = (page - 1) * page_size
end = start + page_size
page_records = records[start:end]

st.sidebar.write(f"Prompts loaded: {len(records)}")
st.sidebar.write(f"Page {page} of {total_pages}")
st.sidebar.write(f"Ratings saved: {len(saved_metrics)}")

for i, record in enumerate(page_records, start=start):
    record_id = get_record_id(record, i)
    prompt = (record.get("prompt") or "").strip()
    response = (record.get("response") or "").strip()
    existing = saved_metrics.get(record_id, {})

    st.divider()
    st.subheader(f"{record_id}. {prompt}")

    if not response:
        st.warning("No response saved yet.")
        continue

    try:
        df = extract_record_df(record)
    except Exception as e:
        st.error(f"Could not load record dataframe: {e}")
        continue

    info_cols = st.columns([2, 1])
    with info_cols[0]:
        st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    with info_cols[1]:
        if existing:
            st.caption(
                f"Saved scores → correctness: {existing.get('correctness')}, "
                f"code: {existing.get('code_quality')}, "
                f"chart: {existing.get('chart_quality')}"
            )

    if show_dataset:
        with st.expander("Dataset preview", expanded=False):
            st.dataframe(df.head(25), use_container_width=True)

    if show_code:
        with st.expander("Code", expanded=False):
            st.code(extract_python(response), language="python")

    output_container = st.container()

    try:
        fig = execute_chart_code(
            code=response,
            df=df,
            record_key=f"record_{record_id}",
            output_container=output_container,
        )
    except Exception as e:
        output_container.error(f"Execution failed: {e}")

    with st.form(key=f"rating_form_{record_id}"):
        correctness = st.radio(
            "Correctness",
            options=[1, 0],
            format_func=lambda x: "1 = correct" if x == 1 else "0 = incorrect",
            index=0 if existing.get("correctness", 1) == 1 else 1,
            horizontal=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            code_quality = st.select_slider(
                "Code quality",
                options=[1, 2, 3, 4, 5],
                value=existing.get("code_quality", 3),
            )
        with col2:
            chart_quality = st.select_slider(
                "Chart quality",
                options=[1, 2, 3, 4, 5],
                value=existing.get("chart_quality", 3),
            )

        notes = st.text_area(
            "Notes",
            value=existing.get("notes", ""),
            placeholder="Optional notes about correctness, code issues, or chart quality",
        )

        submitted = st.form_submit_button("Save rating")

        if submitted:
            saved_metrics[record_id] = {
                "id": record_id,
                "prompt": prompt,
                "correctness": int(correctness),
                "code_quality": int(code_quality),
                "chart_quality": int(chart_quality),
                "notes": notes,
            }
            save_saved_metrics(METRICS_PATH, saved_metrics)
            st.success("Rating saved.")

if saved_metrics:
    st.divider()
    st.subheader("Ratings summary")

    metrics_df = pd.DataFrame(saved_metrics.values())

    if not metrics_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rated prompts", int(len(metrics_df)))
        with col2:
            st.metric("Correctness rate", f"{metrics_df['correctness'].mean():.1%}")
        with col3:
            st.metric("Avg code quality", f"{metrics_df['code_quality'].mean():.2f}")
        with col4:
            st.metric("Avg chart quality", f"{metrics_df['chart_quality'].mean():.2f}")

        with st.expander("All saved ratings", expanded=False):
            st.dataframe(metrics_df.sort_values("id"), use_container_width=True)
