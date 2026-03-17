from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --- universal project paths ---
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[0] if (HERE.parent / "datasets").exists() else HERE.parents[1]

DATASETS_DIR = PROJECT_ROOT / "datasets"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

JSONL_PATH = DATASETS_DIR / "prompt_visual_inspection.jsonl"
DATA_PATH = ANALYSIS_DIR / "master_results.csv"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_df(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def extract_python(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return code


st.set_page_config(page_title="Prompt Visual Inspector", layout="wide")
st.title("Prompt Visual Inspector")

records = load_jsonl(JSONL_PATH)
df = load_df(DATA_PATH)

st.sidebar.header("Controls")
show_code = st.sidebar.checkbox("Show code", value=True)
only_missing = st.sidebar.checkbox("Only show missing responses", value=False)

st.sidebar.write(f"Prompts loaded: {len(records)}")
st.sidebar.write(f"Dataset shape: {df.shape}")

for record in records:
    response = (record.get("response") or "").strip()

    if only_missing and response:
        continue

    st.divider()
    st.subheader(f"{record['id']}. {record['prompt']}")

    meta1, meta2, meta3 = st.columns(3)
    meta1.write(f"**Expected chart:** {record.get('expected_chart', '')}")
    meta2.write(f"**Expected columns:** {record.get('expected_columns', [])}")
    meta3.write(f"**Exec pass:** {record.get('exec_pass', None)}")

    if not response:
        st.warning("No response saved yet.")
        continue

    code = extract_python(response)

    if show_code:
        with st.expander("Code", expanded=False):
            st.code(code, language="python")

    chart_box = st.container()
    try:
        local_vars = {"df": df}
        exec(code, {}, local_vars)

        fig = local_vars.get("fig")
        if fig is not None:
            chart_box.plotly_chart(fig, use_container_width=True)
        else:
            chart_box.info("Code executed, but no variable named `fig` was found.")
    except Exception as e:
        chart_box.error(f"Execution failed: {e}")

    notes_default = record.get("notes", "")
    status_default = record.get("status", "")

    c1, c2 = st.columns([1, 3])
    with c1:
        st.selectbox(
            "Status",
            options=["", "pass", "fail", "needs review"],
            index=["", "pass", "fail", "needs review"].index(status_default)
            if status_default in {"", "pass", "fail", "needs review"}
            else 0,
            key=f"status_{record['id']}",
        )
    with c2:
        st.text_input(
            "Notes",
            value=notes_default,
            key=f"notes_{record['id']}",
        )