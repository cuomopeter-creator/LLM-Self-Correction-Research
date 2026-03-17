from __future__ import annotations
import argparse
import json
import sys
import types
from pathlib import Path

import pandas as pd
import plotly.express as px


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "datasets" / "plotly_streamlit_train_revised.jsonl"
PASS_PATH = BASE_DIR / "datasets" / "plotly_streamlit_train_runtime_pass.jsonl"
FAIL_PATH = BASE_DIR / "datasets" / "plotly_streamlit_train_runtime_fail.jsonl"


def load_examples(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield str(i), json.loads(line)


def parse_schema(user_msg: str) -> dict[str, str]:
    schema = {}
    lines = user_msg.splitlines()
    capture = False

    headings = {
        "Available columns:",
        "Columns in df:",
        "Dataframe columns:",
    }

    for line in lines:
        s = line.strip()

        if s in headings:
            capture = True
            continue

        if capture:
            if not s:
                continue
            if ":" not in s:
                continue
            col, dtype = s.split(":", 1)
            schema[col.strip()] = dtype.strip().lower()

    return schema


def make_dummy_df(schema: dict[str, str]) -> pd.DataFrame:
    n = 8
    data = {}

    for col, dtype in schema.items():
        if "float" in dtype:
            data[col] = [10.5, 12.0, 9.8, 15.2, 8.7, 14.1, 11.3, 13.9]
        elif "int" in dtype:
            if col.lower() == "year":
                data[col] = [2020, 2021, 2022, 2023, 2020, 2021, 2022, 2023]
            else:
                data[col] = [20, 25, 30, 35, 40, 45, 50, 55]
        elif "datetime" in dtype:
            data[col] = pd.date_range("2024-01-01", periods=n, freq="D")
        elif "category" in dtype or "string" in dtype or "object" in dtype:
            if col.lower() == "product":
                data[col] = ["A", "B", "A", "C", "B", "A", "C", "B"]
            elif col.lower() == "region":
                data[col] = ["East", "West", "North", "South", "East", "West", "North", "South"]
            elif col.lower() == "month":
                data[col] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
            elif col.lower() == "segment":
                data[col] = ["Consumer", "Corporate", "Home Office", "Consumer", "Corporate", "Home Office", "Consumer", "Corporate"]
            else:
                data[col] = ["X", "Y", "Z", "X", "Y", "Z", "X", "Y"]
        else:
            data[col] = [1, 2, 3, 4, 5, 6, 7, 8]

    return pd.DataFrame(data)


class StreamlitStub:
    def __init__(self):
        self.plot_calls = 0

    def title(self, *args, **kwargs): return None
    def subheader(self, *args, **kwargs): return None
    def header(self, *args, **kwargs): return None
    def write(self, *args, **kwargs): return None
    def markdown(self, *args, **kwargs): return None
    def warning(self, *args, **kwargs): return None
    def error(self, *args, **kwargs): return None
    def dataframe(self, *args, **kwargs): return None
    def table(self, *args, **kwargs): return None
    def metric(self, *args, **kwargs): return None

    def selectbox(self, label, options, index=0, **kwargs):
        if options:
            return options[index] if index < len(options) else options[0]
        return None

    def multiselect(self, label, options, default=None, **kwargs):
        if default is not None:
            return default
        return options[:1] if options else []

    def slider(self, label, min_value=None, max_value=None, value=None, **kwargs):
        if value is not None:
            return value
        if min_value is not None:
            return min_value
        return 0

    def radio(self, label, options, index=0, **kwargs):
        if options:
            return options[index] if index < len(options) else options[0]
        return None

    def checkbox(self, label, value=False, **kwargs):
        return value

    def plotly_chart(self, fig, **kwargs):
        self.plot_calls += 1
        return None


def make_streamlit_module(stub: StreamlitStub):
    mod = types.ModuleType("streamlit")

    for name in dir(stub):
        if not name.startswith("_"):
            setattr(mod, name, getattr(stub, name))

    return mod


def validate_runtime(code: str, user_msg: str) -> tuple[bool, str | None]:
    schema = parse_schema(user_msg)
    df = make_dummy_df(schema)
    st_stub = StreamlitStub()
    st_module = make_streamlit_module(st_stub)

    original_streamlit = sys.modules.get("streamlit")
    sys.modules["streamlit"] = st_module

    globals_dict = {
        "__builtins__": __builtins__,
        "pd": pd,
        "px": px,
        "df": df,
    }

    try:
        exec(code, globals_dict, globals_dict)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        if original_streamlit is not None:
            sys.modules["streamlit"] = original_streamlit
        else:
            sys.modules.pop("streamlit", None)

    if st_stub.plot_calls == 0:
        return False, "No st.plotly_chart call at runtime"

    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    passed = 0
    failed = 0
    processed = 0

    with PASS_PATH.open("w", encoding="utf-8") as fpass, FAIL_PATH.open("w", encoding="utf-8") as ffail:
        for ex_id, rec in load_examples(INPUT_PATH):
            if args.limit is not None and processed >= args.limit:
                break

            user_msg = rec["messages"][1]["content"]
            code = rec["messages"][2]["content"]

            ok, err = validate_runtime(code, user_msg)

            if ok:
                fpass.write(json.dumps(rec, ensure_ascii=False) + "\n")
                passed += 1
            else:
                rec["runtime_error"] = err
                ffail.write(json.dumps(rec, ensure_ascii=False) + "\n")
                failed += 1

            processed += 1
            print(f"[{processed}] ok={ok} ex_id={ex_id} err={err}")

    print()
    print("Processed:", processed)
    print("Passed:", passed)
    print("Failed:", failed)
    print("Pass file:", PASS_PATH)
    print("Fail file:", FAIL_PATH)


if __name__ == "__main__":
    main()
