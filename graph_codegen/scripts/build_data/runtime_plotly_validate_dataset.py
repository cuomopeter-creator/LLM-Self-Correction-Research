from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from rich.json import args


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"

def resolve_dataset_path(p: str | Path) -> Path:
    p = Path(p)

    # absolute path → use directly
    if p.is_absolute():
        return p

    # if user passed just a filename, assume datasets/
    if p.parent == Path("."):
        candidate = DATASET_DIR / p
        if candidate.exists():
            return candidate

    # otherwise resolve relative to cwd
    return p.resolve()


DEFAULT_INPUT_PATH = DATASET_DIR / "plotly_streamlit_train_revised.jsonl"
DEFAULT_PASS_PATH = DATASET_DIR / "plotly_streamlit_train_runtime_pass.jsonl"
DEFAULT_FAIL_PATH = DATASET_DIR / "plotly_streamlit_train_runtime_fail.jsonl"

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
        col_lower = col.lower()

        if "float" in dtype:
            if col_lower == "profit":
                data[col] = [2.1, 3.4, -1.2, 4.8, 1.7, 5.1, 2.9, 3.8]
            else:
                data[col] = [10.5, 12.0, 9.8, 15.2, 8.7, 14.1, 11.3, 13.9]

        elif "int" in dtype:
            if col_lower == "year":
                data[col] = [2020, 2021, 2022, 2023, 2020, 2021, 2022, 2023]
            elif col_lower in {"units", "orders"}:
                data[col] = [5, 8, 6, 10, 7, 9, 4, 11]
            elif col_lower == "customer_age":
                data[col] = [22, 29, 34, 41, 27, 38, 45, 31]
            else:
                data[col] = [20, 25, 30, 35, 40, 45, 50, 55]

        elif "datetime" in dtype:
            data[col] = pd.date_range("2024-01-01", periods=n, freq="MS")

        elif "category" in dtype or "string" in dtype or "object" in dtype:
            if col_lower == "product":
                data[col] = ["A", "B", "A", "C", "B", "A", "C", "B"]
            elif col_lower == "region":
                data[col] = ["East", "West", "North", "South", "East", "West", "North", "South"]
            elif col_lower == "month":
                data[col] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
            elif col_lower == "segment":
                data[col] = [
                    "Consumer",
                    "Corporate",
                    "Home Office",
                    "Consumer",
                    "Corporate",
                    "Home Office",
                    "Consumer",
                    "Corporate",
                ]
            elif col_lower == "date":
                data[col] = [
                    "2024-01",
                    "2024-02",
                    "2024-03",
                    "2024-04",
                    "2024-05",
                    "2024-06",
                    "2024-07",
                    "2024-08",
                ]
            else:
                data[col] = ["X", "Y", "Z", "X", "Y", "Z", "X", "Y"]

        else:
            data[col] = [1, 2, 3, 4, 5, 6, 7, 8]

    df = pd.DataFrame(data)

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            pass

    return df


class _ContextStub:
    def __init__(self, root_stub: "StreamlitStub"):
        self._root = root_stub

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name):
        return getattr(self._root, name)


class StreamlitStub:
    def __init__(self):
        self.plot_calls = 0
        self.sidebar = self

    def title(self, *args, **kwargs): return None
    def subheader(self, *args, **kwargs): return None
    def header(self, *args, **kwargs): return None
    def write(self, *args, **kwargs): return None
    def markdown(self, *args, **kwargs): return None
    def caption(self, *args, **kwargs): return None
    def text(self, *args, **kwargs): return None
    def success(self, *args, **kwargs): return None
    def info(self, *args, **kwargs): return None
    def warning(self, *args, **kwargs): return None
    def error(self, *args, **kwargs): return None
    def dataframe(self, *args, **kwargs): return None
    def table(self, *args, **kwargs): return None
    def metric(self, *args, **kwargs): return None
    def divider(self, *args, **kwargs): return None

    def selectbox(self, label, options, index=0, **kwargs):
        options = list(options) if options is not None else []
        if options:
            return options[index] if index < len(options) else options[0]
        return None

    def multiselect(self, label, options, default=None, **kwargs):
        options = list(options) if options is not None else []
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
        options = list(options) if options is not None else []
        if options:
            return options[index] if index < len(options) else options[0]
        return None

    def checkbox(self, label, value=False, **kwargs):
        return value

    def columns(self, n, **kwargs):
        if isinstance(n, int):
            count = n
        else:
            count = len(n)
        return [_ContextStub(self) for _ in range(count)]

    def tabs(self, labels):
        return [_ContextStub(self) for _ in labels]

    def container(self, **kwargs):
        return _ContextStub(self)

    def expander(self, label, expanded=False, **kwargs):
        return _ContextStub(self)

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
        "go": go,
        "make_subplots": make_subplots,
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--pass-path", type=Path, default=DEFAULT_PASS_PATH)
    parser.add_argument("--fail-path", type=Path, default=DEFAULT_FAIL_PATH)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.input = resolve_dataset_path(args.input)
    args.pass_path = resolve_dataset_path(args.pass_path)
    args.fail_path = resolve_dataset_path(args.fail_path)

    args.pass_path.parent.mkdir(parents=True, exist_ok=True)
    args.fail_path.parent.mkdir(parents=True, exist_ok=True)

    passed = 0
    failed = 0
    processed = 0

    with args.pass_path.open("w", encoding="utf-8") as fpass, args.fail_path.open("w", encoding="utf-8") as ffail:
        for ex_id, rec in load_examples(args.input):
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
    print("Pass file:", args.pass_path)
    print("Fail file:", args.fail_path)


if __name__ == "__main__":
    main()
