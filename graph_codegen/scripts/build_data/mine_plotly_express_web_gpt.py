from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import requests
from dotenv import load_dotenv


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
DATA_DIR = HERE.parents[1] / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(PROJECT_ROOT / ".env", override=True)


SYSTEM_PROMPT = (
    "Generate a Streamlit app snippet using plotly.express only. "
    "Assume df already exists. Return code only."
)

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

CHART_SPECS = [
    {"chart": "scatter", "goal": "compare two numeric measures", "columns": ["sales", "profit", "segment", "region", "month"]},
    {"chart": "line", "goal": "show a time trend", "columns": ["date", "revenue", "segment", "region", "year"]},
    {"chart": "bar", "goal": "compare aggregated values across categories", "columns": ["region", "sales", "profit", "units", "segment"]},
    {"chart": "histogram", "goal": "show a distribution", "columns": ["customer_age", "sales", "profit", "segment", "region"]},
    {"chart": "box", "goal": "compare distributions across groups", "columns": ["product", "sales", "profit", "segment", "region"]},
    {"chart": "violin", "goal": "compare detailed distributions across groups", "columns": ["segment", "profit", "sales", "region", "product"]},
    {"chart": "density_heatmap", "goal": "show density or cross-category intensity", "columns": ["region", "product", "sales", "profit", "segment"]},
    {"chart": "area", "goal": "show cumulative change over time", "columns": ["date", "sales", "segment", "region", "year"]},
    {"chart": "strip", "goal": "show individual observations by category", "columns": ["segment", "sales", "profit", "region", "product"]},
    {"chart": "scatter_matrix", "goal": "compare several numeric features", "columns": ["sales", "profit", "revenue", "units", "customer_age"]},
    {"chart": "timeline-like line", "goal": "show progression over dates", "columns": ["date", "units", "region", "segment", "product"]},
    {"chart": "facet scatter", "goal": "compare relationships across subgroups", "columns": ["sales", "profit", "segment", "region", "product"]},
]

BUSINESS_CONTEXTS = [
    "retail performance dashboard",
    "sales team reporting",
    "regional revenue analysis",
    "customer segmentation analysis",
    "product performance review",
    "monthly executive dashboard",
    "marketing performance summary",
    "operations trend analysis",
    "inventory and unit sales review",
    "financial KPI exploration",
]

REQUEST_STYLES = [
    "Create",
    "Build",
    "Show",
    "Plot",
    "Visualize",
    "Generate",
]

SCHEMA_HEADERS = [
    "Available columns:",
    "Columns in df:",
    "Dataframe columns:",
]

COLUMN_DTYPES = {
    "sales": "float",
    "profit": "float",
    "revenue": "float",
    "units": "int",
    "customer_age": "int",
    "region": "category",
    "product": "category",
    "segment": "category",
    "year": "int",
    "month": "category",
    "date": "datetime",
}

ALLOWED_SCHEMA_HEADERS = tuple(SCHEMA_HEADERS)


@dataclass
class MineConfig:
    openai_model: str = DEFAULT_OPENAI_MODEL
    max_output_tokens: int = 1800
    openai_timeout_s: int = 180
    sleep_s: float = 0.2
    seed: int = 42
    append: bool = True


def openai_headers() -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def extract_response_text(resp_json: dict[str, Any]) -> str:
    if isinstance(resp_json.get("output_text"), str):
        return resp_json["output_text"]

    parts: list[str] = []
    for item in resp_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))

    return "\n".join(parts).strip()


def normalize_code(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```python\s*", "", code)
    code = re.sub(r"^```\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    return code.strip()


def normalize_user_content(text: str) -> str:
    text = text.strip()
    if not text.startswith("User request:"):
        text = "User request:\n" + text
    return text


def schema_lines_from_columns(columns: list[str]) -> list[str]:
    unique_cols: list[str] = []
    for col in columns:
        if col in COLUMN_DTYPES and col not in unique_cols:
            unique_cols.append(col)

    if len(unique_cols) < 4:
        fallback = ["sales", "profit", "segment", "region", "month", "date", "product", "revenue"]
        for col in fallback:
            if col not in unique_cols:
                unique_cols.append(col)
            if len(unique_cols) >= 5:
                break

    return [f"{col}: {COLUMN_DTYPES[col]}" for col in unique_cols[:8]]


def make_prompt(query: dict[str, Any]) -> str:
    chart = query["chart"]
    goal = query["goal"]
    context = query["context"]
    columns = query["columns"]
    style = query["style"]
    header = query["schema_header"]

    cols_text = ", ".join(columns)
    schema_block = "\n".join(schema_lines_from_columns(columns))

    return f"""
Search the web for a realistic business analytics example using Plotly Express.

Look for examples from:
- dashboards
- tutorials
- blog posts
- analytics guides
- business data analysis examples

Target chart family: {chart}
Analytic goal: {goal}
Business context: {context}

Prefer examples involving business metrics such as:
sales, profit, revenue, units, customer_age, region, product, segment, date, year, month.

Avoid overly simple examples like "tips dataset" or generic toy datasets if possible.

Use the example to create ONE training example.
Preferred columns to anchor the example: {cols_text}
Prefer examples that demonstrate grouping, faceting, or aggregation.

Convert it into ONE training example.

Return EXACT JSON in this format:

{{
  "messages": [
    {{
      "role": "system",
      "content": "{SYSTEM_PROMPT}"
    }},
    {{
      "role": "user",
      "content": "User request:\\n{style} a {chart} chart for {context}\\n\\n{header}\\n{schema_block}"
    }},
    {{
      "role": "assistant",
      "content": "import streamlit as st\\nimport plotly.express as px\\n\\nst.title(\\"Example\\")\\nfig = px.scatter(df, x=\\"sales\\", y=\\"profit\\", color=\\"segment\\")\\nst.plotly_chart(fig, use_container_width=True)"
    }}
  ]
}}

Hard requirements:
1. messages must contain exactly 3 items: system, user, assistant.
2. system content must be exactly:
   {SYSTEM_PROMPT}
3. user content must:
   - start with exactly: User request:
   - include a natural language graph request
   - then a blank line
   - then a schema block starting with exactly one of:
     - Available columns:
     - Columns in df:
     - Dataframe columns:
4. The schema must contain 2 to 11 realistic columns.
5. Use realistic columns consistent with the chosen chart family.
6. Prefer columns from this set when natural:
   sales, profit, revenue, units, customer_age, region, product, segment, year, month, date
7. Do NOT use placeholder names like:
   x_column, y_column, color_column, size_column, hover_column
8. assistant content must be code only.
9. assistant code must:
   - import streamlit as st
   - import plotly.express as px
   - assume df already exists
   - use only plotly.express
   - reference only columns that appear in the schema
   - match the requested chart family
   - end with st.plotly_chart(fig, use_container_width=True) or st.plotly_chart(fig)
10. No markdown fences.
11. No explanation text outside the JSON.
12. IMPORTANT: avoid repeating simple examples like "sales by region" unless the chart truly requires it.
13. IMPORTANT: vary grouping, color fields, faceting, and aggregation patterns.
14. IMPORTANT: vary titles, groupings, x/y choices, color fields, filters, and business framing.
15. IMPORTANT: prioritize diversity over simplicity if the example remains clean and valid.
""".strip()


def call_openai_web_miner(query: dict[str, Any], cfg: MineConfig) -> dict[str, Any]:
    payload = {
        "model": cfg.openai_model,
        "input": make_prompt(query),
        "max_output_tokens": cfg.max_output_tokens,
        "tools": [{"type": "web_search"}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "plotly_training_example",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "messages": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                        }
                    },
                    "required": ["messages"],
                },
            }
        },
    }

    resp = requests.post(
        OPENAI_RESPONSES_URL,
        headers=openai_headers(),
        json=payload,
        timeout=cfg.openai_timeout_s,
    )

    print("\nSTATUS:", resp.status_code)

    if not resp.ok:
        print("RAW RESPONSE:")
        print(resp.text[:4000])
        resp.raise_for_status()

    try:
        resp_json = resp.json()
    except Exception:
        print("NON JSON RESPONSE:")
        print(resp.text[:4000])
        raise

    raw_text = extract_response_text(resp_json)

    print("RAW_TEXT PREVIEW:")
    print(repr(raw_text[:500]))

    if not raw_text:
        print("FULL RESPONSE:")
        print(json.dumps(resp_json, indent=2)[:4000])
        raise RuntimeError("No output text returned")

    try:
        parsed = json.loads(raw_text)
    except Exception:
        print("FAILED JSON PARSE:")
        print(raw_text[:2000])
        raise

    return parsed


def extract_schema_columns(user_content: str) -> set[str]:
    lines = user_content.splitlines()
    start_idx = None

    for i, line in enumerate(lines):
        if line.strip() in ALLOWED_SCHEMA_HEADERS:
            start_idx = i + 1
            break

    if start_idx is None:
        return set()

    cols: set[str] = set()
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        name = line.split(":", 1)[0].strip()
        if name:
            cols.add(name)

    return cols


def validate_record(record: dict[str, Any]) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("messages must contain exactly 3 items")

    roles = [m.get("role") for m in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"invalid role sequence: {roles}")

    system_content = messages[0].get("content", "")
    user_content = messages[1].get("content", "")
    assistant_content = messages[2].get("content", "")

    if system_content != SYSTEM_PROMPT:
        raise ValueError("system prompt mismatch")

    if not user_content.startswith("User request:"):
        raise ValueError("user content must start with 'User request:'")

    if "\n\n" not in user_content:
        raise ValueError("user content must include a blank line before schema block")

    if not any(header in user_content for header in ALLOWED_SCHEMA_HEADERS):
        raise ValueError("schema header missing or invalid")

    schema_cols = extract_schema_columns(user_content)
    if not (2 <= len(schema_cols) <= 11):
        raise ValueError(f"schema must contain 2 to 11 columns, got {len(schema_cols)}")

    if not assistant_content.strip():
        raise ValueError("assistant content is empty")

    if "```" in assistant_content:
        raise ValueError("assistant content must not include markdown fences")

    if "import streamlit as st" not in assistant_content:
        raise ValueError("assistant code must import streamlit as st")

    if "import plotly.express as px" not in assistant_content:
        raise ValueError("assistant code must import plotly.express as px")

    if "st.plotly_chart(" not in assistant_content:
        raise ValueError("assistant code must call st.plotly_chart")

    banned_placeholders = [
        "x_column",
        "y_column",
        "color_column",
        "size_column",
        "hover_column",
    ]
    for token in banned_placeholders:
        if token in user_content or token in assistant_content:
            raise ValueError(f"placeholder column found: {token}")

    referenced = set(re.findall(r'["\']([A-Za-z_][A-Za-z0-9_]*)["\']', assistant_content))
    referenced_cols = {tok for tok in referenced if tok in COLUMN_DTYPES}
    unknown = referenced_cols - schema_cols
    if unknown:
        raise ValueError(f"assistant references columns not in schema: {sorted(unknown)}")


def write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_queries(n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    i = 0

    while len(out) < n:
        spec = CHART_SPECS[i % len(CHART_SPECS)]
        out.append({
            "chart": spec["chart"],
            "goal": spec["goal"],
            "columns": spec["columns"],
            "context": BUSINESS_CONTEXTS[i % len(BUSINESS_CONTEXTS)],
            "style": REQUEST_STYLES[i % len(REQUEST_STYLES)],
            "schema_header": SCHEMA_HEADERS[i % len(SCHEMA_HEADERS)],
        })
        i += 1

    random.shuffle(out)
    return out


def run_sync(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = DATA_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = MineConfig(
        openai_model=args.model,
        max_output_tokens=args.max_output_tokens,
        openai_timeout_s=args.timeout,
        sleep_s=args.sleep_s,
        seed=args.seed,
        append=not args.overwrite,
    )

    random.seed(cfg.seed)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    queries = generate_queries(args.n)
    written = 0
    failed = 0

    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] mining: {q['chart']} | {q['context']}")

        try:
            record = call_openai_web_miner(q, cfg)

            record["messages"][1]["content"] = normalize_user_content(
                record["messages"][1]["content"]
            )
            record["messages"][2]["content"] = normalize_code(
                record["messages"][2]["content"]
            )

            validate_record(record)
            write_record(output_path, record)
            written += 1

        except Exception as e:
            failed += 1
            print("error:", e)

        time.sleep(cfg.sleep_s)

    print(f"Saved {written} rows to {output_path}")
    print(f"Failed rows: {failed}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("mine-sync")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--output", default="plotly_web_mined.jsonl")
    s.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    s.add_argument("--max-output-tokens", type=int, default=1800)
    s.add_argument("--timeout", type=int, default=180)
    s.add_argument("--sleep-s", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=run_sync)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
