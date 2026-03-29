from __future__ import annotations
from datetime import datetime

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from chat_models import build_model, generate_with_model, load_yaml  # noqa: E402

DATA_PATH = PROJECT_ROOT / "graph_codegen" / "experiments" / "prompt_visual_inspection_audited.json"

SYSTEM_PROMPT = (
    "Generate a Streamlit app snippet using plotly.express or plotly.graph_objects only. "
    "A dataframe named df is already loaded and contains the real data for this prompt. "
    "Use only columns that actually exist in df. "
    "Do not create sample data. "
    "Do not redefine df. "
    "Do not read from files. "
    "Return code only."
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_model_text(text: str) -> str:
    text = (text or "").strip()

    if "```python" in text:
        text = text.split("```python", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        return text.strip()

    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()

    return text.strip()


def make_df_sample(df: list[dict], n: int = 5) -> str:
    if not isinstance(df, list) or not df:
        return "[]"
    return json.dumps(df[:n], indent=2, ensure_ascii=False)


def build_prompt(item: dict) -> str:
    prompt = str(item.get("prompt", "")).strip()
    category = str(item.get("category", "")).strip()
    difficulty = str(item.get("difficulty", "")).strip()
    chart_family = str(item.get("chart_family", "")).strip()

    df_spec = item.get("df_spec", {}) or {}
    columns = df_spec.get("columns", {}) or {}
    notes = str(df_spec.get("notes", "")).strip()
    required_features = df_spec.get("required_features", []) or []
    df = item.get("df", []) or []

    column_lines = "\n".join(f"- {col}: {dtype}" for col, dtype in columns.items())
    feature_lines = "\n".join(f"- {x}" for x in required_features) if required_features else "- none"

    return f"""User request:
{prompt}

Metadata:
- category: {category}
- difficulty: {difficulty}
- chart_family: {chart_family}

Available dataframe columns:
{column_lines}

Dataframe notes:
{notes or "none"}

Required features:
{feature_lines}

First 5 rows of df:
{make_df_sample(df, n=5)}

The dataframe is already loaded as df.
Write a Streamlit + Plotly snippet that answers the request using this exact df.
Use only columns shown above.
Return code only."""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_yaml(PROJECT_ROOT / "configs" / "models.yaml")
    model_cfg = cfg["models"][args.model]
    model = build_model(model_cfg)

    data = load_json(DATA_PATH)
    items = data.get("items", [])

    generated = 0
    skipped = 0
    failed = 0

    for item in items:
        item_id = item.get("id")

        if not args.overwrite and item.get("response"):
            print(f"[skip] id={item_id}")
            skipped += 1
            continue

        print(f"[start] id={item_id}")

        try:
            model_prompt = build_prompt(item)
            full_prompt = f"{SYSTEM_PROMPT}\n\n{model_prompt}"

            result = generate_with_model(model, model_cfg, full_prompt)
            text = str(result.get("text", "")).strip()
            code = extract_model_text(text)
            usage = result.get("usage", {}) or {}

            item["response"] = code
            item["exec_pass"] = None
            item["status"] = ""
            if item.get("notes") is None:
                item["notes"] = ""

            print(f"[success] id={item_id}")
            print(code[:1000])
            if usage:
                print(f"[usage] {json.dumps(usage)}")

            generated += 1

        except Exception as e:
            item["response"] = ""
            item["exec_pass"] = False
            item["status"] = "fail"
            item["notes"] = f"generation error: {e}"

            print(f"[fail] id={item_id}")
            print(str(e))

            failed += 1

    run_dir = PROJECT_ROOT / "graph_codegen" / "experiments" / "runs"
    run_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = run_dir / f"{args.model}_{ts}_audited_responses.json"

    data["items"] = items
    save_json(out_path, data)

    print(f"[summary] generated={generated} skipped={skipped} failed={failed}")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
