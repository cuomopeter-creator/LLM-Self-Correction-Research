from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.openai_model import OpenAIModel, OpenAIModelConfig  # noqa: E402


OUTPUT_PATH = HERE.parent / "prompt_visual_inspection.json"
MODEL_KEY = "gpt"
TARGET_N = 100
MAX_ATTEMPTS = 8


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_gpt_model() -> OpenAIModel:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "models.yaml")
    model_cfg = cfg["models"][MODEL_KEY]

    return OpenAIModel(
        OpenAIModelConfig(
            model=model_cfg["model"],
            api_key_env=model_cfg.get("api_key_env", "OPENAI_API_KEY"),
            max_output_tokens=model_cfg.get("max_output_tokens", 5000),
            temperature=model_cfg.get("temperature", 0.0),
        )
    )


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def extract_json_block(text: str) -> str:
    text = strip_fences(text)

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    match = re.search(r"(\[\s*.*\s*\])", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        json.loads(candidate)
        return candidate

    raise ValueError("Could not find JSON array in response.")


def normalize_item(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None

    prompt = str(item.get("prompt", "")).strip()
    category = str(item.get("category", "")).strip()
    difficulty = str(item.get("difficulty", "")).strip()
    chart_family = str(item.get("chart_family", "")).strip().lower()

    if not prompt:
        return None

    if chart_family not in {"plotly_express", "graph_objects"}:
        return None

    return {
        "prompt": prompt,
        "category": category,
        "difficulty": difficulty,
        "chart_family": chart_family,
    }


def dedupe_key(prompt: str, chart_family: str) -> str:
    p = re.sub(r"\s+", " ", prompt.lower().strip())
    return f"{chart_family}:{p}"


def build_row(idx: int, item: dict[str, str]) -> dict[str, Any]:
    return {
        "id": idx,
        "prompt": item["prompt"],
        "category": item["category"],
        "difficulty": item["difficulty"],
        "chart_family": item["chart_family"],
        "df": [],
        "response": "",
        "correctness": None,
        "code_quality": None,
        "chart_quality": None,
        "notes": "",
    }


def build_generation_prompt() -> str:
    return f"""
Generate exactly 100 realistic natural-language prompts for evaluating a model that writes Plotly + Streamlit chart code.

DATASET REQUIREMENTS

Return exactly 100 objects.

Exactly:
50 must have chart_family = "plotly_express"
50 must have chart_family = "graph_objects"

Return ONLY valid JSON.
Return a JSON array with exactly 100 objects.

Each object must contain:

"prompt"
"category"
"difficulty"
"chart_family"

Difficulty must be:
easy
medium
hard

PROMPT STYLE REQUIREMENTS

Each prompt must read like a real analytics request from a user.

Every prompt should clearly imply:
- the chart type
- the variables being compared
- the analytical goal

Avoid generic prompts like:
"Create a bar chart"
"Make a scatter plot"

Instead write prompts like:

"Show total sales by region as a bar chart sorted from highest to lowest so we can quickly identify the strongest markets."

DATA CONTEXT

Prompts may reference common analytical fields such as:

sales
revenue
profit
region
country
category
segment
customer_type
year
month
date
temperature
price
inventory
orders
conversion_rate
ad_spend
traffic
rating
delivery_time

Do NOT define a dataframe schema.

CHART VARIETY

Across the prompts include a mix of:

bar charts
line charts
scatter plots
histograms
box plots
heatmaps
faceted charts
rankings
time series
distribution analysis
correlation analysis
dashboards

GRAPH OBJECTS PROMPTS

Prompts using graph_objects should often involve:

subplots
dual y-axes
annotations
highlighted ranges
multi-trace overlays
stacked traces
dashboards
custom layout
error bars

Example object:

{{
"prompt": "Plot monthly revenue as a line chart and highlight the months where revenue dropped compared to the previous month.",
"category": "time_series",
"difficulty": "medium",
"chart_family": "plotly_express"
}}
""".strip()


def call_model(model: OpenAIModel, prompt: str) -> tuple[list[Any], dict]:
    result = model.generate(prompt)
    text = result.get("text", "")
    usage = result.get("usage", {})
    json_text = extract_json_block(text)
    items = json.loads(json_text)
    return items, usage


def main():
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    model = build_gpt_model()

    px = []
    go = []
    seen = set()
    usage_log = []

    for attempt in range(1, MAX_ATTEMPTS + 1):

        need_px = 50 - len(px)
        need_go = 50 - len(go)

        if need_px <= 0 and need_go <= 0:
            break

        print(f"Attempt {attempt} | need px={need_px} go={need_go}")

        prompt = build_generation_prompt()

        try:
            items, usage = call_model(model, prompt)
            usage_log.append(usage)

            for item in items:

                norm = normalize_item(item)
                if norm is None:
                    continue

                key = dedupe_key(norm["prompt"], norm["chart_family"])
                if key in seen:
                    continue

                if norm["chart_family"] == "plotly_express":
                    if len(px) < 50:
                        px.append(norm)
                        seen.add(key)

                else:
                    if len(go) < 50:
                        go.append(norm)
                        seen.add(key)

        except Exception as e:
            print("Attempt failed:", e)

    if len(px) != 50 or len(go) != 50:
        raise ValueError(
            f"Failed to collect prompts. px={len(px)} go={len(go)}"
        )

    rows = []
    idx = 1

    for item in px:
        rows.append(build_row(idx, item))
        idx += 1

    for item in go:
        rows.append(build_row(idx, item))
        idx += 1

    payload = {
        "created_at_utc": time.time(),
        "n_prompts": len(rows),
        "plotly_express": len(px),
        "graph_objects": len(go),
        "usage": usage_log,
        "items": rows,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("Saved:", OUTPUT_PATH)
    print("PX:", len(px), "GO:", len(go))


if __name__ == "__main__":
    main()
