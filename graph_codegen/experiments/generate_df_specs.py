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
BASE_DIR = HERE.parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.openai_model import OpenAIModel, OpenAIModelConfig  # noqa: E402


PREFERRED_INPUT = BASE_DIR / "prompt_visual_inspection_rewritten.json"
FALLBACK_INPUT = BASE_DIR / "prompt_visual_inspection.json"
OUTPUT_PATH = BASE_DIR / "prompt_visual_inspection_with_specs.json"

MODEL_KEY = "gpt"
MAX_RETRIES = 3
SAVE_EVERY = 1


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


def choose_input_path() -> Path:
    if PREFERRED_INPUT.exists():
        return PREFERRED_INPUT
    if FALLBACK_INPUT.exists():
        return FALLBACK_INPUT
    raise FileNotFoundError(
        f"Could not find input file. Checked:\n- {PREFERRED_INPUT}\n- {FALLBACK_INPUT}"
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

    match = re.search(r"(\{\s*.*\s*\})", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        json.loads(candidate)
        return candidate

    raise ValueError("Could not find JSON object in model response.")


def build_spec_prompt(item: dict[str, Any]) -> str:
    prompt = str(item.get("prompt", "")).strip()
    category = str(item.get("category", "")).strip()
    difficulty = str(item.get("difficulty", "")).strip()
    chart_family = str(item.get("chart_family", "")).strip()

    return f"""
You are designing a compact dataframe specification for a synthetic dataset that will be used to test chart-generation code.

Return ONLY valid JSON.
Return exactly one JSON object.
Do not include markdown.
Do not include explanations.

The spec should make it easy for Python to later synthesize a dataframe that matches this visualization request.

Prompt metadata:
- prompt: {json.dumps(prompt)}
- category: {json.dumps(category)}
- difficulty: {json.dumps(difficulty)}
- chart_family: {json.dumps(chart_family)}

Requirements:
- Infer the minimum useful dataset needed for this prompt
- Include enough columns to support the visualization request fairly
- Keep the schema compact and realistic
- Prefer common analytics columns and plausible business data
- If the prompt implies advanced features like annotations, confidence intervals, error bars, highlighted events, cumulative metrics, or dual axes, include the columns needed to support them
- Do NOT generate raw rows
- Do NOT generate python code

Return a JSON object with exactly these top-level keys:

"columns"
"row_count"
"categorical_values"
"date_range"
"numeric_rules"
"required_features"
"notes"

Definitions:

1. "columns"
A JSON object mapping column name to dtype.
Allowed dtypes:
"float", "int", "category", "datetime", "bool"

2. "row_count"
An integer between 120 and 600.

3. "categorical_values"
A JSON object mapping categorical column names to a short list of example values.

4. "date_range"
Either null or an object with:
"column", "start", "end", "freq"
Use freq values like:
"day", "week", "month", "quarter"

5. "numeric_rules"
A JSON object where each numeric column maps to an object describing how to generate it.
Each numeric rule should include:
"min"
"max"
and may optionally include:
"trend"
"group_effects"
"noise"
"correlated_with"
"derived_from"

6. "required_features"
A short JSON array of strings describing important structural needs such as:
- "supports_group_comparison"
- "supports_time_series"
- "supports_error_bars"
- "supports_annotations"
- "supports_dual_axis"
- "supports_cumulative_metric"
- "supports_heatmap_pivot"
- "supports_distribution_overlay"
- "supports_subplots"
- "supports_outlier_detection"
- "supports_confidence_band"

7. "notes"
A short string explaining any special generation considerations.

Example shape only:
{{
  "columns": {{
    "date": "datetime",
    "region": "category",
    "revenue": "float",
    "profit": "float"
  }},
  "row_count": 240,
  "categorical_values": {{
    "region": ["North", "South", "East", "West"]
  }},
  "date_range": {{
    "column": "date",
    "start": "2023-01-01",
    "end": "2024-12-01",
    "freq": "month"
  }},
  "numeric_rules": {{
    "revenue": {{
      "min": 1000,
      "max": 20000,
      "trend": "upward_over_time",
      "group_effects": "North and West slightly higher",
      "noise": "moderate"
    }},
    "profit": {{
      "min": 50,
      "max": 6000,
      "correlated_with": "revenue",
      "noise": "moderate"
    }}
  }},
  "required_features": ["supports_time_series", "supports_group_comparison"],
  "notes": "Make sure each month has data for each region."
}}
""".strip()


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    required = [
        "columns",
        "row_count",
        "categorical_values",
        "date_range",
        "numeric_rules",
        "required_features",
        "notes",
    ]
    for key in required:
        if key not in spec:
            raise ValueError(f"Missing key in spec: {key}")

    if not isinstance(spec["columns"], dict) or not spec["columns"]:
        raise ValueError("spec['columns'] must be a non-empty object")

    allowed_dtypes = {"float", "int", "category", "datetime", "bool"}
    for col, dtype in spec["columns"].items():
        if dtype not in allowed_dtypes:
            raise ValueError(f"Unsupported dtype for {col}: {dtype}")

    row_count = spec["row_count"]
    if not isinstance(row_count, int) or row_count < 120 or row_count > 600:
        raise ValueError("row_count must be an int between 120 and 600")

    if not isinstance(spec["categorical_values"], dict):
        raise ValueError("categorical_values must be an object")

    for col, vals in spec["categorical_values"].items():
        if not isinstance(vals, list):
            raise ValueError(f"categorical_values[{col}] must be a list")

    date_range = spec["date_range"]
    if date_range is not None:
        if not isinstance(date_range, dict):
            raise ValueError("date_range must be null or an object")
        for key in ["column", "start", "end", "freq"]:
            if key not in date_range:
                raise ValueError(f"date_range missing key: {key}")

    if not isinstance(spec["numeric_rules"], dict):
        raise ValueError("numeric_rules must be an object")

    for col, rule in spec["numeric_rules"].items():
        if not isinstance(rule, dict):
            raise ValueError(f"numeric rule for {col} must be an object")
        if "min" not in rule or "max" not in rule:
            raise ValueError(f"numeric rule for {col} must include min and max")

    if not isinstance(spec["required_features"], list):
        raise ValueError("required_features must be a list")

    if not isinstance(spec["notes"], str):
        raise ValueError("notes must be a string")

    return spec


def generate_spec_for_item(model: OpenAIModel, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = build_spec_prompt(item)
    result = model.generate(prompt)
    text = str(result.get("text", "")).strip()
    usage = result.get("usage", {}) or {}
    json_text = extract_json_block(text)
    spec = json.loads(json_text)
    if not isinstance(spec, dict):
        raise ValueError("Model response was not a JSON object.")
    spec = validate_spec(spec)
    return spec, usage


def save_output(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    input_path = choose_input_path()
    data = json.loads(input_path.read_text(encoding="utf-8"))

    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Input JSON does not contain a non-empty 'items' list.")

    model = build_gpt_model()

    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        existing_items = existing.get("items", [])
        if isinstance(existing_items, list) and len(existing_items) == len(items):
            items = existing_items
            data = existing

    usage_history = data.get("df_spec_usage", [])
    if not isinstance(usage_history, list):
        usage_history = []

    updated = 0

    for i, item in enumerate(items, start=1):
        if item.get("df_spec"):
            print(f"[{i}/{len(items)}] skip id={item.get('id')} (already has df_spec)")
            continue

        print(f"[{i}/{len(items)}] generating df_spec for id={item.get('id')}")

        success = False
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                spec, usage = generate_spec_for_item(model, item)
                item["df_spec"] = spec
                usage_history.append(
                    {
                        "id": item.get("id"),
                        "attempt": attempt,
                        "usage": usage,
                    }
                )
                updated += 1
                success = True
                break
            except Exception as e:
                last_error = e
                print(f"  attempt {attempt} failed: {e}")

        if not success:
            raise RuntimeError(f"Failed on id={item.get('id')}: {last_error}")

        if updated % SAVE_EVERY == 0:
            data["items"] = items
            data["df_spec_generated_at_utc"] = time.time()
            data["df_spec_source_model"] = MODEL_KEY
            data["df_spec_usage"] = usage_history
            save_output(OUTPUT_PATH, data)

    data["items"] = items
    data["df_spec_generated_at_utc"] = time.time()
    data["df_spec_source_model"] = MODEL_KEY
    data["df_spec_usage"] = usage_history
    save_output(OUTPUT_PATH, data)

    print(f"Saved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
