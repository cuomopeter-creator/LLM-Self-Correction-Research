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


PREFERRED_INPUT = BASE_DIR / "prompt_visual_inspection_with_specs.json"
FALLBACK_INPUT = BASE_DIR / "prompt_visual_inspection_rewritten.json"
OUTPUT_PATH = BASE_DIR / "prompt_visual_inspection_with_data.json"

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
            max_output_tokens=model_cfg.get("max_output_tokens", 20000),
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return json.dumps({"df": parsed}, ensure_ascii=False)
        return text
    except Exception:
        pass

    object_match = re.search(r"(\{\s*.*\s*\})", text, flags=re.DOTALL)
    if object_match:
        candidate = object_match.group(1).strip()
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return json.dumps({"df": parsed}, ensure_ascii=False)
        return candidate

    list_match = re.search(r"(\[\s*.*\s*\])", text, flags=re.DOTALL)
    if list_match:
        candidate = list_match.group(1).strip()
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return json.dumps({"df": parsed}, ensure_ascii=False)

    raise ValueError("Could not find JSON payload in model response.")


def build_df_prompt(item: dict[str, Any]) -> str:
    prompt = str(item.get("prompt", "")).strip()
    category = str(item.get("category", "")).strip()
    difficulty = str(item.get("difficulty", "")).strip()
    chart_family = str(item.get("chart_family", "")).strip()
    spec = item.get("df_spec") or {}

    return f"""
You are generating a synthetic dataframe for visualization testing.

Return ONLY valid JSON.
Return exactly one JSON object.
Do not include markdown.
Do not include explanations.
Do not include code fences.
Do not include any top-level keys other than "df".

Your task:
Generate a JSON object with exactly this shape:
{{
  "df": [
    {{"col_a": "...", "col_b": 123}}
  ]
}}

Requirements:
- "df" must be a non-empty array of row objects
- Try to honor row_count, but structural usefulness matters more than exact count
- Every row object must contain exactly the columns listed in the spec
- Do not add extra columns
- Do not omit required columns
- Use realistic synthetic values
- Avoid nulls unless absolutely necessary
- Make the dataset directly usable for plotting without further cleaning
- Keep row values varied enough for meaningful charts
- Prefer stable, internally consistent data over random noise

Type targets:
- "float" -> prefer JSON numbers
- "int" -> prefer JSON integers
- "category" -> prefer JSON strings
- "datetime" -> prefer strings in YYYY-MM-DD format
- "bool" -> prefer true or false

Spec fidelity goals:
- Respect date_range when provided
- Respect categorical_values when provided
- Respect numeric_rules including:
  - min
  - max
  - trend
  - group_effects
  - noise
  - correlated_with
  - derived_from
- Derived columns should be numerically consistent with their source columns
- Boolean flags should be logically consistent with the data
- If the spec implies aggregation support, generate rows that aggregate cleanly
- If the spec implies time-series behavior, make the data support that clearly
- If the spec implies per-group temporal coverage, include the necessary combinations
- If the spec implies anomalies, outliers, promos, max/min markers, sharp drops, top performers, confidence bands, or cumulative behavior, make those patterns visible in the data
- If notes and numeric_rules are in tension, prioritize notes and required_features

Granularity rules:
- Prefer granular rows over pre-aggregated rows when possible
- For distribution/comparison prompts, include enough repeated observations per group to make the chart meaningful
- For time-series prompts, include enough date coverage and group coverage to make the requested chart meaningful

Output quality rules:
- Keep IDs unique when an id-like column exists
- Ensure there are no duplicate rows unless the spec truly implies them

Visualization prompt:
{json.dumps(prompt)}

Metadata:
- category: {json.dumps(category)}
- difficulty: {json.dumps(difficulty)}
- chart_family: {json.dumps(chart_family)}

Dataframe spec:
{json.dumps(spec, indent=2, ensure_ascii=False)}
""".strip()


def validate_df_payload(payload: dict[str, Any], item: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Model response was not a JSON object")

    if set(payload.keys()) != {"df"}:
        raise ValueError('Top-level JSON must contain exactly one key: "df"')

    df = payload["df"]
    if not isinstance(df, list) or not df:
        raise ValueError('"df" must be a non-empty list')

    spec = item.get("df_spec") or {}
    columns = spec.get("columns") or {}
    expected_cols = list(columns.keys())
    expected_set = set(expected_cols)

    if not expected_cols:
        raise ValueError("df_spec.columns is missing or empty")

    for i, row in enumerate(df, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not an object")

        row_keys = set(row.keys())
        if row_keys != expected_set:
            missing = sorted(expected_set - row_keys)
            extra = sorted(row_keys - expected_set)
            raise ValueError(
                f"Row {i} columns mismatch. Missing={missing or []} Extra={extra or []}"
            )

    return df


def generate_df_for_item(
    model: OpenAIModel,
    item: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    prompt = build_df_prompt(item)
    result = model.generate(prompt)
    text = str(result.get("text", "")).strip()
    usage = result.get("usage", {}) or {}

    json_text = extract_json_block(text)
    payload = json.loads(json_text)
    df = validate_df_payload(payload, item)
    return df, usage, text


def save_progress(
    data: dict[str, Any],
    items: list[dict[str, Any]],
    usage_history: list[dict[str, Any]],
) -> None:
    data["items"] = items
    data["df_generated_at_utc"] = time.time()
    data["df_source_model"] = MODEL_KEY
    data["df_generation_usage"] = usage_history
    save_json(OUTPUT_PATH, data)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    input_path = choose_input_path()
    data = load_json(input_path)

    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Input JSON does not contain a non-empty 'items' list.")

    model = build_gpt_model()

    if OUTPUT_PATH.exists():
        existing = load_json(OUTPUT_PATH)
        existing_items = existing.get("items", [])
        if isinstance(existing_items, list) and len(existing_items) == len(items):
            items = existing_items
            data = existing

    usage_history = data.get("df_generation_usage", [])
    if not isinstance(usage_history, list):
        usage_history = []

    updated = 0

    for i, item in enumerate(items, start=1):
        item_id = item.get("id")

        if item.get("df"):
            print(f"[{i}/{len(items)}] skip id={item_id} (already has df)")
            continue

        if not item.get("df_spec"):
            error_msg = f"id={item_id} missing df_spec"
            print(f"[{i}/{len(items)}] FAILED {error_msg}")
            item["df_error"] = error_msg
            item["df_raw_response"] = ""
            item["df_attempts"] = 0
            usage_history.append(
                {
                    "id": item_id,
                    "status": "failed",
                    "attempt": 0,
                    "error": error_msg,
                }
            )
            save_progress(data, items, usage_history)
            continue

        print(f"[{i}/{len(items)}] generating df for id={item_id}")

        success = False
        last_error = ""
        last_text = ""

        for attempt in range(1, MAX_RETRIES + 1):
            raw_text = ""
            try:
                df, usage, raw_text = generate_df_for_item(model, item)
                item["df"] = df
                item.pop("df_error", None)
                item.pop("df_raw_response", None)
                item["df_attempts"] = attempt

                usage_history.append(
                    {
                        "id": item_id,
                        "attempt": attempt,
                        "status": "success",
                        "usage": usage,
                    }
                )

                updated += 1
                success = True
                break

            except Exception as e:
                last_error = str(e)
                last_text = raw_text
                print(f"  attempt {attempt} failed: {last_error}")

        if not success:
            print(f"[{i}/{len(items)}] FAILED id={item_id}: {last_error}")
            item["df_error"] = last_error
            item["df_raw_response"] = last_text
            item["df_attempts"] = MAX_RETRIES

            usage_history.append(
                {
                    "id": item_id,
                    "attempt": MAX_RETRIES,
                    "status": "failed",
                    "error": last_error,
                }
            )

        if updated % SAVE_EVERY == 0 or not success:
            save_progress(data, items, usage_history)

    save_progress(data, items, usage_history)
    print(f"Saved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
