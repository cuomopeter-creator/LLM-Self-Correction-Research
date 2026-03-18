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


INPUT_PATH = BASE_DIR / "prompt_visual_inspection_with_data.json"
OUTPUT_PATH = BASE_DIR / "prompt_visual_inspection_audited.json"

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def extract_json_array(text: str) -> str:
    text = strip_fences(text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return text
    except Exception:
        pass

    match = re.search(r"(\[\s*.*\s*\])", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return candidate

    raise ValueError("Could not find JSON array in model response.")


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return [str(value).strip()]


def get_soft_reviews(item: dict[str, Any]) -> list[str]:
    out: list[str] = []

    review_keys = [
        "review_notes",
        "reviews",
        "revision_notes",
        "revisions",
        "soft_errors",
        "warnings",
        "advisory_notes",
    ]
    for key in review_keys:
        out.extend(normalize_list(item.get(key)))

    if not out:
        out = [
            "Revise the dataframe with the soft principles in mind.",
            "Row count is advisory, not strict.",
            "Preferred dtype cleanliness is advisory, not strict.",
            "Preferred category alignment is advisory, not strict.",
            "Preferred date-range alignment is advisory, not strict.",
            "Preferred numeric min/max plausibility is advisory, not strict.",
            "Keep the dataframe practical and chart-friendly.",
        ]

    return out


def get_hard_errors(item: dict[str, Any]) -> list[str]:
    out = normalize_list(item.get("errors"))

    df_error = str(item.get("df_error", "")).strip()
    if df_error:
        out.append(f"Previous generation failed with error: {df_error}")

    return out


def has_generation_failure(item: dict[str, Any]) -> bool:
    return bool(str(item.get("df_error", "")).strip())


def build_revision_prompt(item: dict[str, Any]) -> str:
    prompt = str(item.get("prompt", "")).strip()
    category = str(item.get("category", "")).strip()
    difficulty = str(item.get("difficulty", "")).strip()
    chart_family = str(item.get("chart_family", "")).strip()
    df_spec = item.get("df_spec") or {}
    df = item.get("df") or []
    soft_reviews = get_soft_reviews(item)
    hard_errors = get_hard_errors(item)

    return f"""
You are revising a synthetic dataframe for visualization testing.

Return ONLY a JSON array of row objects.
Do not include markdown.
Do not include explanations.
Do not include code fences.
Do not include any wrapper object.

There are two instruction levels:

1. HARD ISSUES
These must be addressed firmly.

2. SOFT REVISION PRINCIPLES
These should always be applied thoughtfully, but they are advisory rather than strict pass/fail requirements.

Your job for this item:
- Revise the current dataframe
- Keep the existing schema exactly
- Do not add columns
- Do not remove columns
- Improve the dataframe according to the soft revision principles
- Address all hard issues if any exist
- Keep the result realistic and chart-friendly
- Return only the revised JSON array

Visualization prompt:
{json.dumps(prompt)}

Metadata:
- category: {json.dumps(category)}
- difficulty: {json.dumps(difficulty)}
- chart_family: {json.dumps(chart_family)}

Dataframe spec:
{json.dumps(df_spec, indent=2, ensure_ascii=False)}

Hard issues:
{json.dumps(hard_errors, indent=2, ensure_ascii=False)}

Soft revision principles:
{json.dumps(soft_reviews, indent=2, ensure_ascii=False)}

Important soft principles to consider:
- approximate row count usefulness
- preferred dtype cleanliness
- preferred category alignment
- preferred date-range alignment
- preferred numeric min/max plausibility
- overall usefulness for plotting
- better coverage for groups and dates when that helps the chart intent

Current dataframe:
{json.dumps(df, indent=2, ensure_ascii=False)}
""".strip()


def build_regeneration_prompt(item: dict[str, Any]) -> str:
    prompt = str(item.get("prompt", "")).strip()
    category = str(item.get("category", "")).strip()
    difficulty = str(item.get("difficulty", "")).strip()
    chart_family = str(item.get("chart_family", "")).strip()
    df_spec = item.get("df_spec") or {}
    soft_reviews = get_soft_reviews(item)
    df_error = str(item.get("df_error", "")).strip()
    raw_response = str(item.get("df_raw_response", "")).strip()

    return f"""
You are regenerating a synthetic dataframe for visualization testing because prior generation attempts failed.

Return ONLY a JSON array of row objects.
Do not include markdown.
Do not include explanations.
Do not include code fences.
Do not include any wrapper object.

The previous generation failed. Understand why it failed, avoid repeating that problem, and create a fresh dataframe from the spec.

Previous failure:
{json.dumps(df_error)}

Previous raw response excerpt:
{json.dumps(raw_response[:4000])}

Your job:
- Create a new dataframe from scratch
- Follow the dataframe spec closely
- Make the data practical and chart-friendly
- Apply the soft revision principles as guidance
- Avoid the prior formatting/parsing failure
- Return only a clean JSON array of row objects

Visualization prompt:
{json.dumps(prompt)}

Metadata:
- category: {json.dumps(category)}
- difficulty: {json.dumps(difficulty)}
- chart_family: {json.dumps(chart_family)}

Dataframe spec:
{json.dumps(df_spec, indent=2, ensure_ascii=False)}

Soft revision principles:
{json.dumps(soft_reviews, indent=2, ensure_ascii=False)}

Important:
- The prior failure may have been invalid JSON formatting, truncation, or malformed row structure
- Output valid JSON only
- Every row should use the schema from the spec
- Keep the result realistic and useful for chart generation
""".strip()


def validate_basic_array_schema(repaired_df: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if not isinstance(repaired_df, list) or not repaired_df:
        raise ValueError("Model response was not a non-empty JSON list.")

    spec = item.get("df_spec") or {}
    columns = spec.get("columns") or {}
    expected = set(columns.keys())

    if not expected:
        raise ValueError("df_spec.columns is missing or empty.")

    for i, row in enumerate(repaired_df, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not an object.")
        row_keys = set(row.keys())
        if row_keys != expected:
            missing = sorted(expected - row_keys)
            extra = sorted(row_keys - expected)
            raise ValueError(
                f"Row {i} columns mismatch. Missing={missing or []} Extra={extra or []}"
            )


def generate_repaired_df(
    model: OpenAIModel,
    item: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    if has_generation_failure(item):
        mode = "regenerate"
        prompt = build_regeneration_prompt(item)
    else:
        mode = "revise"
        prompt = build_revision_prompt(item)

    result = model.generate(prompt)
    text = str(result.get("text", "")).strip()
    usage = result.get("usage", {}) or {}

    json_text = extract_json_array(text)
    repaired_df = json.loads(json_text)

    validate_basic_array_schema(repaired_df, item)

    return repaired_df, usage, text, mode


def save_progress(
    data: dict[str, Any],
    items: list[dict[str, Any]],
    usage_history: list[dict[str, Any]],
) -> None:
    data["items"] = items
    data["repair_usage"] = usage_history
    data["repair_timestamp"] = time.time()
    save_json(OUTPUT_PATH, data)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    data = load_json(INPUT_PATH)
    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Input JSON does not contain items.")

    model = build_gpt_model()

    usage_history = data.get("repair_usage", [])
    if not isinstance(usage_history, list):
        usage_history = []

    updated = 0

    for i, item in enumerate(items, start=1):
        item_id = item.get("id")

        if not item.get("df_spec"):
            print(f"[{i}/{len(items)}] skip id={item_id} (missing df_spec)")
            continue

        mode_label = "regenerating" if has_generation_failure(item) else "revising"
        print(f"[{i}/{len(items)}] {mode_label} id={item_id}")

        success = False
        last_error = ""
        last_text = ""
        mode_used = ""

        for attempt in range(1, MAX_RETRIES + 1):
            raw_text = ""
            try:
                repaired_df, usage, raw_text, mode_used = generate_repaired_df(model, item)

                item["df"] = repaired_df
                item["repair_attempts"] = attempt
                item["repair_mode"] = mode_used
                item["review_status"] = "addressed"

                item.pop("repair_error", None)
                item.pop("repair_raw_response", None)

                # hard issue fields are considered addressed after successful rewrite
                item["errors"] = []
                item.pop("df_error", None)
                item.pop("df_raw_response", None)

                usage_history.append(
                    {
                        "id": item_id,
                        "attempt": attempt,
                        "status": "success",
                        "mode": mode_used,
                        "usage": usage,
                    }
                )

                print(f"  success on attempt {attempt} ({mode_used})")
                updated += 1
                success = True
                break

            except Exception as e:
                last_error = str(e)
                last_text = raw_text
                print(f"  attempt {attempt} failed: {last_error}")

        if not success:
            print(f"[{i}/{len(items)}] FAILED id={item_id}: {last_error}")
            item["repair_error"] = last_error
            item["repair_raw_response"] = last_text
            item["repair_attempts"] = MAX_RETRIES
            item["repair_mode"] = "regenerate" if has_generation_failure(item) else "revise"

            usage_history.append(
                {
                    "id": item_id,
                    "attempt": MAX_RETRIES,
                    "status": "failed",
                    "mode": item["repair_mode"],
                    "error": last_error,
                }
            )

        if updated % SAVE_EVERY == 0 or not success:
            save_progress(data, items, usage_history)

    save_progress(data, items, usage_history)
    print(f"Saved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
