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


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
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


def normalize_list_of_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return [str(value).strip()]


def get_hard_errors(item: dict[str, Any]) -> list[str]:
    return normalize_list_of_strings(item.get("errors"))


def get_soft_reviews(item: dict[str, Any]) -> list[str]:
    review_keys = [
        "review_notes",
        "reviews",
        "revision_notes",
        "revisions",
        "soft_errors",
        "warnings",
        "advisory_notes",
    ]

    collected: list[str] = []
    for key in review_keys:
        collected.extend(normalize_list_of_strings(item.get(key)))

    return collected


def build_repair_prompt(item: dict[str, Any]) -> str:
    prompt = str(item.get("prompt", "")).strip()
    category = str(item.get("category", "")).strip()
    difficulty = str(item.get("difficulty", "")).strip()
    chart_family = str(item.get("chart_family", "")).strip()
    df_spec = item.get("df_spec") or {}
    df = item.get("df") or []
    hard_errors = get_hard_errors(item)
    soft_reviews = get_soft_reviews(item)

    hard_text = json.dumps(hard_errors, indent=2, ensure_ascii=False)
    soft_text = json.dumps(soft_reviews, indent=2, ensure_ascii=False)

    return f"""
You are revising a synthetic dataframe used for visualization testing.

Return ONLY a JSON array of row objects.
Do not include markdown.
Do not include explanations.
Do not include code fences.
Do not include any wrapper object.

There are TWO categories of revision instructions:

1. HARD FIXES
These are actual errors that must be fixed firmly.
If any hard fix conflicts with the current dataframe, the hard fix wins.

2. SOFT REVIEWS / REVISIONS
These are advisory improvements.
They should guide the revision when reasonable, but they are not strict rejection criteria.
Do not radically rewrite the dataframe just to satisfy a soft review.
Use them as quality-improvement suggestions.

Global rules:
- Preserve the existing schema exactly
- Do not change column names
- Do not add columns
- Do not remove columns
- Preserve the general intent of the dataset
- Keep the dataset realistic and plot-friendly
- Keep values internally coherent
- Keep the returned data as a JSON array of rows

Hard fixes to apply firmly:
{hard_text}

Soft reviews to consider gently:
{soft_text}

These soft reviews correspond to previously removed validation ideas.
Treat them as revision guidance only, not as mandatory pass/fail constraints:
- approximate row count usefulness
- preferred dtype cleanliness
- preferred category alignment
- preferred date-range alignment
- preferred numeric min/max plausibility

Visualization prompt:
{json.dumps(prompt)}

Metadata:
- category: {json.dumps(category)}
- difficulty: {json.dumps(difficulty)}
- chart_family: {json.dumps(chart_family)}

Dataframe spec:
{json.dumps(df_spec, indent=2, ensure_ascii=False)}

Current dataframe:
{json.dumps(df, indent=2, ensure_ascii=False)}

Revision instructions:
- First fix all hard errors
- Then improve the dataframe using the soft reviews where reasonable
- Keep the result practical for the intended chart
- Do not output anything except the corrected JSON array
""".strip()


def repair_item(model: OpenAIModel, item: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    prompt = build_repair_prompt(item)
    result = model.generate(prompt)

    text = str(result.get("text", "")).strip()
    usage = result.get("usage", {}) or {}

    json_text = extract_json_block(text)
    repaired_df = json.loads(json_text)

    if not isinstance(repaired_df, list):
        raise ValueError("Model response was not a JSON list.")

    for i, row in enumerate(repaired_df, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not an object.")

    return repaired_df, usage, text


def save_output(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def should_repair_item(item: dict[str, Any]) -> bool:
    hard_errors = get_hard_errors(item)
    soft_reviews = get_soft_reviews(item)
    return bool(hard_errors or soft_reviews)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
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

        if not should_repair_item(item):
            continue

        if not isinstance(item.get("df"), list) or not item.get("df"):
            print(f"[{i}/{len(items)}] skipping id={item_id} (no dataframe to repair)")
            continue

        print(f"[{i}/{len(items)}] repairing id={item_id}")

        success = False
        last_error = ""
        last_text = ""

        for attempt in range(1, MAX_RETRIES + 1):
            raw_text = ""
            try:
                repaired_df, usage, raw_text = repair_item(model, item)

                item["df"] = repaired_df
                item["errors"] = []

                if get_soft_reviews(item):
                    item["review_status"] = "addressed"

                item["repair_attempts"] = attempt
                item.pop("repair_error", None)
                item.pop("repair_raw_response", None)

                usage_history.append(
                    {
                        "id": item_id,
                        "attempt": attempt,
                        "status": "success",
                        "hard_error_count": len(get_hard_errors(item)),
                        "soft_review_count": len(get_soft_reviews(item)),
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
            item["repair_error"] = last_error
            item["repair_raw_response"] = last_text
            item["repair_attempts"] = MAX_RETRIES

            usage_history.append(
                {
                    "id": item_id,
                    "attempt": MAX_RETRIES,
                    "status": "failed",
                    "hard_error_count": len(get_hard_errors(item)),
                    "soft_review_count": len(get_soft_reviews(item)),
                    "error": last_error,
                }
            )

        if updated % SAVE_EVERY == 0 or not success:
            data["items"] = items
            data["repair_usage"] = usage_history
            data["repair_timestamp"] = time.time()
            save_output(OUTPUT_PATH, data)

    data["items"] = items
    data["repair_usage"] = usage_history
    data["repair_timestamp"] = time.time()
    save_output(OUTPUT_PATH, data)

    print(f"Saved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
