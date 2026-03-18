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


INPUT_PATH = BASE_DIR / "prompt_visual_inspection_audited.json"
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
            max_output_tokens=model_cfg.get("max_output_tokens", 5000),
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
        json.loads(text)
        return text
    except Exception:
        pass

    match = re.search(r"(\[\s*.*\s*\])", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        json.loads(candidate)
        return candidate

    raise ValueError("Could not find JSON array in model response.")


def build_repair_prompt(item: dict[str, Any]) -> str:

    return f"""
You are repairing a synthetic dataset used for visualization testing.

Prompt:
{item["prompt"]}

Detected issues:
{item["errors"]}

Dataset:
{json.dumps(item["df"], indent=2)}

Repair the dataset so that:

- No NULL values exist
- ID columns contain unique values
- Numeric metrics are not heavily clipped at min/max
- Derived metrics are correctly computed
- Do NOT change column names or schema

Return ONLY the corrected dataframe.

Return a JSON array of rows.
Do not include explanations.
Do not include markdown.
"""


def repair_item(model: OpenAIModel, item: dict[str, Any]) -> tuple[list[dict], dict]:

    prompt = build_repair_prompt(item)

    result = model.generate(prompt)

    text = str(result.get("text", "")).strip()
    usage = result.get("usage", {}) or {}

    json_text = extract_json_block(text)

    repaired_df = json.loads(json_text)

    if not isinstance(repaired_df, list):
        raise ValueError("Model response was not a JSON list.")

    return repaired_df, usage


def save_output(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():

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

        if not item.get("errors"):
            continue

        print(f"[{i}/{len(items)}] repairing id={item.get('id')}")

        success = False
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):

            try:

                repaired_df, usage = repair_item(model, item)

                item["df"] = repaired_df
                item["errors"] = []

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
            print(f"Failed repairing id={item.get('id')}: {last_error}")

        if updated % SAVE_EVERY == 0:
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
