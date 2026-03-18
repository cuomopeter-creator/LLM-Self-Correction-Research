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


INPUT_PATH = HERE.parent / "prompt_visual_inspection.json"
OUTPUT_PATH = HERE.parent / "prompt_visual_inspection_rewritten.json"
MODEL_KEY = "gpt"
BATCH_SIZE = 10
MAX_RETRIES = 3


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

    raise ValueError("Could not find JSON array in model response.")


def build_rewrite_prompt(batch: list[dict[str, Any]]) -> str:
    batch_json = json.dumps(batch, indent=2, ensure_ascii=False)

    return f"""
Rewrite the "prompt" field for each object below.

Rules:
- Preserve id exactly
- Preserve category exactly
- Preserve difficulty exactly
- Preserve chart_family exactly
- Rewrite ONLY the prompt field
- Return the same number of objects you received
- Return ONLY valid JSON
- Return a JSON array, nothing else

Prompt quality requirements:
- Each rewritten prompt must be 2 to 4 sentences long
- Each prompt must sound like a realistic request from a data analyst, business user, or stakeholder
- Each prompt must clearly include:
  - at least one metric, such as revenue, sales, profit, orders, cost, conversion rate, traffic, rating, inventory, or delivery time
  - at least one dimension, such as region, category, segment, product, channel, country, date, month, quarter, or customer type
  - a clear analytical goal or business question
- Do not mention Plotly, Streamlit, plotly.express, or graph_objects
- Do not define a dataframe schema
- Do not use placeholders like "x" or "y"
- Do not make the prompt generic

Special instruction for chart_family = "graph_objects":
- Do not write feature-only prompts like "add annotations" or "use subplots"
- Instead embed the advanced chart requirement into a real analytical request
- These prompts should naturally call for things like subplots, dual axes, annotations, overlays, highlighted intervals, error bars, dashboards, or custom layout

Bad prompt example:
"Show grouped bar chart with error bars and annotations."

Good prompt example:
"Compare average delivery time across regions with grouped bars by quarter so we can see where service is most inconsistent. Include error bars to show variability and annotate the region with the worst average delivery time."

Objects to rewrite:
{batch_json}
""".strip()


def validate_rewrite(
    original_batch: list[dict[str, Any]],
    rewritten_batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(original_batch) != len(rewritten_batch):
        raise ValueError(
            f"Batch length mismatch. Expected {len(original_batch)}, got {len(rewritten_batch)}."
        )

    original_by_id = {int(item["id"]): item for item in original_batch}
    cleaned: list[dict[str, Any]] = []

    for item in rewritten_batch:
        if not isinstance(item, dict):
            raise ValueError("Rewritten item is not an object.")

        if "id" not in item:
            raise ValueError("Rewritten item missing id.")

        item_id = int(item["id"])
        if item_id not in original_by_id:
            raise ValueError(f"Unexpected id returned: {item_id}")

        original = original_by_id[item_id]

        if str(item.get("category", "")).strip() != str(original["category"]).strip():
            raise ValueError(f"Category changed for id {item_id}")

        if str(item.get("difficulty", "")).strip() != str(original["difficulty"]).strip():
            raise ValueError(f"Difficulty changed for id {item_id}")

        if str(item.get("chart_family", "")).strip() != str(original["chart_family"]).strip():
            raise ValueError(f"chart_family changed for id {item_id}")

        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Empty prompt for id {item_id}")

        sentence_count = len([s for s in re.split(r"[.!?]+", prompt) if s.strip()])
        if sentence_count < 2 or sentence_count > 4:
            raise ValueError(
                f"Prompt for id {item_id} has {sentence_count} sentences; expected 2 to 4."
            )

        merged = dict(original)
        merged["prompt"] = prompt
        cleaned.append(merged)

    cleaned.sort(key=lambda x: int(x["id"]))
    return cleaned


def rewrite_batch(model: OpenAIModel, batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = build_rewrite_prompt(batch)
    result = model.generate(prompt)
    text = str(result.get("text", "")).strip()
    usage = result.get("usage", {}) or {}

    json_text = extract_json_block(text)
    rewritten = json.loads(json_text)
    if not isinstance(rewritten, list):
        raise ValueError("Model response was not a JSON array.")

    cleaned = validate_rewrite(batch, rewritten)
    return cleaned, usage


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Input JSON does not contain a non-empty 'items' list.")

    model = build_gpt_model()

    rewritten_items: list[dict[str, Any]] = []
    usage_history: list[dict[str, Any]] = []

    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        batch_num = start // BATCH_SIZE + 1

        success = False
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"Batch {batch_num} | items {start + 1}-{start + len(batch)} | attempt {attempt}")
            try:
                rewritten_batch, usage = rewrite_batch(model, batch)
                rewritten_items.extend(rewritten_batch)
                usage_history.append(usage)
                success = True
                break
            except Exception as e:
                last_error = e
                print(f"  failed: {e}")

        if not success:
            raise RuntimeError(f"Failed batch {batch_num}: {last_error}")

    rewritten_items.sort(key=lambda x: int(x["id"]))

    output = dict(data)
    output["rewritten_at_utc"] = time.time()
    output["rewrite_source_model"] = MODEL_KEY
    output["rewrite_batch_size"] = BATCH_SIZE
    output["rewrite_usage"] = usage_history
    output["items"] = rewritten_items

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved rewritten prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
