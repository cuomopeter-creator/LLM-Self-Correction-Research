from __future__ import annotations
from datetime import datetime

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]

DATA_PATH = PROJECT_ROOT / "experiments" / "prompt_visual_inspection.jsonl"
CHAT_SCRIPT = PROJECT_ROOT / "scripts" / "chat_model.py"

SYSTEM_PROMPT = (
    "Generate a Streamlit app snippet using plotly.express or plotly.graph_objects only. "
    "Assume df already exists. Return code only."
)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def extract_model_text(stdout: str) -> str:
    text = stdout.strip()

    if "model>" in text:
        text = text.split("model>", 1)[1].strip()

    if "\n[usage]\n" in text:
        text = text.split("\n[usage]\n", 1)[0].strip()

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


def run_model(prompt: str, model_name: str) -> str:
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    result = subprocess.run(
        ["python", str(CHAT_SCRIPT), "--model", model_name, "--prompt", full_prompt],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "model call failed")

    return extract_model_text(result.stdout)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(DATA_PATH)

    for r in rows:
        if r.get("response"):
            continue

        print(f"Generating for prompt {r['id']}...")

        try:
            code = run_model(r["prompt"], args.model)
            r["response"] = code
            r["exec_pass"] = None
            r["status"] = r.get("status", "")
        except Exception as e:
            r["response"] = ""
            r["exec_pass"] = False
            r["status"] = "fail"
            r["notes"] = f"generation error: {e}"

    run_dir = PROJECT_ROOT / "experiments" / "runs"
    run_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = run_dir / f"{args.model}_{ts}.jsonl"

    save_jsonl(out_path, rows)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
