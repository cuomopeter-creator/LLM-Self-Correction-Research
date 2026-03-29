from __future__ import annotations
import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent

load_dotenv(ROOT_DIR / ".env", override=True)

sys.path.append(str(ROOT_DIR))

from logger import JSONLLogger, RunMeta, make_run_id
from models.anthropic_model import AnthropicModel, AnthropicModelConfig


INPUT_PATH = BASE_DIR / "datasets" / "plotly_streamlit_train_raw_3000.jsonl"
OUTPUT_PATH = BASE_DIR / "datasets" / "plotly_streamlit_train_revised.jsonl"
FAILED_PATH = BASE_DIR / "datasets" / "plotly_streamlit_train_failed.jsonl"


def load_examples(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield str(i), json.loads(line)


def strip_code_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    return text.strip()


def build_repair_prompt(system_msg: str, user_msg: str, assistant_code: str) -> str:
    return f"""You are repairing a Python Streamlit + Plotly Express training example.

Original system instruction:
{system_msg}

Original user request:
{user_msg}

Original assistant code:
{assistant_code}

Revise the code so it is valid and likely to run correctly.

Rules:
- preserve the user's requested chart intent
- use streamlit and plotly.express only
- assume df already exists
- return code only
- do not use markdown fences
- do not explain anything
"""


def build_retry_prompt(bad_code: str, error: str) -> str:
    return f"""The following Python code failed validation.

Error:
{error}

Code:
{bad_code}

Fix the code.

Rules:
- return only corrected Python code
- no explanations
- no markdown
"""


def validate_code(code: str):
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    if "st.plotly_chart(" not in code:
        return False, "Missing st.plotly_chart call"

    if "px." not in code:
        return False, "Missing plotly express usage"

    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model_cfg = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 1200,
        "temperature": 0.0,
    }

    model = AnthropicModel(
        AnthropicModelConfig(
            model=model_cfg["model"],
            api_key_env=model_cfg["api_key_env"],
            max_tokens=model_cfg["max_tokens"],
            temperature=model_cfg["temperature"],
        )
    )

    run_id = make_run_id("revise")
    run_dir = ROOT_DIR / "runs" / run_id

    logger = JSONLLogger(
        run_dir=str(run_dir),
        meta=RunMeta(
            run_id=run_id,
            created_at_utc=time.time(),
            model_name="claude_reviser",
            model_cfg=model_cfg,
            strategy_name="dataset_revision",
            task_name="plotly_streamlit_repair",
        ),
    )

    processed = 0
    revised_ok = 0
    failed = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as fout, FAILED_PATH.open("w", encoding="utf-8") as ffail:
        for ex_id, rec in load_examples(INPUT_PATH):

            if args.limit and processed >= args.limit:
                break

            msgs = rec["messages"]
            system_msg = msgs[0]["content"]
            user_msg = msgs[1]["content"]
            assistant_code = msgs[2]["content"]

            prompt = build_repair_prompt(system_msg, user_msg, assistant_code)

            print(f"[{processed+1}] Revising example {ex_id}")

            try:
                result = model.generate(prompt, max_tokens=1200)
                code = strip_code_fences(result["text"])

                ok, error = validate_code(code)

                if not ok:
                    retry_prompt = build_retry_prompt(code, error)
                    retry = model.generate(retry_prompt, max_tokens=1200)
                    code = strip_code_fences(retry["text"])
                    ok, error = validate_code(code)

                if ok:
                    rec["messages"][-1]["content"] = code
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    revised_ok += 1
                    status = "revised"
                else:
                    ffail.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    failed += 1
                    status = "failed_validation"

                logger.log(
                    example_id=ex_id,
                    prompt=user_msg,
                    output=code,
                    latency_s=0,
                    score=None,
                    extra={
                        "status": status,
                        "validation_error": error,
                    },
                )

            except Exception as e:
                ffail.write(json.dumps(rec, ensure_ascii=False) + "\n")
                failed += 1

                logger.log(
                    example_id=ex_id,
                    prompt=user_msg,
                    output="",
                    latency_s=0,
                    score=None,
                    extra={
                        "status": "exception",
                        "error": str(e),
                    },
                )

            processed += 1

    print()
    print("Processed:", processed)
    print("Revised OK:", revised_ok)
    print("Failed:", failed)
    print("Revised file:", OUTPUT_PATH)
    print("Failed file:", FAILED_PATH)


if __name__ == "__main__":
    main()
