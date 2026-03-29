from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from chat_exec import (
    execute_python_isolated,
    extract_python_code,
    maybe_open_plot,
    print_usage,
)
from chat_files import load_and_summarize_files
from chat_models import build_model, generate_with_model, load_yaml
from chat_prompting import build_prompt


MAX_HISTORY_TURNS = 2


def run_and_maybe_execute_response(
    model,
    model_cfg: dict,
    prompt: str,
    *,
    execute_python: bool,
    timeout: int,
    workdir: str | Path,
) -> tuple[str, dict]:
    resp = generate_with_model(model, model_cfg, prompt)
    text = (resp.get("text") or "").strip()
    usage = resp.get("usage", {}) or {}

    print("\nmodel>\n")
    print(text)
    print_usage(usage)

    if execute_python:
        code = extract_python_code(text)
        if not code:
            print("\n[no python code detected]")
        else:
            result = execute_python_isolated(
                code,
                workdir=workdir,
                timeout=timeout,
            )

            print("\n[python detected and executed]")
            print(f"temp file: {result.code_path}")
            print(f"return code: {result.returncode}")

            if result.stdout:
                print("\n[stdout]\n")
                print(result.stdout)

            if result.stderr:
                print("\n[stderr]\n")
                print(result.stderr)

            maybe_open_plot(result.plot_html_path)

    return text, usage


def trim_history(history: list[dict[str, str]], max_turns: int) -> list[dict[str, str]]:
    max_messages = max_turns * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def interactive_chat(
    model,
    model_cfg: dict,
    *,
    execute_python: bool = False,
    timeout: int = 120,
    workdir: str | Path | None = None,
    file_context: str = "",
    max_history_turns: int = MAX_HISTORY_TURNS,
) -> None:
    print("Type 'exit' or 'quit' to stop.\n")
    history: list[dict[str, str]] = []

    while True:
        try:
            user_prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if user_prompt.lower() in {"exit", "quit"}:
            print("bye")
            break

        if not user_prompt:
            continue

        prompt = build_prompt(
            user_prompt,
            file_context=file_context,
            history=history,
        )

        assistant_text, _ = run_and_maybe_execute_response(
            model,
            model_cfg,
            prompt,
            execute_python=execute_python,
            timeout=timeout,
            workdir=workdir or PROJECT_ROOT,
        )

        history.append({"role": "user", "content": user_prompt})
        history.append({"role": "assistant", "content": assistant_text})
        history = trim_history(history, max_history_turns)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with any configured model.")
    p.add_argument("--model", required=True, help="Model key from configs/models.yaml")
    p.add_argument("--prompt", default=None, help="Single prompt mode")
    p.add_argument("--file", action="append", default=[], help="Optional data file path. Repeat for multiple files.")
    p.add_argument("--execute-python", action="store_true", help="Execute returned python in a subprocess")
    p.add_argument("--timeout", type=int, default=120, help="Execution timeout in seconds")
    p.add_argument("--workdir", default=str(PROJECT_ROOT), help="Working directory for executed python")
    p.add_argument("--max-history-turns", type=int, default=MAX_HISTORY_TURNS, help="How many prior user/assistant turns to keep")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_yaml(PROJECT_ROOT / "configs" / "models.yaml")
    model_cfg = cfg["models"][args.model]
    model = build_model(model_cfg)

    file_context = load_and_summarize_files(args.file)

    if file_context:
        print("\n[file context loaded]\n")
        print(file_context)
        print()

    if args.prompt:
        prompt = build_prompt(args.prompt, file_context=file_context, history=[])
        run_and_maybe_execute_response(
            model,
            model_cfg,
            prompt,
            execute_python=args.execute_python,
            timeout=args.timeout,
            workdir=args.workdir,
        )
        return

    interactive_chat(
        model,
        model_cfg,
        execute_python=args.execute_python,
        timeout=args.timeout,
        workdir=args.workdir,
        file_context=file_context,
        max_history_turns=args.max_history_turns,
    )


if __name__ == "__main__":
    main()
