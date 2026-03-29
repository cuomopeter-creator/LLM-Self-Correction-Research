from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = PROJECT_ROOT / "graph_codegen" / "past_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_MARKER = "__LLM_PLOT_HTML__="


def extract_python_code(text: str) -> str | None:
    fenced = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()

    any_fenced = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if any_fenced:
        candidate = any_fenced[0].strip()
        if looks_like_python(candidate):
            return candidate

    stripped = text.strip()
    if looks_like_python(stripped):
        return stripped

    return None


def looks_like_python(text: str) -> bool:
    starters = (
        "import ",
        "from ",
        "def ",
        "class ",
        "print(",
        "for ",
        "while ",
        "if ",
        "try:",
        "with ",
        "x =",
        "y =",
    )
    return text.startswith(starters)


def prepare_code_for_execution(code: str, workdir: str | Path) -> tuple[str, str | None]:
    plot_path: str | None = None
    patched = code

    if "plotly" in code.lower():
        plot_path = str((PLOTS_DIR / f"plot_{int(time.time())}.html").resolve())

        def repl(match: re.Match) -> str:
            fig_var = match.group(1)
            return (
                f'{fig_var}.write_html(r"{plot_path}")\n'
                f'print("{PLOT_MARKER}{plot_path}")'
            )

        patched = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\.show\(\)", repl, patched)

    return patched, plot_path


@dataclass
class ExecResult:
    code_path: str
    returncode: int
    stdout: str
    stderr: str
    plot_html_path: str | None = None


def extract_plot_path(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith(PLOT_MARKER):
            return line.split(PLOT_MARKER, 1)[1].strip()
    return None


def execute_python_isolated(
    code: str,
    *,
    workdir: str | Path | None = None,
    timeout: int = 120,
    python_executable: str | None = None,
) -> ExecResult:
    workdir = Path(workdir or PROJECT_ROOT)
    python_executable = python_executable or sys.executable

    prepared_code, hinted_plot_path = prepare_code_for_execution(code, workdir)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="llm_exec_",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(prepared_code)
        temp_path = f.name

    try:
        proc = subprocess.run(
            [python_executable, temp_path],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        plot_html_path = extract_plot_path(proc.stdout) or hinted_plot_path
        return ExecResult(
            code_path=temp_path,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            plot_html_path=plot_html_path,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return ExecResult(
            code_path=temp_path,
            returncode=124,
            stdout=stdout,
            stderr=stderr + f"\nExecution timed out after {timeout} seconds.",
            plot_html_path=hinted_plot_path,
        )


def maybe_open_plot(plot_html_path: str | None) -> None:
    if not plot_html_path:
        return

    path = Path(plot_html_path)
    if not path.exists():
        return

    try:
        windows_path = subprocess.check_output(
            ["wslpath", "-w", str(path)],
            text=True,
        ).strip()

        subprocess.Popen(
            ["cmd.exe", "/C", "start", "", windows_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"\n[plot saved] {path}")
        print("[opened in default Windows browser]")

    except Exception as e:
        print(f"\n[plot saved] {path}")
        print(f"[could not auto-open browser] {e}")


def print_usage(usage: dict) -> None:
    if not usage:
        return
    print("\n[usage]")
    print(json.dumps(usage, indent=2))
