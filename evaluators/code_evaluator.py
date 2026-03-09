from __future__ import annotations
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass


@dataclass
class CodeEvalResult:
    passed: bool
    returncode: int
    stdout: str
    stderr: str


def _extract_python_stub(prompt: str) -> str:
    marker = "\n\nfrom "
    alt_marker = "\n\ndef "

    if marker in prompt:
        return "from " + prompt.split(marker, 1)[1]
    if alt_marker in prompt:
        return "def " + prompt.split(alt_marker, 1)[1]

    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("from ") or line.startswith("def "):
            return "\n".join(lines[i:])

    return prompt


def _strip_code_fences(text: str) -> str:
    t = text.strip()

    if t.startswith("```python"):
        t = t[len("```python"):].lstrip()
    elif t.startswith("```"):
        t = t[len("```"):].lstrip()

    if t.endswith("```"):
        t = t[:-3].rstrip()

    return t


def _indent_body(completion: str) -> str:
    cleaned = _strip_code_fences(completion)
    normalized = textwrap.dedent(cleaned).strip("\n")
    lines = normalized.splitlines()

    out = []
    for line in lines:
        if line.strip() == "":
            out.append("")
        else:
            out.append("    " + line)
    return "\n".join(out)


def evaluate_humaneval(prompt: str, completion: str, test_code: str, entry_point: str, timeout_s: int = 5) -> CodeEvalResult:
    python_stub = _extract_python_stub(prompt)
    indented_completion = _indent_body(completion)

    full_code = (
        python_stub.rstrip() + "\n"
        + indented_completion + "\n\n"
        + test_code.rstrip() + "\n\n"
        + f"check({entry_point})\n"
        + 'print("PASS")\n'
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        path = f.name

    try:
        proc = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CodeEvalResult(
            passed=(proc.returncode == 0),
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired as e:
        return CodeEvalResult(
            passed=False,
            returncode=-1,
            stdout=e.stdout or "",
            stderr=f"TIMEOUT after {timeout_s}s",
        )
