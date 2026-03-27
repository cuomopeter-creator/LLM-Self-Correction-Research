import json
from pathlib import Path

DATASET_PATH = Path("datasets/plotly_streamlit_train.jsonl")

plotly_calls = [
    "px.scatter",
    "px.line",
    "px.bar",
    "px.histogram",
    "px.box",
    "px.density_heatmap",
]

json_errors = 0
structure_errors = 0
syntax_errors = 0
plotly_missing = 0
total = 0


def validate_structure(ex):
    if "messages" not in ex:
        return False

    if len(ex["messages"]) != 3:
        return False

    roles = [m["role"] for m in ex["messages"]]
    return roles == ["system", "user", "assistant"]


def main():
    global json_errors, structure_errors, syntax_errors, plotly_missing, total

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                ex = json.loads(line)
            except Exception:
                json_errors += 1
                continue

            total += 1

            if not validate_structure(ex):
                structure_errors += 1
                continue

            code = ex["messages"][2]["content"]

            try:
                compile(code, "<string>", "exec")
            except Exception:
                syntax_errors += 1

            if not any(call in code for call in plotly_calls):
                plotly_missing += 1

    print()
    print("Dataset validation results")
    print("--------------------------")
    print(f"Examples scanned: {total}")
    print(f"JSON errors: {json_errors}")
    print(f"Structure errors: {structure_errors}")
    print(f"Syntax errors: {syntax_errors}")
    print(f"Missing plotly call: {plotly_missing}")
    print()


if __name__ == "__main__":
    main()
