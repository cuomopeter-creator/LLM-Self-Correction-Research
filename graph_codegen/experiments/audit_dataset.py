import json
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("/workspace/self_correction_eval/graph_codegen/experiments/prompt_visual_inspection_with_data.json")
OUTPUT_PATH = Path("/workspace/self_correction_eval/graph_codegen/experiments/prompt_visual_inspection_audited.json")

CLIP_THRESHOLD = 0.05

ID_SKIP_TOKENS = ["id", "year", "month", "day", "index"]


def is_identifier(col):
    name = col.lower()
    return any(tok in name for tok in ID_SKIP_TOKENS)


def audit_dataframe(df):
    errors = []

    if df.empty:
        errors.append("EMPTY_DATAFRAME")
        return errors

    # null values
    if df.isna().any().any():
        errors.append("NULL_VALUES")

    # duplicate rows
    if df.duplicated().any():
        errors.append("DUPLICATE_ROWS")

    # duplicate ids
    id_cols = [c for c in df.columns if "id" in c.lower()]
    for col in id_cols:
        if df[col].duplicated().any():
            errors.append("DUPLICATE_ID_VALUES")

    # numeric columns only for metric checks
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:

        if is_identifier(col):
            continue

        series = df[col]

        # constant numeric metric
        if series.nunique() <= 1:
            errors.append(f"CONSTANT_METRIC_{col}")

        # clipping check
        min_val = series.min()
        max_val = series.max()

        min_ratio = (series == min_val).mean()
        max_ratio = (series == max_val).mean()

        if min_ratio > CLIP_THRESHOLD or max_ratio > CLIP_THRESHOLD:
            errors.append(f"CLIPPED_VALUES_{col}")

    return list(set(errors))


def main():
    data = json.loads(INPUT_PATH.read_text())

    items = data["items"]

    for item in items:

        if not item.get("df"):
            item["errors"] = ["MISSING_DATAFRAME"]
            continue

        df = pd.DataFrame(item["df"])

        errors = audit_dataframe(df)

        item["errors"] = errors

    OUTPUT_PATH.write_text(json.dumps(data, indent=2))

    print("Audit complete")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
