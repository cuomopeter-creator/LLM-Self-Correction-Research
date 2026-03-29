from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


def load_tabular_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".json":
        return pd.read_json(path)

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    if suffix in {".db", ".sqlite"}:
        conn = sqlite3.connect(path)
        try:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1",
                conn,
            )
            if tables.empty:
                raise ValueError(f"No tables found in SQLite file: {path}")
            table_name = tables.iloc[0, 0]
            return pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
        finally:
            conn.close()

    raise ValueError(f"Unsupported file type: {suffix}")


def summarize_dataframe(df: pd.DataFrame, path: str | Path, sample_rows: int = 5) -> str:
    path = Path(path)
    lines: list[str] = []
    lines.append(f"Loaded file: {path}")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append("")
    lines.append("Schema:")
    for col, dtype in df.dtypes.items():
        lines.append(f"- {col}: {dtype}")

    if not df.empty:
        lines.append("")
        lines.append(f"Sample rows ({min(sample_rows, len(df))}):")
        sample_df = df.head(sample_rows).copy()
        lines.append(sample_df.to_string(index=False))

    return "\n".join(lines)


def load_and_summarize_files(paths: list[str] | None) -> str:
    if not paths:
        return ""

    blocks: list[str] = []
    for raw_path in paths:
        df = load_tabular_file(raw_path)
        blocks.append(summarize_dataframe(df, raw_path))

    return "\n\n".join(blocks)
