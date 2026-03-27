import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]          # graph_codegen/
DATA_DIR = PROJECT_ROOT / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dedupe_key(ex: dict) -> tuple[str, str]:
    messages = ex.get("messages", [])
    user_msg = messages[1]["content"].strip() if len(messages) > 1 else ""
    assistant_msg = messages[2]["content"].strip() if len(messages) > 2 else ""
    return user_msg, assistant_msg


def resolve_dataset_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return DATA_DIR / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default="plotly_streamlit_master.jsonl")
    parser.add_argument("--new", default="plotly_streamlit_train_runtime_pass.jsonl")
    args = parser.parse_args()

    master_path = resolve_dataset_path(args.master)
    new_path = resolve_dataset_path(args.new)

    if not new_path.exists():
        raise FileNotFoundError(f"New dataset not found: {new_path}")

    master_keys = set()
    existing_count = 0

    if master_path.exists():
        for ex in load_jsonl(master_path):
            master_keys.add(dedupe_key(ex))
            existing_count += 1

    new_examples = list(load_jsonl(new_path))
    added = 0
    skipped = 0

    with master_path.open("a", encoding="utf-8") as fout:
        for ex in new_examples:
            key = dedupe_key(ex)
            if key in master_keys:
                skipped += 1
                continue

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            master_keys.add(key)
            added += 1

    print()
    print("Existing master rows:", existing_count)
    print("New examples processed:", len(new_examples))
    print("Added to master:", added)
    print("Skipped as duplicates:", skipped)
    print("Master dataset:", master_path)


if __name__ == "__main__":
    main()