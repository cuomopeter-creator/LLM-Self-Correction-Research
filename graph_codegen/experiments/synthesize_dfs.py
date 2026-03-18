from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
BASE_DIR = HERE.parent

PREFERRED_INPUT = BASE_DIR / "prompt_visual_inspection_with_specs.json"
FALLBACK_INPUT = BASE_DIR / "prompt_visual_inspection_rewritten.json"
OUTPUT_PATH = BASE_DIR / "prompt_visual_inspection_with_data.json"

SAVE_EVERY = 1


def choose_input_path() -> Path:
    if PREFERRED_INPUT.exists():
        return PREFERRED_INPUT
    if FALLBACK_INPUT.exists():
        return FALLBACK_INPUT
    raise FileNotFoundError(
        f"Could not find input file. Checked:\n- {PREFERRED_INPUT}\n- {FALLBACK_INPUT}"
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def stable_seed(item_id: Any) -> int:
    try:
        return int(item_id) + 12345
    except Exception:
        return abs(hash(str(item_id))) % (2**32)


def parse_group_effects(text: str | None, categories: list[str]) -> dict[str, float]:
    if not text or not categories:
        return {}

    s = text.lower()
    effects: dict[str, float] = {}

    for cat in categories:
        cat_lower = str(cat).lower()
        effect = 0.0

        if cat_lower in s:
            if any(token in s for token in ["highest", "strongest", "faster", "higher", "upward", "accelerate"]):
                effect += 0.6
            if any(token in s for token in ["lowest", "slower", "lower", "declining", "weakest"]):
                effect -= 0.6
            if "mid" in s or "moderate" in s:
                effect += 0.15

        effects[cat] = effect

    return effects


def noise_scale_from_text(text: str | None) -> float:
    s = (text or "").lower()
    if "none" in s:
        return 0.0
    if "low" in s:
        return 0.05
    if "moderate" in s:
        return 0.12
    if "high" in s:
        return 0.22
    return 0.10


def is_right_skewed(text: str | None) -> bool:
    s = (text or "").lower()
    return "right_skew" in s or "log-normal" in s or "lognormal" in s or "skew" in s


def has_outliers(text: str | None) -> bool:
    s = (text or "").lower()
    return "outlier" in s or "spike" in s or "extreme" in s


def safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def build_base_frame(spec: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    columns = spec["columns"]
    row_count = safe_int(spec["row_count"], 240)
    cat_vals = spec.get("categorical_values", {}) or {}
    date_range = spec.get("date_range")

    cat_cols = [c for c, d in columns.items() if d == "category"]
    dt_cols = [c for c, d in columns.items() if d == "datetime"]

    if date_range and cat_cols:
        date_col = date_range["column"]
        dates = pd.date_range(
            start=date_range["start"],
            end=date_range["end"],
            freq=date_freq_to_pandas(date_range["freq"]),
        )
        if len(dates) == 0:
            dates = pd.date_range(start="2024-01-01", periods=max(12, row_count // 4), freq="MS")

        combos = [dates]
        combo_names = [date_col]

        for col in cat_cols:
            vals = cat_vals.get(col)
            if not vals:
                vals = default_categories_for_col(col)
            combos.append(vals)
            combo_names.append(col)

        grid = pd.MultiIndex.from_product(combos, names=combo_names).to_frame(index=False)

        if len(grid) >= row_count:
            df = grid.sample(n=row_count, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        else:
            repeats = math.ceil(row_count / len(grid))
            df = pd.concat([grid] * repeats, ignore_index=True).iloc[:row_count].copy()

    elif date_range:
        date_col = date_range["column"]
        dates = pd.date_range(
            start=date_range["start"],
            end=date_range["end"],
            freq=date_freq_to_pandas(date_range["freq"]),
        )
        if len(dates) == 0:
            dates = pd.date_range(start="2024-01-01", periods=row_count, freq="D")
        repeats = math.ceil(row_count / len(dates))
        df = pd.DataFrame({date_col: list(dates) * repeats}).iloc[:row_count].copy()

    elif cat_cols:
        data: dict[str, Any] = {}
        for col in cat_cols:
            vals = cat_vals.get(col)
            if not vals:
                vals = default_categories_for_col(col)
            data[col] = rng.choice(vals, size=row_count, replace=True)
        df = pd.DataFrame(data)

    else:
        df = pd.DataFrame(index=np.arange(row_count))

    for col, dtype in columns.items():
        if col not in df.columns:
            if dtype == "category":
                vals = cat_vals.get(col) or default_categories_for_col(col)
                df[col] = rng.choice(vals, size=len(df), replace=True)
            elif dtype == "datetime":
                df[col] = pd.date_range("2024-01-01", periods=len(df), freq="D")
            elif dtype == "bool":
                df[col] = False
            else:
                df[col] = np.nan

    return df.reset_index(drop=True)


def date_freq_to_pandas(freq: str) -> str:
    f = (freq or "").lower()
    if f == "day":
        return "D"
    if f == "week":
        return "W"
    if f == "month":
        return "MS"
    if f == "quarter":
        return "QS"
    return "D"


def default_categories_for_col(col: str) -> list[str]:
    c = col.lower()
    if "region" in c:
        return ["North", "South", "East", "West"]
    if "segment" in c:
        return ["Consumer", "Corporate", "Small Business"]
    if "category" in c:
        return ["A", "B", "C", "D"]
    if "channel" in c:
        return ["Search", "Social", "Email", "Display"]
    if "country" in c:
        return ["United States", "Canada", "Germany", "Japan"]
    if "type" in c:
        return ["Type A", "Type B", "Type C"]
    if "method" in c:
        return ["Standard", "Express", "Overnight"]
    return ["Group A", "Group B", "Group C"]


def add_time_signal(df: pd.DataFrame, date_col: str) -> np.ndarray:
    dt = pd.to_datetime(df[date_col])
    if len(dt) <= 1:
        return np.zeros(len(df))
    order = dt.rank(method="dense").to_numpy()
    order = (order - order.min()) / max(1.0, order.max() - order.min())
    return order.astype(float)


def apply_group_effect_vector(
    df: pd.DataFrame,
    candidate_cols: list[str],
    rule: dict[str, Any],
    spec: dict[str, Any],
) -> np.ndarray:
    vec = np.zeros(len(df), dtype=float)
    text = str(rule.get("group_effects", "")).strip()
    if not text:
        return vec

    cat_vals = spec.get("categorical_values", {}) or {}

    for col in candidate_cols:
        if col not in df.columns:
            continue
        categories = list(cat_vals.get(col, []))
        if not categories:
            categories = sorted(df[col].dropna().astype(str).unique().tolist())
        effects = parse_group_effects(text, categories)
        if not effects:
            continue
        vec += df[col].astype(str).map(effects).fillna(0.0).to_numpy()

    return vec


def generate_numeric_column(
    df: pd.DataFrame,
    col: str,
    dtype: str,
    rule: dict[str, Any],
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(df)
    lo = safe_float(rule.get("min"), 0.0)
    hi = safe_float(rule.get("max"), max(1.0, lo + 1.0))
    if hi <= lo:
        hi = lo + 1.0

    base = rng.uniform(lo, hi, size=n)

    if is_right_skewed(rule.get("noise")):
        sigma = 0.6
        raw = rng.lognormal(mean=0.0, sigma=sigma, size=n)
        raw = (raw - raw.min()) / max(1e-9, raw.max() - raw.min())
        base = lo + raw * (hi - lo)

    date_cols = [c for c, d in spec["columns"].items() if d == "datetime" and c in df.columns]
    trend_text = str(rule.get("trend", "")).lower()

    if date_cols:
        t = add_time_signal(df, date_cols[0])
        if "upward" in trend_text or "increase" in trend_text or "accelerate" in trend_text:
            base += t * 0.30 * (hi - lo)
        if "downward" in trend_text or "declining" in trend_text:
            base -= t * 0.30 * (hi - lo)
        if "seasonal" in trend_text or "sine" in trend_text:
            base += np.sin(2 * np.pi * t * 2) * 0.18 * (hi - lo)
        if "lagged" in trend_text:
            base += np.roll(t, 1) * 0.08 * (hi - lo)

    cat_candidate_cols = [c for c, d in spec["columns"].items() if d == "category" and c in df.columns]
    base += apply_group_effect_vector(df, cat_candidate_cols, rule, spec) * 0.12 * (hi - lo)

    ns = noise_scale_from_text(rule.get("noise"))
    if ns > 0:
        base += rng.normal(0.0, ns * (hi - lo), size=n)

    if has_outliers(rule.get("noise")):
        k = max(1, n // 25)
        idx = rng.choice(n, size=k, replace=False)
        base[idx] += rng.uniform(0.15, 0.50, size=k) * (hi - lo)

    base = np.clip(base, lo, hi)

    if dtype == "int":
        return np.rint(base).astype(int)
    return base.astype(float)


def realize_derived_columns(df: pd.DataFrame, spec: dict[str, Any], rng: np.random.Generator) -> None:
    rules = spec.get("numeric_rules", {}) or {}
    cols = spec["columns"]

    for col, dtype in cols.items():
        if col not in rules:
            continue

        rule = rules[col]
        derived = str(rule.get("derived_from", "")).lower()
        correlated = str(rule.get("correlated_with", "")).strip()

        if derived:
            if "revenue - cost" in derived and {"revenue", "cost"}.issubset(df.columns):
                vals = df["revenue"].astype(float) - df["cost"].astype(float)
                vals += rng.normal(0, max(1.0, vals.std(ddof=0) * 0.03), size=len(df))
                df[col] = vals
            elif "conversions / traffic_volume" in derived and {"conversions", "traffic_volume"}.issubset(df.columns):
                denom = df["traffic_volume"].replace(0, 1).astype(float)
                df[col] = df["conversions"].astype(float) / denom
            elif "delivery_time_days - promised_time_days" in derived and {"delivery_time_days", "promised_time_days"}.issubset(df.columns):
                df[col] = df["delivery_time_days"].astype(float) - df["promised_time_days"].astype(float)
            elif "avg_temperature - normal_temperature" in derived and {"avg_temperature", "normal_temperature"}.issubset(df.columns):
                df[col] = df["avg_temperature"].astype(float) - df["normal_temperature"].astype(float)
            elif "revenue / ad_spend" in derived and {"revenue", "ad_spend"}.issubset(df.columns):
                denom = df["ad_spend"].replace(0, 1).astype(float)
                df[col] = df["revenue"].astype(float) / denom
            elif "revenue / order_count" in derived and {"revenue", "order_count"}.issubset(df.columns):
                denom = df["order_count"].replace(0, 1).astype(float)
                df[col] = df["revenue"].astype(float) / denom
            elif "sales" in derived and "cumulative" in col.lower() and "sales" in df.columns:
                if any(spec["columns"].get(c) == "datetime" for c in df.columns):
                    dt_cols = [c for c, d in spec["columns"].items() if d == "datetime" and c in df.columns]
                    cat_cols = [c for c, d in spec["columns"].items() if d == "category" and c in df.columns]
                    sort_cols = dt_cols.copy()
                    if sort_cols:
                        if cat_cols:
                            group_col = cat_cols[0]
                            df.sort_values([group_col] + sort_cols, inplace=True)
                            df[col] = df.groupby(group_col)["sales"].cumsum()
                        else:
                            df.sort_values(sort_cols, inplace=True)
                            df[col] = df["sales"].cumsum()
                        df.reset_index(drop=True, inplace=True)
                else:
                    df[col] = df["sales"].cumsum()
            elif "sales" in derived and "growth" in col.lower() and "sales" in df.columns:
                dt_cols = [c for c, d in spec["columns"].items() if d == "datetime" and c in df.columns]
                cat_cols = [c for c, d in spec["columns"].items() if d == "category" and c in df.columns]
                if dt_cols:
                    sort_cols = dt_cols.copy()
                    if cat_cols:
                        g = cat_cols[0]
                        df.sort_values([g] + sort_cols, inplace=True)
                        df[col] = df.groupby(g)["sales"].pct_change().fillna(0.0)
                    else:
                        df.sort_values(sort_cols, inplace=True)
                        df[col] = df["sales"].pct_change().fillna(0.0)
                    df.reset_index(drop=True, inplace=True)
            elif derived == "revenue and cost" and {"revenue", "cost"}.issubset(df.columns):
                denom = df["revenue"].replace(0, 1).astype(float)
                df[col] = (df["revenue"].astype(float) - df["cost"].astype(float)) / denom

        elif correlated and correlated in df.columns:
            ref_series = df[correlated]
            if pd.api.types.is_numeric_dtype(ref_series):
                ref = ref_series.astype(float)
                lo = safe_float(rule.get("min"), float(ref.min()))
                hi = safe_float(rule.get("max"), float(ref.max()))
                ref_norm = (ref - ref.min()) / max(1e-9, ref.max() - ref.min())
                vals = lo + ref_norm * (hi - lo)
                vals += rng.normal(0, noise_scale_from_text(rule.get("noise")) * (hi - lo), size=len(df))
                df[col] = np.clip(vals, lo, hi)

    for col, dtype in cols.items():
        if dtype == "bool":
            lower = col.lower()
            if lower == "is_anomaly" and "temperature_deviation" in df.columns:
                df[col] = df["temperature_deviation"].abs() >= 5
            elif lower.startswith("is_") and "discount_pct" in df.columns:
                df[col] = df["discount_pct"] > 0.15
            else:
                df[col] = False


def coerce_types(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    for col, dtype in spec["columns"].items():
        if dtype == "datetime":
            df[col] = pd.to_datetime(df[col])
        elif dtype == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int)
        elif dtype == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        elif dtype == "bool":
            df[col] = df[col].astype(bool)
        elif dtype == "category":
            df[col] = df[col].astype(str)
    return df


def build_dataframe_from_spec(item: dict[str, Any]) -> pd.DataFrame:
    spec = item.get("df_spec") or {}
    if not spec:
        raise ValueError(f"id={item.get('id')} missing df_spec")

    rng = np.random.default_rng(stable_seed(item.get("id")))
    df = build_base_frame(spec, rng)

    rules = spec.get("numeric_rules", {}) or {}
    columns = spec["columns"]

    for col, dtype in columns.items():
        if dtype not in {"int", "float"}:
            continue

        rule = rules.get(col)
        if not rule:
            if dtype == "int":
                df[col] = rng.integers(1, 1000, size=len(df))
            else:
                df[col] = rng.uniform(0, 100, size=len(df))
            continue

        if rule.get("derived_from") or rule.get("correlated_with"):
            continue

        df[col] = generate_numeric_column(df, col, dtype, rule, spec, rng)

    realize_derived_columns(df, spec, rng)
    df = coerce_types(df, spec)

    ordered_cols = list(columns.keys())
    df = df[ordered_cols]

    return df


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(bool)
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out.to_dict(orient="records")


def main() -> None:
    input_path = choose_input_path()
    data = load_json(input_path)

    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Input JSON does not contain a non-empty 'items' list.")

    if OUTPUT_PATH.exists():
        existing = load_json(OUTPUT_PATH)
        existing_items = existing.get("items", [])
        if isinstance(existing_items, list) and len(existing_items) == len(items):
            items = existing_items
            data = existing

    updated = 0

    for i, item in enumerate(items, start=1):
        if item.get("df"):
            print(f"[{i}/{len(items)}] skip id={item.get('id')} (already has df)")
            continue

        print(f"[{i}/{len(items)}] synthesizing df for id={item.get('id')}")
        df = build_dataframe_from_spec(item)
        item["df"] = dataframe_to_records(df)
        updated += 1

        if updated % SAVE_EVERY == 0:
            data["items"] = items
            data["df_synthesized_at_utc"] = pd.Timestamp.utcnow().timestamp()
            save_json(OUTPUT_PATH, data)

    data["items"] = items
    data["df_synthesized_at_utc"] = pd.Timestamp.utcnow().timestamp()
    save_json(OUTPUT_PATH, data)

    print(f"Saved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
