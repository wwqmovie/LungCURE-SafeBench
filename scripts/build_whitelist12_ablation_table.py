"""Build a model-level gate-ablation table for the included models."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_leaderboard_data import read_union, round_float, safety_summary  # noqa: E402
from scripts.build_whitelist12_leaderboard_data import INCLUDED_MODELS  # noqa: E402
from scripts.update_new_models_progress import MODEL_GROUPS, paths_for  # noqa: E402


OUT_DIR = ROOT / "paper" / "generated"
METHODS = ["safe", "safe_no_mig", "safe_no_uef", "safe_no_gcv", "safe_no_hrc"]
MODEL_ORDER = [
    "Agnes-2.0-Flash",
    "DeepSeek-v4-Flash",
    "MiMo-v2.5",
    "GPT-OSS-120B",
    "Llama-4-Maverick",
    "Step-3.5-Flash",
    "Step-3.7-Flash",
    "Qwen3.5-2B",
    "Qwen3.5-35B-A3B",
    "Qwen3.6-35B-A3B",
    "GLM-Z1-9B",
    "GLM-4-9B",
    "DeepSeek-R1-Qwen3-8B",
]


def display_model(model: str) -> str:
    mapping = {
        "agnes-2.0-flash": "Agnes-2.0-Flash",
        "deepseek-v4-flash": "DeepSeek-v4-Flash",
        "mimo-v2.5": "MiMo-v2.5",
        "openai/gpt-oss-120b": "GPT-OSS-120B",
        "llama-4-maverick-17b-128e-instruct": "Llama-4-Maverick",
        "step-3.5-flash": "Step-3.5-Flash",
        "step-3.7-flash": "Step-3.7-Flash",
        "Qwen3.5-2B": "Qwen3.5-2B",
        "Qwen3.5-35B-A3B": "Qwen3.5-35B-A3B",
        "Qwen3.6-35B-A3B": "Qwen3.6-35B-A3B",
        "GLM-Z1-9B-0414": "GLM-Z1-9B",
        "GLM-4-9B-0414": "GLM-4-9B",
        "DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-Qwen3-8B",
    }
    return mapping.get(model, model)


def fmt_delta(value: float) -> str:
    return f"{value:+.3f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    model_groups = [(p, m, s) for p, m, s in MODEL_GROUPS if (p, m) in INCLUDED_MODELS]
    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}

    for provider, model, slugs in model_groups:
        for method in METHODS:
            print(f"Scoring {display_model(model)} / {method}", flush=True)
            rows = read_union([ROOT / path for path in paths_for("safebench", method, slugs)], target=4000)
            if len(rows) < 4000:
                raise RuntimeError(f"{provider}/{model}/{method} has only {len(rows)} rows")
            summaries[(provider, model, method)] = safety_summary(rows)

    table_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for provider, model, _ in model_groups:
        full = summaries[(provider, model, "safe")]
        full_scss = float(full["SCSS"])
        row = {
            "Model": display_model(model),
            "Full Safe SCSS": f"{full_scss:.3f}",
            "w/o MIG": fmt_delta(float(summaries[(provider, model, "safe_no_mig")]["SCSS"]) - full_scss),
            "w/o UEF": fmt_delta(float(summaries[(provider, model, "safe_no_uef")]["SCSS"]) - full_scss),
            "w/o GCV": fmt_delta(float(summaries[(provider, model, "safe_no_gcv")]["SCSS"]) - full_scss),
            "w/o HRC": fmt_delta(float(summaries[(provider, model, "safe_no_hrc")]["SCSS"]) - full_scss),
        }
        table_rows.append(row)
        for method in METHODS:
            summary = summaries[(provider, model, method)]
            raw = {
                "provider": provider,
                "model": model,
                "display_model": display_model(model),
                "method": method,
                **summary,
            }
            for key, value in list(raw.items()):
                if isinstance(value, float):
                    raw[key] = round_float(value)
            raw_rows.append(raw)

    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    table_rows.sort(key=lambda row: order.get(row["Model"], 999))
    raw_rows.sort(key=lambda row: (order.get(row["display_model"], 999), row["method"]))

    means = {"Model": "Mean", "Full Safe SCSS": f"{sum(float(r['Full Safe SCSS']) for r in table_rows) / len(table_rows):.3f}"}
    for key in ["w/o MIG", "w/o UEF", "w/o GCV", "w/o HRC"]:
        means[key] = fmt_delta(sum(float(r[key]) for r in table_rows) / len(table_rows))
    table_rows.append(means)

    write_csv(OUT_DIR / "whitelist12_ablation_model_matrix.csv", table_rows)
    write_csv(OUT_DIR / "whitelist12_ablation_raw_safety.csv", raw_rows)
    print(f"Wrote {OUT_DIR / 'whitelist12_ablation_model_matrix.csv'}")


if __name__ == "__main__":
    main()
