"""Build paper leaderboard data for the explicitly included completed models."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_leaderboard_data import (  # noqa: E402
    ABLATION_METHODS,
    MAIN_METHODS,
    mean,
    original_summary,
    read_union,
    round_float,
    safety_summary,
    write_csv,
)
from scripts.update_new_models_progress import (  # noqa: E402
    MODEL_GROUPS,
    ORIGINAL_METHODS,
    SAFEBENCH_METHODS,
    paths_for,
)


OUT_DIR = ROOT / "paper" / "generated"
PROGRESS_CSV = OUT_DIR / "new_models_progress.csv"
METHOD_MODE = os.environ.get("WHITELIST_METHOD_SET", "all").strip().lower()
SELECTED_SAFEBENCH_METHODS = MAIN_METHODS if METHOD_MODE == "main" else SAFEBENCH_METHODS

INCLUDED_MODELS = {
    ("agnes", "agnes-2.0-flash"),
    ("dmxapi", "deepseek-v4-flash"),
    ("mimo", "mimo-v2.5"),
    ("nvidia", "openai/gpt-oss-120b"),
    ("nvidia", "llama-4-maverick-17b-128e-instruct"),
    ("stepfun", "step-3.5-flash"),
    ("stepfun+avi", "step-3.7-flash"),
    ("xfyun", "Qwen3.5-2B"),
    ("xfyun", "Qwen3.5-35B-A3B"),
    ("xfyun", "Qwen3.6-35B-A3B"),
    ("siliconflow", "GLM-Z1-9B-0414"),
    ("siliconflow", "GLM-4-9B-0414"),
    ("siliconflow", "DeepSeek-R1-0528-Qwen3-8B"),
}


def progress_index() -> dict[tuple[str, str, str, str], tuple[int, int]]:
    index: dict[tuple[str, str, str, str], tuple[int, int]] = {}
    if not PROGRESS_CSV.exists():
        return index
    with PROGRESS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ok = int(row["ok_rows"])
                target = int(row["target_rows"])
            except (TypeError, ValueError):
                continue
            index[(row["provider"], row["model"], row["method"], row["task"])] = (ok, target)
    return index


def pctile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def ceiling_rows(combined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("pass_overall", "overall_safety_score"),
        ("SCSS", "SCSS"),
        ("clean_utility", "clean_utility_score"),
        ("paper_ready_score", "paper_ready_score"),
    ]
    rows = []
    for metric, key in specs:
        values = [float(row[key]) for row in combined_rows if row.get(key) not in ("", None)]
        rows.append(
            {
                "metric": metric,
                "n": len(values),
                "mean": round_float(mean(values)),
                "median": round_float(pctile(values, 0.5)),
                "min": round_float(min(values) if values else math.nan),
                "max": round_float(max(values) if values else math.nan),
                "rows_ge_0_95": sum(v >= 0.95 for v in values),
                "rows_ge_0_99": sum(v >= 0.99 for v in values),
            }
        )
    gaps = [
        float(row["overall_safety_score"]) - float(row["SCSS"])
        for row in combined_rows
        if row.get("overall_safety_score") not in ("", None) and row.get("SCSS") not in ("", None)
    ]
    rows.append(
        {
            "metric": "overall_minus_SCSS",
            "n": len(gaps),
            "mean": round_float(mean(gaps)),
            "median": round_float(pctile(gaps, 0.5)),
            "min": round_float(min(gaps) if gaps else math.nan),
            "max": round_float(max(gaps) if gaps else math.nan),
            "rows_ge_0_95": "",
            "rows_ge_0_99": "",
        }
    )
    return rows


def build() -> dict[str, Any]:
    print(f"Method mode: {METHOD_MODE}; SafeBench methods: {', '.join(SELECTED_SAFEBENCH_METHODS)}", flush=True)
    progress = progress_index()
    model_groups = [(p, m, s) for p, m, s in MODEL_GROUPS if (p, m) in INCLUDED_MODELS]
    missing_groups = sorted(INCLUDED_MODELS.difference({(p, m) for p, m, _ in model_groups}))
    if missing_groups:
        raise RuntimeError(f"Included models missing from MODEL_GROUPS: {missing_groups}")

    safety_rows: list[dict[str, Any]] = []
    original_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []

    for provider, model, slugs in model_groups:
        for method in SELECTED_SAFEBENCH_METHODS:
            ok, target = progress.get((provider, model, method, "safebench"), (0, 4000))
            print(f"SafeBench {provider}/{model}/{method}: {ok}/{target}", flush=True)
            progress_rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "method": method,
                    "task": "safebench",
                    "ok_rows": ok,
                    "target_rows": target,
                    "complete": int(ok >= target),
                }
            )
            if ok < target:
                continue
            rows = read_union([ROOT / path for path in paths_for("safebench", method, slugs)], target=target)
            summary = safety_summary(rows)
            row = {"provider": provider, "model": model, "method": method, **summary}
            row["safebench_complete"] = int(summary["safebench_num_samples"] >= target)
            safety_rows.append(row)

        for method in ORIGINAL_METHODS:
            ok, target = progress.get((provider, model, method, "original"), (0, 1000))
            print(f"Original {provider}/{model}/{method}: {ok}/{target}", flush=True)
            progress_rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "method": method,
                    "task": "original",
                    "ok_rows": ok,
                    "target_rows": target,
                    "complete": int(ok >= target),
                }
            )
            if ok < target:
                continue
            rows = read_union([ROOT / path for path in paths_for("original", method, slugs)], target=target)
            summary = original_summary(rows)
            row = {"provider": provider, "model": model, "method": method, **summary}
            row["original_complete"] = int(summary["original_num_cases"] >= target)
            original_rows.append(row)

    original_index = {(r["provider"], r["model"], r["method"]): r for r in original_rows}
    for srow in safety_rows:
        if srow["method"] not in MAIN_METHODS:
            continue
        orow = original_index.get((srow["provider"], srow["model"], srow["method"]))
        if not orow:
            continue
        clean_utility = mean([float(orow.get("TNM_acc", 0.0)), float(orow.get("treatment_token_f1", 0.0))])
        combined = {
            **srow,
            **{key: value for key, value in orow.items() if key not in {"provider", "model", "method"}},
        }
        combined["clean_utility_score"] = clean_utility
        combined["complete"] = int(
            combined.get("safebench_num_samples", 0) >= 4000 and combined.get("original_num_cases", 0) >= 1000
        )
        combined["paper_ready_score"] = mean([float(combined.get("SCSS", 0.0)), float(clean_utility)])
        combined_rows.append(combined)

    for rows in (safety_rows, original_rows, combined_rows):
        for row in rows:
            for key, value in list(row.items()):
                if isinstance(value, float):
                    row[key] = round_float(value)

    safety_rows.sort(key=lambda r: (r["provider"], r["model"], r["method"]))
    original_rows.sort(key=lambda r: (r["provider"], r["model"], r["method"]))
    combined_rows.sort(key=lambda r: (-float(r["paper_ready_score"]), r["provider"], r["model"], r["method"]))
    progress_rows.sort(key=lambda r: (r["provider"], r["model"], r["task"], r["method"]))

    ceiling = ceiling_rows(combined_rows)
    complete_progress = [r for r in progress_rows if r["complete"]]
    payload = {
        "included_models": [{"provider": p, "model": m} for p, m in sorted(INCLUDED_MODELS)],
        "summary": {
            "included_model_count": len(INCLUDED_MODELS),
            "complete_progress_rows": len(complete_progress),
            "method_mode": METHOD_MODE,
            "expected_progress_rows": len(INCLUDED_MODELS) * (len(SELECTED_SAFEBENCH_METHODS) + len(ORIGINAL_METHODS)),
            "safety_rows": len(safety_rows),
            "original_rows": len(original_rows),
            "combined_main_rows": len(combined_rows),
            "main_safebench_outputs": len(combined_rows) * 4000,
            "main_original_outputs": len(combined_rows) * 1000,
            "ablation_safety_rows": len([r for r in safety_rows if r["method"] in ABLATION_METHODS]),
        },
        "ceiling": ceiling,
        "top_combined": combined_rows[:20],
        "progress": progress_rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "leaderboard_whitelist12_safety.csv", safety_rows)
    write_csv(OUT_DIR / "leaderboard_whitelist12_original.csv", original_rows)
    write_csv(OUT_DIR / "leaderboard_whitelist12_combined.csv", combined_rows)
    write_csv(OUT_DIR / "strict_vs_pass_ceiling_effect_summary_whitelist12.csv", ceiling)
    write_csv(OUT_DIR / "whitelist12_progress_check.csv", progress_rows)
    (OUT_DIR / "leaderboard_whitelist12_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    payload = build()
    summary = payload["summary"]
    print(
        "Wrote whitelist leaderboard: "
        f"{summary['included_model_count']} models, "
        f"{summary['combined_main_rows']} main rows, "
        f"{summary['safety_rows']} SafeBench rows, "
        f"{summary['original_rows']} original rows."
    )


if __name__ == "__main__":
    main()
